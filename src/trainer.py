import gc
import time
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification

from src.dataset import MyDataset, dataloader
from src.loss import get_loss_fn
from src.model import BERT
from src.utils import set_seed


class Trainer:
    def __init__(
        self,
        model_path: str,
        data_path: str,
        loss_key: str,
        epochs: int,
        cache_dir: str | None = None,
        num_labels: int = 2,
        drop_path_rate: float = 0.1,
        dropout: float = 0.1,
        id2label: dict | None = None,
        label2id: dict | None = None,
        trust_remote_code: bool = False,
        max_length: int = 512,
        batch_size: int = 32,
        num_workers: int = 4,
        weight_decay: float = 0.1,
        lr: float = 3e-5,
        label_smoothing: float = 0.1,
        gamma: float = 2.0,
        alpha: float | None = None,
        seed: int = 42,
        outputs_path: str = "outputs",
        model_name: str | None = None,
        save_checkpoint: bool = True,
    ):
        set_seed(seed)

        if id2label is None:
            id2label = {
                0: "non-deceptive",
                1: "deceptive",
            }

        if label2id is None:
            label2id = {
                "non-deceptive": 0,
                "deceptive": 1,
            }

        self.data_path = Path(data_path)
        self.outputs_path = Path(outputs_path)
        self.loss_key = loss_key.lower().strip()
        self.epochs = epochs
        self.num_labels = num_labels
        self.trust_remote_code = trust_remote_code
        self.save_checkpoint = save_checkpoint

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        encoder = BERT(
            model_path=model_path,
            cache_dir=cache_dir,
            drop_path_rate=drop_path_rate,
            dropout=dropout,
            id2label=id2label,
            label2id=label2id,
            num_labels=num_labels,
            trust_remote_code=trust_remote_code,
        )

        self.model = encoder.model.to(self.device)
        self.tokenizer = encoder.tokenizer

        if model_name is None:
            model_name = str(self.model.config.model_type).lower()

        self.model_name = model_name
        self.checkpoint_path = (
            self.outputs_path
            / "checkpoints"
            / self.model_name
            / self.loss_key
            / "best_model"
        )

        train_dataset = MyDataset(
            tokenizer=self.tokenizer,
            jsonl_path=self.data_path / "train.jsonl",
            max_length=max_length,
        )

        validation_dataset = MyDataset(
            tokenizer=self.tokenizer,
            jsonl_path=self.data_path / "validation.jsonl",
            max_length=max_length,
        )

        test_dataset = MyDataset(
            tokenizer=self.tokenizer,
            jsonl_path=self.data_path / "test.jsonl",
            max_length=max_length,
        )

        self.train_loader = dataloader(
            dataset=train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
        )

        self.val_loader = dataloader(
            dataset=validation_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        self.test_loader = dataloader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,
        )

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        self.criterion = get_loss_fn(
            loss_key=self.loss_key,
            alpha=alpha,
            gamma=gamma,
            label_smoothing=label_smoothing,
        )

        if isinstance(self.criterion, nn.Module):
            self.criterion = self.criterion.to(self.device)

        self.training_time_seconds = 0.0
        self.best_validation_accuracy = 0.0

        self.history = {
            "epoch": [],
            "loss": [],
            "grad_norm": [],
            "learning_rate": [],
            "eval_loss": [],
            "eval_accuracy": [],
            "eval_precision": [],
            "eval_recall": [],
            "eval_f1": [],
            "eval_mcc": [],
            "eval_cohen_kappa": [],
            "eval_false_negative_rate": [],
        }

    def _train_epoch(self) -> dict:
        self.model.train()

        total_loss = 0.0
        total_grad_norm = 0.0
        total_samples = 0
        total_batches = 0

        for batch in self.train_loader:
            labels = batch["labels"].to(self.device)

            inputs = {
                key: value.to(self.device)
                for key, value in batch.items()
                if key != "labels"
            }

            self.optimizer.zero_grad(set_to_none=True)

            logits = self.model(**inputs).logits
            loss = self.criterion(logits, labels)

            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float("inf"),
            ).item()

            self.optimizer.step()

            current_batch_size = labels.size(0)
            total_loss += loss.item() * current_batch_size
            total_grad_norm += grad_norm
            total_samples += current_batch_size
            total_batches += 1

        return {
            "loss": total_loss / total_samples,
            "grad_norm": total_grad_norm / total_batches,
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }

    def evaluate(self, data_loader) -> dict:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        all_labels = []
        all_predictions = []

        with torch.inference_mode():
            for batch in data_loader:
                labels = batch["labels"].to(self.device)

                inputs = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                    if key != "labels"
                }

                logits = self.model(**inputs).logits
                loss = self.criterion(logits, labels)
                predictions = torch.argmax(logits, dim=1)

                current_batch_size = labels.size(0)
                total_loss += loss.item() * current_batch_size
                total_samples += current_batch_size

                all_labels.extend(labels.cpu().tolist())
                all_predictions.extend(predictions.cpu().tolist())

        tn, fp, fn, tp = confusion_matrix(
            all_labels,
            all_predictions,
            labels=[0, 1],
        ).ravel()

        false_negative_rate = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0.0
        )

        return {
            "eval_loss": total_loss / total_samples,
            "eval_accuracy": accuracy_score(
                all_labels,
                all_predictions,
            ),
            "eval_precision": precision_score(
                all_labels,
                all_predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            ),
            "eval_recall": recall_score(
                all_labels,
                all_predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            ),
            "eval_f1": f1_score(
                all_labels,
                all_predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            ),
            "eval_mcc": matthews_corrcoef(
                all_labels,
                all_predictions,
            ),
            "eval_cohen_kappa": cohen_kappa_score(
                all_labels,
                all_predictions,
            ),
            "eval_false_negative_rate": false_negative_rate,
        }

    def train(self) -> dict:
        start_time = time.perf_counter()
        best_accuracy = -1.0

        for epoch in range(1, self.epochs + 1):
            train_results = self._train_epoch()
            validation_results = self.evaluate(self.val_loader)

            self.history["epoch"].append(epoch)

            for metric_name, metric_value in train_results.items():
                self.history[metric_name].append(metric_value)

            for metric_name, metric_value in validation_results.items():
                self.history[metric_name].append(metric_value)

            current_accuracy = validation_results["eval_accuracy"]

            if self.save_checkpoint and current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.checkpoint_path.mkdir(parents=True, exist_ok=True)
                self.model.save_pretrained(self.checkpoint_path)
                self.tokenizer.save_pretrained(self.checkpoint_path)

            print(
                f"Epoch [{epoch}/{self.epochs}] | "
                f"Train Loss: {train_results['loss']:.4f} | "
                f"Grad Norm: {train_results['grad_norm']:.4f} | "
                f"LR: {train_results['learning_rate']:.8f} | "
                f"Val Loss: {validation_results['eval_loss']:.4f} | "
                f"Accuracy: {validation_results['eval_accuracy']:.4f} | "
                f"Precision: {validation_results['eval_precision']:.4f} | "
                f"Recall: {validation_results['eval_recall']:.4f} | "
                f"F1: {validation_results['eval_f1']:.4f} | "
                f"MCC: {validation_results['eval_mcc']:.4f} | "
                f"Kappa: {validation_results['eval_cohen_kappa']:.4f} | "
                f"FNR: {validation_results['eval_false_negative_rate']:.4f}"
            )

        self.training_time_seconds = time.perf_counter() - start_time
        self.best_validation_accuracy = max(self.history["eval_accuracy"])

        if self.save_checkpoint:
            del self.model
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.checkpoint_path,
                trust_remote_code=self.trust_remote_code,
                use_safetensors=True,
            ).to(self.device)

            print(f"Best model saved to {self.checkpoint_path}")

        return self.history
