import json
import math
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)
from torch.nn import functional as F


class Calibrator:
    def __init__(
        self,
        model,
        validation_loader,
        model_name: str,
        loss_key: str,
        max_iter: int = 200,
        lr: float = 0.05,
        confidence_thresholds: list[float] | None = None,
        coverage_levels: list[float] | None = None,
        device: str | torch.device | None = None,
        ece_bins: int = 15,
        selective_threshold: float = 0.90,
        outputs_path: str = "outputs",
        checkpoint_path: str | None = None,
        training_time_seconds: float = 0.0,
    ):
        self.model = model
        self.validation_loader = validation_loader
        self.model_name = model_name.lower().strip()
        self.loss_key = loss_key.lower().strip()
        self.max_iter = max_iter
        self.lr = lr
        self.ece_bins = ece_bins
        self.selective_threshold = selective_threshold
        self.outputs_path = Path(outputs_path)
        self.training_time_seconds = training_time_seconds

        self.confidence_thresholds = confidence_thresholds or [
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
        ]

        self.coverage_levels = coverage_levels or [
            1.00,
            0.95,
            0.90,
            0.80,
            0.70,
            0.60,
            0.50,
        ]

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)
        self.model = self.model.to(self.device)
        self.temperature = 1.0
        self.calibration_results = None

        if checkpoint_path is None:
            checkpoint_path = (
                self.outputs_path
                / "checkpoints"
                / self.model_name
                / self.loss_key
                / "best_model"
            )

        self.checkpoint_path = Path(checkpoint_path)

    def collect_logits(self, data_loader):
        self.model.eval()

        all_logits = []
        all_labels = []

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        with torch.inference_mode():
            for batch in data_loader:
                labels = batch["labels"].to(self.device)

                inputs = {
                    key: value.to(self.device)
                    for key, value in batch.items()
                    if key != "labels"
                }

                logits = self.model(**inputs).logits

                all_logits.append(logits.detach().cpu())
                all_labels.append(labels.detach().cpu())

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        elapsed_seconds = time.perf_counter() - start_time

        return (
            torch.cat(all_logits, dim=0),
            torch.cat(all_labels, dim=0).long(),
            elapsed_seconds,
        )

    def expected_calibration_error(
        self,
        probabilities: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        confidences, predictions = probabilities.max(dim=1)
        accuracies = predictions.eq(labels)
        bin_edges = torch.linspace(0.0, 1.0, self.ece_bins + 1)
        ece = torch.tensor(0.0)

        for index in range(self.ece_bins):
            lower = bin_edges[index]
            upper = bin_edges[index + 1]

            if index == 0:
                in_bin = (
                    (confidences >= lower)
                    & (confidences <= upper)
                )
            else:
                in_bin = (
                    (confidences > lower)
                    & (confidences <= upper)
                )

            if in_bin.any():
                bin_accuracy = accuracies[in_bin].float().mean()
                bin_confidence = confidences[in_bin].mean()
                bin_weight = in_bin.float().mean()
                ece += abs(bin_accuracy - bin_confidence) * bin_weight

        return float(ece.item())

    def calibrate(self) -> dict:
        logits, labels, _ = self.collect_logits(
            self.validation_loader
        )
        logits = logits.double()

        nll_before = float(F.cross_entropy(logits, labels).item())
        log_temperature = torch.nn.Parameter(
            torch.zeros(1, dtype=torch.float64)
        )
        optimizer = torch.optim.Adam(
            [log_temperature],
            lr=self.lr,
        )

        minimum_log_temperature = math.log(1e-3)
        maximum_log_temperature = math.log(1e3)
        best_temperature = 1.0
        best_nll = nll_before

        for _ in range(self.max_iter):
            optimizer.zero_grad()

            temperature = log_temperature.exp()
            loss = F.cross_entropy(
                logits / temperature,
                labels,
            )

            loss.backward()
            optimizer.step()

            with torch.no_grad():
                log_temperature.clamp_(
                    minimum_log_temperature,
                    maximum_log_temperature,
                )

                current_temperature = float(
                    log_temperature.exp().item()
                )
                current_nll = float(
                    F.cross_entropy(
                        logits / current_temperature,
                        labels,
                    ).item()
                )

                if current_nll < best_nll:
                    best_nll = current_nll
                    best_temperature = current_temperature

        self.temperature = best_temperature
        self.calibration_results = {
            "model": self.model_name,
            "loss": self.loss_key,
            "temperature": self.temperature,
            "validation_nll_before": nll_before,
            "validation_nll_after": best_nll,
            "validation_samples": int(labels.size(0)),
            "checkpoint_path": self.checkpoint_path.as_posix(),
        }

        print("Calibration completed")
        print(f"Temperature: {self.temperature:.4f}")
        print(f"Validation NLL: {nll_before:.4f} -> {best_nll:.4f}")

        return self.calibration_results

    def _evaluate_temperature(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        temperature: float,
        calibration_name: str,
        latency_ms_per_sample: float,
    ):
        probabilities = torch.softmax(
            logits / temperature,
            dim=1,
        )
        confidences, predictions = probabilities.max(dim=1)
        correct = predictions.eq(labels)

        labels_list = labels.tolist()
        predictions_list = predictions.tolist()

        tn, fp, fn, tp = confusion_matrix(
            labels_list,
            predictions_list,
            labels=[0, 1],
        ).ravel()

        false_negative_rate = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0.0
        )

        sorted_indices = torch.argsort(
            confidences,
            descending=True,
        )
        sorted_errors = (~correct[sorted_indices]).float()
        cumulative_risk = sorted_errors.cumsum(dim=0) / torch.arange(
            1,
            labels.size(0) + 1,
            dtype=torch.float32,
        )
        risk_coverage_auc = float(cumulative_risk.mean().item())

        selected = confidences >= self.selective_threshold
        selected_count = int(selected.sum().item())

        if selected_count == 0:
            selective_accuracy = 0.0
            selective_f1 = 0.0
            high_confidence_error_rate = 0.0
        else:
            selected_labels = labels[selected].tolist()
            selected_predictions = predictions[selected].tolist()
            selective_accuracy = accuracy_score(
                selected_labels,
                selected_predictions,
            )
            selective_f1 = f1_score(
                selected_labels,
                selected_predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
            high_confidence_error_rate = 1.0 - selective_accuracy

        num_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
        )
        trainable_params = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

        result = {
            "model": self.model_name,
            "loss": self.loss_key,
            "calibration": calibration_name,
            "accuracy": accuracy_score(
                labels_list,
                predictions_list,
            ),
            "precision": precision_score(
                labels_list,
                predictions_list,
                average="binary",
                pos_label=1,
                zero_division=0,
            ),
            "recall": recall_score(
                labels_list,
                predictions_list,
                average="binary",
                pos_label=1,
                zero_division=0,
            ),
            "f1": f1_score(
                labels_list,
                predictions_list,
                average="binary",
                pos_label=1,
                zero_division=0,
            ),
            "mcc": matthews_corrcoef(
                labels_list,
                predictions_list,
            ),
            "cohen_kappa": cohen_kappa_score(
                labels_list,
                predictions_list,
            ),
            "false_negative_rate": false_negative_rate,
            "ece": self.expected_calibration_error(
                probabilities,
                labels,
            ),
            "brier_score": float(
                ((probabilities[:, 1] - labels.float()) ** 2)
                .mean()
                .item()
            ),
            "nll": float(
                F.cross_entropy(
                    logits / temperature,
                    labels,
                ).item()
            ),
            "high_confidence_error_rate": high_confidence_error_rate,
            "high_confidence_count": selected_count,
            "selective_threshold": self.selective_threshold,
            "selective_accuracy": selective_accuracy,
            "selective_f1": selective_f1,
            "coverage": selected_count / labels.size(0),
            "risk_coverage_auc": risk_coverage_auc,
            "latency_ms_per_sample": latency_ms_per_sample,
            "num_params": num_params,
            "trainable_params": trainable_params,
            "training_time_seconds": self.training_time_seconds,
            "temperature": temperature,
            "checkpoint_path": self.checkpoint_path.as_posix(),
        }

        selective_results = []

        for threshold in sorted(
            self.confidence_thresholds,
            reverse=True,
        ):
            accepted = confidences >= threshold
            accepted_count = int(accepted.sum().item())

            if accepted_count == 0:
                selective_accuracy_at_value = 0.0
                selective_f1_at_value = 0.0
                risk = 0.0
            else:
                accepted_labels = labels[accepted].tolist()
                accepted_predictions = predictions[accepted].tolist()
                selective_accuracy_at_value = accuracy_score(
                    accepted_labels,
                    accepted_predictions,
                )
                selective_f1_at_value = f1_score(
                    accepted_labels,
                    accepted_predictions,
                    average="binary",
                    pos_label=1,
                    zero_division=0,
                )
                risk = 1.0 - selective_accuracy_at_value

            selective_results.append(
                {
                    "selection_type": "confidence_threshold",
                    "selection_value": threshold,
                    "coverage": accepted_count / labels.size(0),
                    "selective_accuracy": selective_accuracy_at_value,
                    "selective_f1": selective_f1_at_value,
                    "accepted_count": accepted_count,
                    "rejected_count": labels.size(0) - accepted_count,
                    "risk": risk,
                    "model": self.model_name,
                    "loss": self.loss_key,
                    "calibration": calibration_name,
                    "risk_coverage_auc": risk_coverage_auc,
                }
            )

        for coverage_level in self.coverage_levels:
            accepted_count = max(
                1,
                math.ceil(labels.size(0) * coverage_level),
            )
            accepted_indices = sorted_indices[:accepted_count]
            accepted_labels = labels[accepted_indices].tolist()
            accepted_predictions = predictions[accepted_indices].tolist()

            selective_accuracy_at_value = accuracy_score(
                accepted_labels,
                accepted_predictions,
            )
            selective_f1_at_value = f1_score(
                accepted_labels,
                accepted_predictions,
                average="binary",
                pos_label=1,
                zero_division=0,
            )

            selective_results.append(
                {
                    "selection_type": "coverage_level",
                    "selection_value": coverage_level,
                    "coverage": accepted_count / labels.size(0),
                    "selective_accuracy": selective_accuracy_at_value,
                    "selective_f1": selective_f1_at_value,
                    "accepted_count": accepted_count,
                    "rejected_count": labels.size(0) - accepted_count,
                    "risk": 1.0 - selective_accuracy_at_value,
                    "model": self.model_name,
                    "loss": self.loss_key,
                    "calibration": calibration_name,
                    "risk_coverage_auc": risk_coverage_auc,
                }
            )

        return result, selective_results

    def run(self, test_loader) -> dict:
        def update_json(path: Path, new_items: list, key_names: tuple):
            path.parent.mkdir(parents=True, exist_ok=True)

            if path.exists():
                with path.open("r", encoding="utf-8") as file:
                    existing_items = json.load(file)
            else:
                existing_items = []

            new_keys = {
                tuple(item[key] for key in key_names)
                for item in new_items
            }

            existing_items = [
                item
                for item in existing_items
                if tuple(item[key] for key in key_names) not in new_keys
            ]
            existing_items.extend(new_items)

            with path.open("w", encoding="utf-8") as file:
                json.dump(
                    existing_items,
                    file,
                    indent=2,
                    ensure_ascii=False,
                )

        calibration_results = self.calibrate()
        test_logits, test_labels, elapsed_seconds = self.collect_logits(
            test_loader
        )
        latency_ms_per_sample = (
            elapsed_seconds * 1000.0 / test_labels.size(0)
        )

        uncalibrated_result, uncalibrated_selective = (
            self._evaluate_temperature(
                logits=test_logits,
                labels=test_labels,
                temperature=1.0,
                calibration_name="uncalibrated",
                latency_ms_per_sample=latency_ms_per_sample,
            )
        )

        calibrated_result, calibrated_selective = (
            self._evaluate_temperature(
                logits=test_logits,
                labels=test_labels,
                temperature=self.temperature,
                calibration_name="calibrated",
                latency_ms_per_sample=latency_ms_per_sample,
            )
        )

        temperature_path = (
            self.outputs_path
            / "calibration"
            / self.model_name
            / self.loss_key
            / "temperature.json"
        )
        temperature_path.parent.mkdir(parents=True, exist_ok=True)

        with temperature_path.open("w", encoding="utf-8") as file:
            json.dump(
                calibration_results,
                file,
                indent=2,
                ensure_ascii=False,
            )

        results = [
            uncalibrated_result,
            calibrated_result,
        ]
        selective_results = (
            uncalibrated_selective
            + calibrated_selective
        )

        results_path = self.outputs_path / "results" / "results.json"
        selective_path = (
            self.outputs_path
            / "results"
            / "selective_results.json"
        )

        update_json(
            results_path,
            results,
            ("model", "loss", "calibration"),
        )
        update_json(
            selective_path,
            selective_results,
            (
                "model",
                "loss",
                "calibration",
                "selection_type",
                "selection_value",
            ),
        )

        print(f"Temperature saved to {temperature_path}")
        print(f"Results saved to {results_path}")
        print(f"Selective results saved to {selective_path}")

        return {
            "temperature": calibration_results,
            "results": results,
            "selective_results": selective_results,
        }
