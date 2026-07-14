import gc
import json
import time
from pathlib import Path

import optuna
import torch
from optuna.samplers import TPESampler
from transformers import AutoConfig, AutoTokenizer

from src.trainer import Trainer


class Tuner:
    def __init__(
        self,
        model_path: str,
        data_path: str,
        epochs: int,
        n_trials: int,
        learning_rates: list[float],
        batch_sizes: list[int],
        max_lengths: list[int],
        weight_decays: list[float],
        drop_path_rates: list[float],
        label_smoothing_epsilons: list[float],
        objective_loss: str = "label_smoothing",
        metric: str = "accuracy",
        direction: str = "maximize",
        cache_dir: str | None = None,
        num_labels: int = 2,
        dropout: float = 0.1,
        id2label: dict | None = None,
        label2id: dict | None = None,
        trust_remote_code: bool = False,
        num_workers: int = 4,
        gamma: float = 2.0,
        alpha: float | None = None,
        seed: int = 42,
        outputs_path: str = "outputs",
        model_name: str | None = None,
    ):
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

        self.model_path = model_path
        self.data_path = data_path
        self.epochs = epochs
        self.n_trials = n_trials

        self.learning_rates = learning_rates
        self.batch_sizes = batch_sizes
        self.max_lengths = max_lengths
        self.weight_decays = weight_decays
        self.drop_path_rates = drop_path_rates
        self.label_smoothing_epsilons = label_smoothing_epsilons

        self.objective_loss = objective_loss
        self.metric = metric
        self.direction = direction

        self.cache_dir = cache_dir
        self.num_labels = num_labels
        self.dropout = dropout
        self.id2label = id2label
        self.label2id = label2id
        self.trust_remote_code = trust_remote_code
        self.num_workers = num_workers
        self.gamma = gamma
        self.alpha = alpha
        self.seed = seed
        self.outputs_path = Path(outputs_path)

        config = AutoConfig.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            use_fast=True,
        )

        if model_name is None:
            model_name = str(config.model_type).lower()

        self.model_name = model_name

        possible_limits = []
        tokenizer_limit = int(tokenizer.model_max_length)

        if tokenizer_limit < 1_000_000:
            possible_limits.append(tokenizer_limit)

        config_limit = getattr(config, "max_position_embeddings", None)

        if config_limit is not None:
            possible_limits.append(int(config_limit))

        self.native_max_length = (
            min(possible_limits)
            if possible_limits
            else max(max_lengths)
        )

    def tune(self) -> dict:
        metric_key = (
            self.metric
            if self.metric.startswith("eval_")
            else f"eval_{self.metric}"
        )

        def objective(trial):
            learning_rate = trial.suggest_categorical(
                "learning_rate",
                self.learning_rates,
            )
            batch_size = trial.suggest_categorical(
                "batch_size",
                self.batch_sizes,
            )
            max_length = trial.suggest_categorical(
                "max_length",
                self.max_lengths,
            )

            if len(set(self.weight_decays)) == 1:
                weight_decay = trial.suggest_categorical(
                    "weight_decay",
                    self.weight_decays,
                )
            else:
                weight_decay = trial.suggest_float(
                    "weight_decay",
                    min(self.weight_decays),
                    max(self.weight_decays),
                )

            drop_path_rate = trial.suggest_categorical(
                "drop_path_rate",
                self.drop_path_rates,
            )
            label_smoothing_epsilon = trial.suggest_categorical(
                "label_smoothing_epsilon",
                self.label_smoothing_epsilons,
            )

            if max_length > self.native_max_length:
                trial.set_user_attr(
                    "pruned_reason",
                    f"max_length={max_length} exceeds the native "
                    f"tokenizer/model limit of {self.native_max_length} "
                    f"for {self.model_name}.",
                )
                raise optuna.TrialPruned()

            start_time = time.perf_counter()

            trainer = Trainer(
                model_path=self.model_path,
                data_path=self.data_path,
                loss_key=self.objective_loss,
                epochs=self.epochs,
                cache_dir=self.cache_dir,
                num_labels=self.num_labels,
                drop_path_rate=drop_path_rate,
                dropout=self.dropout,
                id2label=self.id2label,
                label2id=self.label2id,
                trust_remote_code=self.trust_remote_code,
                max_length=max_length,
                batch_size=batch_size,
                num_workers=self.num_workers,
                weight_decay=weight_decay,
                lr=learning_rate,
                label_smoothing=label_smoothing_epsilon,
                gamma=self.gamma,
                alpha=self.alpha,
                seed=self.seed,
                outputs_path=str(self.outputs_path),
                model_name=self.model_name,
                save_checkpoint=False,
            )

            try:
                history = trainer.train()
                metric_values = history[metric_key]

                if self.direction == "maximize":
                    score = max(metric_values)
                else:
                    score = min(metric_values)

                trial.set_user_attr(
                    "duration_seconds",
                    time.perf_counter() - start_time,
                )

                return float(score)

            finally:
                del trainer
                gc.collect()

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        sampler = TPESampler(
            seed=self.seed,
            n_startup_trials=min(2, self.n_trials),
        )

        study = optuna.create_study(
            direction=self.direction,
            sampler=sampler,
        )

        study.optimize(
            objective,
            n_trials=self.n_trials,
            n_jobs=1,
            gc_after_trial=True,
            show_progress_bar=True,
        )

        trials = []

        for trial in study.trials:
            trials.append(
                {
                    "number": trial.number,
                    "state": trial.state.name,
                    "value": trial.value,
                    "params": trial.params,
                    "user_attrs": trial.user_attrs,
                }
            )

        tuning_path = (
            self.outputs_path
            / "tuning"
            / self.model_name
            / "trials.json"
        )
        tuning_path.parent.mkdir(parents=True, exist_ok=True)

        with tuning_path.open("w", encoding="utf-8") as file:
            json.dump(
                trials,
                file,
                indent=2,
                ensure_ascii=False,
            )

        completed_trials = [
            trial
            for trial in study.trials
            if trial.state == optuna.trial.TrialState.COMPLETE
        ]

        if not completed_trials:
            print("Tuning completed, but no trial was completed.")
            print(f"Trials saved to {tuning_path}")
            return {}

        best_parameters = study.best_params

        print("\nTuning completed")
        print(f"Best {self.metric}: {study.best_value:.4f}")
        print("\nBest hyperparameters:")

        for name, value in best_parameters.items():
            print(f"{name}: {value}")

        print(f"Trials saved to {tuning_path}")

        return best_parameters
