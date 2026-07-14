import argparse
import gc
from pathlib import Path

import torch
import yaml

from src.calibrate import Calibrator
from src.preprocessing import preprocessing
from src.trainer import Trainer
from src.tuner import Tuner


def main(config_path: str):
    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    seed = config.get("seed", 42)
    output_dir = Path(config.get("output_dir", "outputs"))
    cache_dir = Path(config.get("cache_dir", "cache"))

    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    training_config = config["training"]
    tuning_config = config.get("tuning", {})
    calibration_config = config.get("calibration", {})
    selective_config = config.get("selective_prediction", {})
    evaluation_config = config.get("evaluation", {})

    if data_config.get("preprocess", True):
        preprocessing(
            root_path=data_config["root_dir"],
            max_chars=data_config.get("max_chars", 12000),
        )

    data_path = Path(
        data_config.get(
            "processed_dir",
            Path(data_config["root_dir"]) / "preprocessed",
        )
    )

    losses = training_config.get(
        "losses",
        ["ce", "label_smoothing", "focal"],
    )

    for model_name, model_config in config["models"].items():
        print("\n" + "=" * 70)
        print(f"Model: {model_name}")
        print("=" * 70)

        best_parameters = {}

        if tuning_config.get("enabled", False):
            search_space = tuning_config["search_space"]
            weight_decay_space = search_space["weight_decay"]

            if isinstance(weight_decay_space, dict):
                weight_decays = [
                    weight_decay_space["low"],
                    weight_decay_space["high"],
                ]
            else:
                weight_decays = weight_decay_space

            tuner = Tuner(
                model_path=model_config["model_name"],
                data_path=str(data_path),
                epochs=tuning_config.get("epochs", model_config["epochs"]),
                n_trials=tuning_config.get("n_trials", 5),
                learning_rates=search_space["learning_rate"],
                batch_sizes=search_space["batch_size"],
                max_lengths=search_space["max_length"],
                weight_decays=weight_decays,
                drop_path_rates=search_space["drop_path_rate"],
                label_smoothing_epsilons=(
                    search_space["label_smoothing_epsilon"]
                ),
                objective_loss=tuning_config.get(
                    "objective_loss",
                    "label_smoothing",
                ),
                metric=tuning_config.get("metric", "accuracy"),
                direction=tuning_config.get("direction", "maximize"),
                cache_dir=str(cache_dir),
                num_labels=training_config.get("num_labels", 2),
                dropout=model_config.get("dropout", 0.1),
                trust_remote_code=model_config.get(
                    "trust_remote_code",
                    False,
                ),
                num_workers=training_config.get("num_workers", 4),
                gamma=training_config.get("focal_gamma", 2.0),
                alpha=training_config.get("focal_alpha"),
                seed=seed,
                outputs_path=str(output_dir),
                model_name=model_name,
            )

            best_parameters = tuner.tune()
            del tuner
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        learning_rate = best_parameters.get(
            "learning_rate",
            model_config.get("learning_rate", 3e-5),
        )
        batch_size = best_parameters.get(
            "batch_size",
            model_config.get("batch_size", 16),
        )
        max_length = best_parameters.get(
            "max_length",
            model_config.get("max_length", 512),
        )
        weight_decay = best_parameters.get(
            "weight_decay",
            model_config.get("weight_decay", 0.01),
        )
        drop_path_rate = best_parameters.get(
            "drop_path_rate",
            model_config.get("drop_path_rate", 0.1),
        )
        label_smoothing = best_parameters.get(
            "label_smoothing_epsilon",
            training_config.get("label_smoothing_epsilon", 0.1),
        )

        for loss_key in losses:
            print("\n" + "-" * 70)
            print(f"Training {model_name} with {loss_key}")
            print("-" * 70)

            trainer = Trainer(
                model_path=model_config["model_name"],
                data_path=str(data_path),
                loss_key=loss_key,
                epochs=model_config["epochs"],
                cache_dir=str(cache_dir),
                num_labels=training_config.get("num_labels", 2),
                drop_path_rate=drop_path_rate,
                dropout=model_config.get("dropout", 0.1),
                trust_remote_code=model_config.get(
                    "trust_remote_code",
                    False,
                ),
                max_length=max_length,
                batch_size=batch_size,
                num_workers=training_config.get("num_workers", 4),
                weight_decay=weight_decay,
                lr=learning_rate,
                label_smoothing=label_smoothing,
                gamma=training_config.get("focal_gamma", 2.0),
                alpha=training_config.get("focal_alpha"),
                seed=seed,
                outputs_path=str(output_dir),
                model_name=model_name,
                save_checkpoint=True,
            )

            trainer.train()

            if calibration_config.get("enabled", True):
                calibrator = Calibrator(
                    model=trainer.model,
                    validation_loader=trainer.val_loader,
                    model_name=model_name,
                    loss_key=loss_key,
                    max_iter=calibration_config.get("max_iter", 200),
                    lr=calibration_config.get("lr", 0.05),
                    confidence_thresholds=selective_config.get(
                        "confidence_thresholds"
                    ),
                    coverage_levels=selective_config.get(
                        "coverage_levels"
                    ),
                    ece_bins=evaluation_config.get("ece_bins", 15),
                    selective_threshold=evaluation_config.get(
                        "high_confidence_threshold",
                        0.90,
                    ),
                    outputs_path=str(output_dir),
                    checkpoint_path=trainer.checkpoint_path,
                    training_time_seconds=trainer.training_time_seconds,
                )

                calibrator.run(trainer.test_loader)
                del calibrator

            del trainer
            gc.collect()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\nAll experiments completed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tune, train, calibrate, and evaluate all models."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    arguments = parser.parse_args()
    main(arguments.config)