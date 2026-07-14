from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class AppSettings:
    classifier_threshold: float
    ner_threshold: float
    temperature: float
    max_chars: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "classifier_threshold": self.classifier_threshold,
            "ner_threshold": self.ner_threshold,
            "temperature": self.temperature,
            "max_chars": self.max_chars,
        }


def load_config(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    config_path = project_root / "config.yaml"
    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


def load_settings(project_root: Path = PROJECT_ROOT) -> AppSettings:
    config = load_config(project_root)
    classifier = config.get("classifier_inference", {})
    ner = config.get("ner_inference", {})
    generation = config.get("reasoner_inference", {}).get("generation", {})

    return AppSettings(
        classifier_threshold=float(
            classifier.get(
                "confidence_threshold",
                config.get("evaluation", {}).get("high_confidence_threshold", 0.90),
            )
        ),
        ner_threshold=float(ner.get("threshold", 0.50)),
        temperature=float(generation.get("temperature", 0.70)),
        max_chars=int(config.get("data", {}).get("max_chars", 12_000)),
    )

