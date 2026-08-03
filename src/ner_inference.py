from pathlib import Path

import torch
import yaml
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download
from src.utils import NER_LABELS


# Spans this short, or made only of these words, carry no evidence but do clutter
# the highlighted review.
STOPWORD_SPANS = frozenset(
    {
        "it", "i", "we", "he", "she", "they", "you", "us", "me", "my", "your", "our",
        "this", "that", "these", "those", "the", "a", "an", "and", "or", "but",
        "item", "product", "thing", "one",
    }
)


class NERInference:
    NER_LABELS = NER_LABELS

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        inference_config = config["ner_inference"]

        self.model_name = inference_config["model_name"]
        self.cache_dir = Path(inference_config["cache_dir"])
        self.threshold = inference_config["threshold"]
        # The neutral slider position. Moving the request threshold away from it
        # shifts every per-label cut by the same delta.
        self.default_threshold = inference_config["threshold"]
        self.model_floor = float(inference_config.get("model_floor", 0.05))
        self.min_span_chars = int(inference_config.get("min_span_chars", 3))
        self.label_thresholds = dict(inference_config.get("label_thresholds", {}))
        self.device = inference_config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        # predict() reads self.LABELS; bind it here so the class is usable on its
        # own instead of depending on a caller to patch it in.
        self.LABELS = self.NER_LABELS

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        model_path = Path(self.model_name)

        if model_path.is_dir():
            local_model_path = str(model_path)
        else:
            local_model_path = snapshot_download(
                repo_id=self.model_name,
                cache_dir=str(self.cache_dir),
            )

        self.model = GLiNER2.from_pretrained(
            local_model_path,
            map_location=self.device,
        )
        self.model.eval()

    @torch.inference_mode()
    def predict(self, text: str):
        result = self.model.extract_entities(
            text,
            self.LABELS,
            threshold=min(self.model_floor, self.threshold),
            include_confidence=True,
            include_spans=True,
        )

        extracted_entities = []

        for label, entities in result.get("entities", {}).items():
            for entity in entities:
                extracted_entities.append(
                    {
                        "text": entity["text"],
                        "label": label,
                        "confidence": round(float(entity["confidence"]), 4),
                        "start": int(entity["start"]),
                        "end": int(entity["end"]),
                    }
                )

        extracted_entities = self.filter_entities(extracted_entities)
        extracted_entities.sort(key=lambda entity: entity["start"])

        return {
            "input_text": text,
            "model": self.model_name,
            "threshold": self.threshold,
            "device": self.device,
            "suspicious": len(extracted_entities) > 0,
            "risk_labels": sorted(
                {
                    entity["label"]
                    for entity in extracted_entities
                }
            ),
            "entities": extracted_entities,
        }

    def filter_entities(self, entities: list[dict]) -> list[dict]:
        """Apply the per-label cut, then drop filler and redundant spans."""
        shift = self.threshold - self.default_threshold
        kept = [
            entity
            for entity in entities
            if entity["confidence"] >= _shifted(
                self.label_thresholds.get(entity["label"], self.default_threshold), shift
            )
        ]
        kept = drop_filler_spans(kept, self.min_span_chars)
        return resolve_contained_spans(kept)


def _shifted(threshold: float, shift: float) -> float:
    return min(0.99, max(0.01, threshold + shift))


def drop_filler_spans(entities: list[dict], min_span_chars: int) -> list[dict]:
    """Remove spans that are too short or are a bare pronoun/article."""
    survivors = []
    for entity in entities:
        span = entity["text"].strip()
        if len(span) < min_span_chars or span.lower() in STOPWORD_SPANS:
            continue
        survivors.append(entity)
    return survivors


def resolve_contained_spans(entities: list[dict]) -> list[dict]:
    """Drop a span sitting inside a longer, at-least-as-confident one.

    The extractor routinely returns the same characters more than once -- the same
    name under two labels, or "remove" nested inside "remove my 1 star review".
    Every duplicate splits the highlighted review into another fragment, so the
    stronger span wins and the redundant one goes. A nested span that is *more*
    confident than its container survives, because it is usually the sharper piece
    of evidence.
    """
    ordered = sorted(
        entities,
        key=lambda entity: (
            -(entity["end"] - entity["start"]),
            -entity["confidence"],
            entity["start"],
        ),
    )
    kept: list[dict] = []
    for entity in ordered:
        covered = any(
            keeper["start"] <= entity["start"]
            and entity["end"] <= keeper["end"]
            and keeper["confidence"] >= entity["confidence"]
            for keeper in kept
        )
        if not covered:
            kept.append(entity)
    return kept
