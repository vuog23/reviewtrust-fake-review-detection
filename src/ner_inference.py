from pathlib import Path

import torch
import yaml
from gliner2 import GLiNER2
from huggingface_hub import snapshot_download
from src.utils import NER_LABELS


class NERInference:
    NER_LABELS=NER_LABELS

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        inference_config = config["ner_inference"]

        self.model_name = inference_config["model_name"]
        self.cache_dir = Path(inference_config["cache_dir"])
        self.threshold = inference_config["threshold"]
        self.device = inference_config.get("device") or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

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
            threshold=self.threshold,
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