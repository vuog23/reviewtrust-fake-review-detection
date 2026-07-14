import json
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def Classifierinference(
    text: str | list[str],
    model_name: str = None,
    loss_key: str = None,
    config_path: str = "config.yaml",
    confidence_threshold: float = None,
    device: str = None,
):
    resolved_config_path = Path(config_path).resolve()
    project_root = resolved_config_path.parent
    with resolved_config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    inference_config = config.get("classifier_inference", {})

    model_name = model_name or inference_config.get("model_name", "distilbert")
    loss_key = loss_key or inference_config.get("loss_key", "label_smoothing")
    device = device or inference_config.get("device")

    model_name = model_name.lower().strip()
    loss_key = loss_key.lower().strip()

    output_dir = _project_path(project_root, config.get("output_dir", "models"))
    cache_dir = _project_path(project_root, config.get("cache_dir", "cache"))
    model_config = config["models"][model_name]
    max_length = model_config.get("max_length", 512)
    trust_remote_code = model_config.get("trust_remote_code", False)
    use_safetensors = inference_config.get("use_safetensors", True)

    if confidence_threshold is None:
        confidence_threshold = inference_config.get(
            "confidence_threshold",
            config.get("evaluation", {}).get("high_confidence_threshold", 0.90),
        )

    configured_checkpoint = inference_config.get("checkpoint_path")
    checkpoint_path = (
        _project_path(project_root, configured_checkpoint)
        if configured_checkpoint
        else output_dir / "checkpoints" / model_name / loss_key / "best_model"
    )
    configured_temperature = inference_config.get("temperature_path")
    temperature_path = (
        _project_path(project_root, configured_temperature)
        if configured_temperature
        else output_dir / "calibration" / model_name / loss_key / "temperature.json"
    )

    with temperature_path.open("r", encoding="utf-8") as file:
        temperature_data = json.load(file)

    temperature = float(temperature_data["temperature"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_fast=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        cache_dir=cache_dir,
        trust_remote_code=trust_remote_code,
        use_safetensors=use_safetensors,
    ).to(device)
    model.eval()

    single_text = isinstance(text, str)
    texts = [text] if single_text else text

    encoded = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
    )
    encoded = {
        key: value.to(device)
        for key, value in encoded.items()
    }

    with torch.inference_mode():
        logits = model(**encoded).logits
        uncalibrated_probabilities = torch.softmax(logits, dim=1)
        calibrated_probabilities = torch.softmax(
            logits / temperature,
            dim=1,
        )

    id2label = {
        int(label_id): label_name
        for label_id, label_name in model.config.id2label.items()
    }

    outputs = []

    for index, current_text in enumerate(texts):
        uncalibrated_confidence, uncalibrated_id = (
            uncalibrated_probabilities[index].max(dim=0)
        )
        calibrated_confidence, calibrated_id = (
            calibrated_probabilities[index].max(dim=0)
        )

        uncalibrated_id = int(uncalibrated_id.item())
        calibrated_id = int(calibrated_id.item())
        calibrated_confidence = float(calibrated_confidence.item())

        outputs.append(
            {
                "text": current_text,
                "model": model_name,
                "loss": loss_key,
                "temperature": temperature,
                "uncalibrated": {
                    "label_id": uncalibrated_id,
                    "label": id2label[uncalibrated_id],
                    "confidence": float(uncalibrated_confidence.item()),
                    "probabilities": {
                        id2label[class_id]: float(probability)
                        for class_id, probability in enumerate(
                            uncalibrated_probabilities[index].cpu().tolist()
                        )
                    },
                },
                "calibrated": {
                    "label_id": calibrated_id,
                    "label": id2label[calibrated_id],
                    "confidence": calibrated_confidence,
                    "probabilities": {
                        id2label[class_id]: float(probability)
                        for class_id, probability in enumerate(
                            calibrated_probabilities[index].cpu().tolist()
                        )
                    },
                },
                "selective_prediction": {
                    "confidence_threshold": confidence_threshold,
                    "accepted": calibrated_confidence >= confidence_threshold,
                    "decision": (
                        "accepted"
                        if calibrated_confidence >= confidence_threshold
                        else "rejected"
                    ),
                },
            }
        )

    return outputs[0] if single_text else outputs


def _project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path
