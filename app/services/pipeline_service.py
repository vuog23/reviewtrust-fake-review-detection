from __future__ import annotations

import json
import io
import logging
import re
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.services.entity_segments import segment_entities
from app.services.markdown_service import MarkdownService


LOGGER = logging.getLogger(__name__)
REASONER_TASK = (
    "Explain the calibrated classifier prediction using only evidence that literally "
    "appears in the original text and the supplied NER spans. The calibrated classifier "
    "is the prediction source: do not replace its label. If selective_prediction.accepted "
    "is false, the displayed decision must be uncertain. Return only the Markdown format "
    "required by the system prompt. Do not reproduce raw classifier or NER fields."
)


class PipelineService:
    """Web-safe adapter around the existing synchronous inference components."""

    def __init__(self, project_root: Path) -> None:
        from src.classifier_inference import Classifierinference
        from src.pipeline import InferencePipeline

        self.project_root = project_root.resolve()
        self.config_path = self.project_root / "config.yaml"
        self._classifier = Classifierinference
        startup_output = io.StringIO()
        with redirect_stdout(startup_output), redirect_stderr(startup_output):
            self._pipeline = InferencePipeline(config_path=str(self.config_path))
        captured_output = startup_output.getvalue().strip()
        if captured_output:
            LOGGER.debug("Model initialization output:\n%s", captured_output)
        self._markdown = MarkdownService()
        self._inference_lock = threading.Lock()

    def analyze(
        self,
        text: str,
        classifier_threshold: float,
        ner_threshold: float,
        temperature: float,
    ) -> dict[str, Any]:
        request_id = str(uuid4())
        timings: dict[str, float] = {}
        total_started = time.perf_counter()

        with self._inference_lock:
            classifier_started = time.perf_counter()
            classifier_output = self._classifier(
                text=text,
                model_name=self._pipeline.classifier_model_name,
                loss_key=self._pipeline.classifier_loss_key,
                config_path=str(self.config_path),
                confidence_threshold=classifier_threshold,
            )
            timings["classifier"] = _elapsed_ms(classifier_started)
            LOGGER.info("request_id=%s classifier_ms=%.2f", request_id, timings["classifier"])

            original_ner_threshold = self._pipeline.ner.threshold
            ner_started = time.perf_counter()
            try:
                self._pipeline.ner.threshold = ner_threshold
                ner_output = self._pipeline.ner.predict(text)
            finally:
                self._pipeline.ner.threshold = original_ner_threshold
            timings["ner"] = _elapsed_ms(ner_started)
            LOGGER.info("request_id=%s ner_ms=%.2f", request_id, timings["ner"])

            reasoner_prompt = json.dumps(
                {
                    "task": REASONER_TASK,
                    "original_text": text,
                    "classifier_output": classifier_output,
                    "ner_output": ner_output,
                },
                indent=2,
                ensure_ascii=False,
            )
            original_temperature = self._pipeline.reasoner.temperature
            reasoner_started = time.perf_counter()
            try:
                self._pipeline.reasoner.temperature = temperature
                reasoner_markdown = self._pipeline.reasoner.inference(reasoner_prompt)
            finally:
                self._pipeline.reasoner.temperature = original_temperature
            timings["reasoner"] = _elapsed_ms(reasoner_started)
            LOGGER.info("request_id=%s reasoner_ms=%.2f", request_id, timings["reasoner"])

        accepted = bool(classifier_output["selective_prediction"]["accepted"])
        label = str(classifier_output.get("calibrated", {}).get("label", "prediction"))
        status = f"accepted_{_slug(label)}" if accepted else "uncertain"
        response_ner = dict(ner_output)
        response_ner["segments"] = segment_entities(text, ner_output.get("entities", []))
        presentation_markdown = self._markdown.presentation_markdown(
            reasoner_markdown,
            original_text=text,
            classifier_output=classifier_output,
            ner_output=ner_output,
        )
        timings["total"] = _elapsed_ms(total_started)

        return {
            "request_id": request_id,
            "input_text": text,
            "settings": {
                "classifier_threshold": classifier_threshold,
                "ner_threshold": ner_threshold,
                "temperature": temperature,
            },
            "status": status,
            "classifier": classifier_output,
            "ner": response_ner,
            "reasoner": {
                "markdown": reasoner_markdown,
                "html": self._markdown.render(presentation_markdown),
            },
            "timings_ms": {key: round(value, 2) for key, value in timings.items()},
        }


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "prediction"
