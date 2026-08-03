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
# The reasoner runs on a hosted 131k-context model, so the review no longer has to
# be trimmed to fit. These bounds now exist to keep one request from sending an
# unbounded payload to a metered API: the budget matches data.max_chars, which the
# API layer already enforces, so an ordinary review passes through untouched.
REASONER_TEXT_BUDGET = 12_000
REASONER_TASK = (
    "Explain the calibrated classifier prediction using only evidence that literally "
    "appears in the original text. The calibrated classifier is the prediction source: "
    "do not replace its label. If selective_prediction.accepted is false, the displayed "
    "decision must be uncertain. Return only the Markdown format required by the system "
    "prompt. Do not reproduce raw classifier fields."
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

        # The lock only guards the two local GPU models. The reasoner is a hosted
        # HTTP call now, so holding it across that round trip would queue every
        # concurrent request behind one network wait for no benefit. Temperature is
        # passed per call rather than assigned to the shared reasoner, which would
        # be a race once two requests can be in here at once.
        reasoner_prompt = _build_reasoner_prompt(text, classifier_output)
        reasoner_started = time.perf_counter()
        reasoner_markdown = self._pipeline.reasoner.inference(
            reasoner_prompt,
            temperature=temperature,
        )
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


def _build_reasoner_prompt(text: str, classifier_output: dict[str, Any]) -> str:
    """Serialize the evidence the reasoner needs, and nothing more.

    The reasoner reads the review and the calibrated classifier verdict only. Entity
    extraction is a separate, parallel stage whose output drives the highlighting in
    the UI; feeding it here as well made the model anchor on whatever the extractor
    happened to find and quote its spans as though they were its own conclusions.

    The classifier output also echoes the full review back, so the untrimmed payload
    sent the same text twice and then indented the result. Sending each piece once
    keeps the request small, which is now billed per token, and keeps the model and
    calibration metadata the system prompt forbids quoting out of the payload.
    """
    return json.dumps(
        {
            "task": REASONER_TASK,
            "original_text": _truncate(text, REASONER_TEXT_BUDGET),
            "classifier_output": {
                "uncalibrated": classifier_output.get("uncalibrated", {}),
                "calibrated": classifier_output.get("calibrated", {}),
                "selective_prediction": classifier_output.get("selective_prediction", {}),
            },
        },
        ensure_ascii=False,
    )


def _truncate(text: str, budget: int) -> str:
    if len(text) <= budget:
        return text
    LOGGER.info("Trimming a %d character review to %d for the reasoner", len(text), budget)
    return f"{text[:budget].rstrip()} […truncated]"


def _elapsed_ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_") or "prediction"
