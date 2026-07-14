import json
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.markdown_service import MarkdownService
from app.services.pipeline_service import PipelineService


class FakeNER:
    threshold = 0.50

    def predict(self, text):
        needle = "gift card"
        start = text.lower().find(needle)
        entities = [] if start < 0 else [{
            "text": text[start:start + len(needle)], "label": "payment_or_reward",
            "confidence": 0.94, "start": start, "end": start + len(needle),
        }]
        return {
            "input_text": text, "model": "fake-ner", "threshold": self.threshold,
            "device": "cpu", "suspicious": bool(entities),
            "risk_labels": ["payment_or_reward"] if entities else [], "entities": entities,
        }


class FakeReasoner:
    temperature = 0.70

    def __init__(self):
        self.last_prompt = None
        self.used_temperature = None

    def inference(self, prompt):
        self.last_prompt = json.loads(prompt)
        self.used_temperature = self.temperature
        return "## Evidence summary\n\nA **gift card** was detected. <script>alert(1)</script>"


class FakeOCR:
    model_name = "deepseek-ai/DeepSeek-OCR"
    max_upload_bytes = 10 * 1024 * 1024

    def __init__(self):
        self.calls = 0
        self.unload_calls = 0
        self.setup_calls = 0

    def ensure_worker_environment(self):
        self.setup_calls += 1

    def extract_text(self, _):
        self.calls += 1
        return "The seller offered a gift card for a five star review."

    def unload(self):
        self.unload_calls += 1


def fake_classifier(*, text, confidence_threshold, **_):
    confidence = 0.82
    accepted = confidence >= confidence_threshold
    return {
        "text": text, "model": "fake-model", "loss": "fake-loss", "temperature": 1.25,
        "uncalibrated": {"label_id": 1, "label": "FAKE", "confidence": 0.91, "probabilities": {"REAL": 0.09, "FAKE": 0.91}},
        "calibrated": {"label_id": 1, "label": "FAKE", "confidence": confidence, "probabilities": {"REAL": 0.18, "FAKE": 0.82}},
        "selective_prediction": {"confidence_threshold": confidence_threshold, "accepted": accepted, "decision": "accepted" if accepted else "rejected"},
    }


def build_fake_service():
    service = PipelineService.__new__(PipelineService)
    service.project_root = Path.cwd()
    service.config_path = Path.cwd() / "config.yaml"
    service._classifier = fake_classifier
    service._pipeline = SimpleNamespace(
        classifier_model_name=None, classifier_loss_key=None, ner=FakeNER(), reasoner=FakeReasoner()
    )
    service._markdown = MarkdownService()
    service._inference_lock = threading.Lock()
    return service


@pytest.fixture
def fake_service():
    return build_fake_service()


@pytest.fixture
def client(fake_service):
    app = create_app(
        service_factory=lambda _: fake_service,
        ocr_service_factory=lambda _: FakeOCR(),
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def ocr_client(fake_service):
    fake_ocr = FakeOCR()
    app = create_app(
        service_factory=lambda _: fake_service,
        ocr_service_factory=lambda _: fake_ocr,
    )
    with TestClient(app) as test_client:
        yield test_client, fake_ocr
