from io import BytesIO

from PIL import Image


def test_home_returns_200(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "ReviewTrust" in response.text
    assert 'id="ocr-image"' in response.text
    assert 'id="ocr-analyze"' in response.text


def test_health_reports_ready(client):
    assert client.get("/api/health").json() == {
        "status": "ok", "pipeline_ready": True, "hosted_models_ready": True,
    }


def test_health_is_degraded_without_a_groq_key(client):
    """Local models load fine with no key, but every analysis would 503."""
    client.app.state.ocr_service.is_configured = False
    try:
        payload = client.get("/api/health").json()
    finally:
        client.app.state.ocr_service.is_configured = True

    assert payload == {"status": "degraded", "pipeline_ready": True, "hosted_models_ready": False}


def test_defaults_come_from_config(client):
    defaults = client.get("/api/defaults").json()
    assert defaults["classifier_threshold"] == 0.90
    assert defaults["ner_threshold"] == 0.50
    assert defaults["temperature"] == 0.70
    assert defaults["max_chars"] == 12_000


def test_blank_review_is_rejected(client):
    response = client.post("/api/analyze", json={"text": "   "})
    assert response.status_code == 422


def test_accepted_prediction(client):
    response = client.post("/api/analyze", json={
        "text": "I received a gift card.", "classifier_threshold": 0.80,
        "ner_threshold": 0.55, "temperature": 0.40,
    })
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "accepted_fake"
    assert payload["classifier"]["selective_prediction"]["accepted"] is True


def test_uncertain_prediction_still_has_ner_and_reasoner(client):
    response = client.post("/api/analyze", json={
        "text": "I received a gift card.", "classifier_threshold": 0.90,
        "ner_threshold": 0.55, "temperature": 0.45,
    })
    payload = response.json()
    assert response.status_code == 200
    assert payload["status"] == "uncertain"
    assert payload["classifier"]["selective_prediction"]["accepted"] is False
    assert payload["ner"]["entities"][0]["text"] == "gift card"
    assert "Evidence summary" in payload["reasoner"]["markdown"]


def test_image_text_runs_through_complete_pipeline(ocr_client):
    client, fake_ocr = ocr_client
    buffer = BytesIO()
    Image.new("RGB", (120, 60), "white").save(buffer, format="PNG")
    response = client.post(
        "/api/analyze-image",
        files={"image": ("review.png", buffer.getvalue(), "image/png")},
        data={
            "classifier_threshold": "0.90",
            "ner_threshold": "0.50",
            "temperature": "0.70",
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert fake_ocr.calls == 1
    # The image is handed over as bytes plus its declared type; no temp file involved.
    assert fake_ocr.received == [(len(buffer.getvalue()), "image/png")]
    assert payload["input_text"].startswith("The seller offered")
    assert payload["ocr"]["model"] == "qwen/qwen3.6-27b"
    assert payload["ocr"]["filename"] == "review.png"
    assert payload["classifier"]
    assert payload["ner"]
    assert payload["reasoner"]


def test_image_analysis_rejects_unsupported_file_type(ocr_client):
    client, fake_ocr = ocr_client
    response = client.post(
        "/api/analyze-image",
        files={"image": ("review.txt", b"not an image", "text/plain")},
    )
    assert response.status_code == 422
    assert fake_ocr.calls == 0


def test_decompression_bomb_is_rejected_as_a_bad_image(ocr_client):
    """A tiny file declaring a huge bitmap must not escape as a 500."""
    client, fake_ocr = ocr_client
    buffer = BytesIO()
    Image.new("L", (25_000, 25_000), 255).save(buffer, format="PNG")
    assert len(buffer.getvalue()) < 10 * 1024 * 1024

    response = client.post(
        "/api/analyze-image",
        files={"image": ("bomb.png", buffer.getvalue(), "image/png")},
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "The uploaded file is not a valid image."
    assert fake_ocr.calls == 0
