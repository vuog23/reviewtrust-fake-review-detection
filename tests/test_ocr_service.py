import base64
import io

import pytest
from PIL import Image, ImageDraw

from app.services.ocr_service import (
    GroqOCRService,
    OCRExtractionError,
    OCRUnavailableError,
    _encode_lossless_or_jpeg,
)
from src.groq_client import GroqClient
from tests.conftest import FakeGroqSDK


CONFIG = """
ocr_inference:
  model_name: qwen/qwen3.6-27b
  temperature: 0.0
  max_completion_tokens: 4096
  max_upload_mb: 10
  max_image_edge: 200
  max_encoded_mb: 3
  max_image_pixels: 40000000
  prompt: Transcribe the text.
"""


def _service(tmp_path, replies=None):
    (tmp_path / "config.yaml").write_text(CONFIG, encoding="utf-8")
    sdk = FakeGroqSDK(replies or ["The seller offered a gift card."])
    client = GroqClient(model="qwen/qwen3.6-27b", client=sdk)
    return GroqOCRService(tmp_path, client=client), sdk


def _png(size=(120, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _sent_image(sdk):
    """Pull the decoded image back out of the recorded request."""
    parts = sdk.requests[0]["messages"][0]["content"]
    url = next(part["image_url"]["url"] for part in parts if part["type"] == "image_url")
    header, encoded = url.split(",", 1)
    return header, Image.open(io.BytesIO(base64.b64decode(encoded)))


def test_transcription_is_returned(tmp_path):
    service, _ = _service(tmp_path)
    assert service.extract_text(_png(), "image/png") == "The seller offered a gift card."


def test_prompt_and_image_are_sent_in_one_user_message(tmp_path):
    service, sdk = _service(tmp_path)

    service.extract_text(_png(), "image/png")

    parts = sdk.requests[0]["messages"][0]["content"]
    assert sdk.requests[0]["messages"][0]["role"] == "user"
    assert {part["type"] for part in parts} == {"text", "image_url"}
    assert parts[0]["text"] == "Transcribe the text."
    header, _ = _sent_image(sdk)
    assert header == "data:image/png;base64"


def test_large_image_is_downscaled_before_upload(tmp_path):
    service, sdk = _service(tmp_path)  # max_image_edge is 200 in the test config

    service.extract_text(_png((900, 300)), "image/png")

    _, sent = _sent_image(sdk)
    assert max(sent.size) == 200
    assert sent.size == (200, 67)


def test_small_image_is_left_at_its_original_size(tmp_path):
    service, sdk = _service(tmp_path)

    service.extract_text(_png((120, 60)), "image/png")

    _, sent = _sent_image(sdk)
    assert sent.size == (120, 60)


def test_webp_upload_is_transcoded_to_png(tmp_path):
    """Groq documents PNG and JPEG only, so a WebP must not go out as WebP."""
    service, sdk = _service(tmp_path)
    buffer = io.BytesIO()
    Image.new("RGB", (120, 60), "white").save(buffer, format="WEBP")

    service.extract_text(buffer.getvalue(), "image/webp")

    header, sent = _sent_image(sdk)
    assert header == "data:image/png;base64"
    assert sent.format == "PNG"


def test_declared_mime_always_matches_the_bytes_sent(tmp_path):
    for content_type, pil_format in (("image/png", "PNG"), ("image/jpeg", "JPEG")):
        service, sdk = _service(tmp_path)
        buffer = io.BytesIO()
        Image.new("RGB", (120, 60), "white").save(buffer, format=pil_format)

        service.extract_text(buffer.getvalue(), content_type)

        header, sent = _sent_image(sdk)
        assert header == f"data:{content_type};base64"
        assert sent.format == pil_format


def test_transparent_png_is_flattened_onto_white_not_black(tmp_path):
    """A bare convert("RGB") leaves dark text on black, which no model can read."""
    service, sdk = _service(tmp_path)
    source = Image.new("RGBA", (300, 80), (0, 0, 0, 0))
    ImageDraw.Draw(source).text((10, 30), "gift card for a review", fill=(20, 20, 20, 255))
    buffer = io.BytesIO()
    source.save(buffer, format="PNG")

    service.extract_text(buffer.getvalue(), "image/png")

    _, sent = _sent_image(sdk)
    darkest, brightest = sent.convert("L").getextrema()
    # Before flattening, the transparent field kept the black underneath it and the
    # whole frame came out dark: brightest was the text itself, around 20.
    assert brightest == 255, "background must be white"
    assert darkest < 128, "text must still be dark against it"


def test_palette_png_with_transparency_is_also_flattened(tmp_path):
    service, sdk = _service(tmp_path)
    source = Image.new("RGBA", (60, 60), (0, 0, 0, 0)).convert("P", palette=Image.ADAPTIVE)
    buffer = io.BytesIO()
    source.save(buffer, format="PNG", transparency=0)

    service.extract_text(buffer.getvalue(), "image/png")

    _, sent = _sent_image(sdk)
    assert sent.getpixel((0, 0)) == (255, 255, 255)


def test_pixel_bomb_is_rejected_before_the_bitmap_is_decoded(tmp_path):
    """A small file can declare an enormous bitmap; reject on dimensions, not bytes."""
    service, sdk = _service(tmp_path)
    buffer = io.BytesIO()
    Image.new("L", (9000, 9000), 255).save(buffer, format="PNG")
    assert len(buffer.getvalue()) < 200_000

    with pytest.raises(OCRExtractionError, match="too large to process"):
        service.extract_text(buffer.getvalue(), "image/png")
    assert sdk.requests == []


@pytest.mark.parametrize("kind", ["smooth", "noisy"])
def test_over_budget_fallback_never_enlarges_the_payload(kind):
    """Quality 85 can be bigger than the first attempt; when it is, keep the first."""
    image = Image.new("RGB", (400, 400))
    for x in range(400):
        for y in range(400):
            if kind == "smooth":
                image.putpixel((x, y), (x // 2, y // 2, 128))
            else:
                image.putpixel((x, y), ((x * 7 + y * 13) % 256, (x * 3) % 256, (y * 5) % 256))
    first_attempt = io.BytesIO()
    image.save(first_attempt, format="JPEG")

    # budget=0 forces the over-budget branch every time.
    sent, mime = _encode_lossless_or_jpeg(image, "image/jpeg", 0)

    assert len(sent) <= len(first_attempt.getvalue())
    assert mime == "image/jpeg"
    assert Image.open(io.BytesIO(sent)).size == (400, 400)


def test_no_text_sentinel_becomes_a_user_facing_error(tmp_path):
    service, _ = _service(tmp_path, ["NO_TEXT_FOUND"])

    with pytest.raises(OCRExtractionError, match="No readable review text"):
        service.extract_text(_png(), "image/png")


def test_empty_reply_becomes_a_user_facing_error(tmp_path):
    service, _ = _service(tmp_path, ["   "])

    with pytest.raises(OCRExtractionError):
        service.extract_text(_png(), "image/png")


def test_code_fences_are_stripped_from_the_transcription(tmp_path):
    service, _ = _service(tmp_path, ["```markdown\nGreat product!\n```"])

    assert service.extract_text(_png(), "image/png") == "Great product!"


def test_unreadable_bytes_are_rejected_before_any_api_call(tmp_path):
    service, sdk = _service(tmp_path)

    with pytest.raises(OCRExtractionError, match="could not be read"):
        service.extract_text(b"this is not an image", "image/png")
    assert sdk.requests == []


def test_provider_failure_surfaces_as_unavailable(tmp_path):
    service, _ = _service(tmp_path, [type("AuthenticationError", (Exception,), {})("nope")])

    with pytest.raises(OCRUnavailableError, match="rejected the API key"):
        service.extract_text(_png(), "image/png")


def test_upload_limit_comes_from_config(tmp_path):
    service, _ = _service(tmp_path)
    assert service.max_upload_bytes == 10 * 1024 * 1024
