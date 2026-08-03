from __future__ import annotations

import base64
import io
import logging
import re
from pathlib import Path
from typing import Any

import yaml
from PIL import Image

from src.groq_client import GroqClient, GroqUnavailableError


LOGGER = logging.getLogger(__name__)

# Groq documents PNG and JPEG for image input, so a WebP upload is transcoded to
# PNG on the way out rather than forwarded in a format the API may refuse.
_FORMAT_BY_CONTENT_TYPE = {
    "image/png": ("PNG", "image/png"),
    "image/jpeg": ("JPEG", "image/jpeg"),
    "image/webp": ("PNG", "image/png"),
}


class OCRUnavailableError(RuntimeError):
    """Raised when the transcription model cannot be reached or configured."""


class OCRExtractionError(RuntimeError):
    """Raised when the image contains no usable review text."""


class GroqOCRService:
    """Read review text out of an uploaded image with a hosted multimodal model.

    This replaces the previous local DeepSeek-OCR worker. That design needed a
    second virtualenv pinned to an older Transformers, a subprocess to isolate it,
    and a full teardown of the text pipeline to free VRAM for every image. Sending
    the image to Groq removes all of it.
    """

    def __init__(self, project_root: Path, client: GroqClient | None = None) -> None:
        self.project_root = project_root.resolve()
        config_path = self.project_root / "config.yaml"
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        settings = config.get("ocr_inference", {})

        self.model_name = str(settings.get("model_name", "qwen/qwen3.6-27b"))
        self.prompt = str(settings.get("prompt", _DEFAULT_PROMPT))
        self.temperature = float(settings.get("temperature", 0.0))
        self.max_completion_tokens = int(settings.get("max_completion_tokens", 4096))
        self.max_upload_bytes = int(settings.get("max_upload_mb", 10)) * 1024 * 1024
        self.max_image_edge = int(settings.get("max_image_edge", 2000))
        self.max_encoded_bytes = int(settings.get("max_encoded_mb", 3)) * 1024 * 1024
        # A small file can still decode to an enormous bitmap: a 25000x25000 PNG of one
        # flat colour compresses to a few hundred KB but needs ~1.9 GB decoded.
        self.max_image_pixels = int(settings.get("max_image_pixels", 40_000_000))

        self._client = client or GroqClient(
            model=self.model_name,
            api_key_env=settings.get("api_key_env", "GROQ_API_KEY"),
            base_url=settings.get("base_url"),
            timeout=float(settings.get("timeout_seconds", 120)),
            max_retries=int(settings.get("max_retries", 3)),
            reasoning_effort=settings.get("reasoning_effort", "none"),
            reasoning_format=settings.get("reasoning_format", "hidden"),
        )

    @property
    def is_configured(self) -> bool:
        return self._client.is_configured

    def extract_text(self, image_bytes: bytes, content_type: str = "image/png") -> str:
        data_url, mime = self._encode(image_bytes, content_type)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": self.prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ]
        LOGGER.info("Transcribing a %s image with %s", mime, self.model_name)
        try:
            reply = self._client.complete(
                messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )
        except GroqUnavailableError as exc:
            raise OCRUnavailableError(str(exc)) from exc

        text = _clean_ocr_text(reply)
        if not text or text.upper().startswith("NO_TEXT_FOUND"):
            raise OCRExtractionError("No readable review text was found in this image.")
        return text

    def _encode(self, image_bytes: bytes, content_type: str) -> tuple[str, str]:
        """Shrink and re-encode the upload so the request stays comfortably small.

        A 10 MB upload becomes roughly 13 MB once base64-encoded, which is large
        enough to be refused and slow enough to be worth avoiding. Review
        screenshots stay legible far below that.
        """
        try:
            with Image.open(io.BytesIO(image_bytes)) as image:
                pixels = image.width * image.height
                if pixels > self.max_image_pixels:
                    raise OCRExtractionError(
                        f"That image decodes to {pixels:,} pixels, which is too large to process."
                    )
                image.load()
                image = _flatten_transparency(image)
                longest = max(image.size)
                if longest > self.max_image_edge:
                    scale = self.max_image_edge / longest
                    resized = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
                    LOGGER.info("Downscaling image from %s to %s for transcription", image.size, resized)
                    image = image.resize(resized, Image.LANCZOS)

                encoded, mime = _encode_lossless_or_jpeg(image, content_type, self.max_encoded_bytes)
        except OCRExtractionError:
            raise
        except Exception as exc:
            raise OCRExtractionError("The uploaded image could not be read.") from exc

        return f"data:{mime};base64,{base64.b64encode(encoded).decode('ascii')}", mime


def _flatten_transparency(image: Any) -> Any:
    """Composite onto white before dropping the alpha channel.

    A bare ``convert("RGB")`` keeps whatever sits *under* transparent pixels, which for
    a screenshot exported with a transparent background is black -- turning dark text
    on nothing into dark text on black, which no model can read.
    """
    has_alpha = image.mode in {"RGBA", "LA"} or (
        image.mode == "P" and "transparency" in image.info
    )
    if not has_alpha:
        return image.convert("RGB")

    LOGGER.info("Flattening a transparent %s image onto white", image.mode)
    rgba = image.convert("RGBA")
    backdrop = Image.new("RGB", rgba.size, (255, 255, 255))
    backdrop.paste(rgba, mask=rgba.split()[-1])
    return backdrop


def _encode_lossless_or_jpeg(image: Any, content_type: str, budget: int) -> tuple[bytes, str]:
    """Prefer lossless so text edges stay crisp, but fall back to JPEG when it is bulky."""
    image_format, mime = _FORMAT_BY_CONTENT_TYPE.get(content_type, ("PNG", "image/png"))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    encoded = buffer.getvalue()
    if len(encoded) <= budget:
        return encoded, mime

    fallback = io.BytesIO()
    image.save(fallback, format="JPEG", quality=85, optimize=True)
    recoded = fallback.getvalue()
    # Re-encoding a JPEG that was already saved below quality 85 makes it bigger,
    # so only take the fallback when it actually helps.
    if len(recoded) >= len(encoded):
        LOGGER.info("Keeping the original %d byte image; JPEG fallback was larger", len(encoded))
        return encoded, mime
    LOGGER.info("Re-encoded a %d byte image to %d bytes of JPEG", len(encoded), len(recoded))
    return recoded, "image/jpeg"


_DEFAULT_PROMPT = (
    "Transcribe every piece of text visible in this image, exactly as written. "
    "This is a screenshot of a product review or a message about one. "
    "Return only the transcribed text, with no commentary, no markdown fences, and no description "
    "of the image. Preserve the original line breaks. "
    "If the image contains no readable text at all, reply with exactly: NO_TEXT_FOUND"
)


def _clean_ocr_text(value: str) -> str:
    value = re.sub(r"<\|[^>]+\|>", "", value)
    value = re.sub(r"!\[[^]]*]\([^)]*\)", "", value)
    value = value.replace("```markdown", "").replace("```", "")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
