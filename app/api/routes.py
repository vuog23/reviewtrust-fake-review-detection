from __future__ import annotations

import gc
import logging
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.templating import Jinja2Templates

from app.schemas import AnalyzeRequest
from app.services.ocr_service import OCRExtractionError, OCRUnavailableError
from src.groq_client import GroqUnavailableError


LOGGER = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent.parent / "templates"))


@router.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"max_chars": request.app.state.settings.max_chars},
    )


@router.get("/api/health")
async def health(request: Request) -> dict:
    ready = request.app.state.pipeline_service is not None
    # Without a Groq key the local models load fine but every analysis 503s, so
    # reporting plain "ok" would hide a fully non-functional deployment.
    hosted_ready = bool(getattr(request.app.state.ocr_service, "is_configured", True))
    return {
        "status": "ok" if ready and hosted_ready else "degraded",
        "pipeline_ready": ready,
        "hosted_models_ready": hosted_ready,
    }


@router.get("/api/defaults")
async def defaults(request: Request) -> dict:
    return request.app.state.settings.as_dict()


@router.post("/api/analyze")
async def analyze(payload: AnalyzeRequest, request: Request) -> dict:
    settings = request.app.state.settings
    if len(payload.text) > settings.max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"Review text must be {settings.max_chars:,} characters or fewer.",
        )

    if request.app.state.pipeline_service is None:
        raise HTTPException(
            status_code=503,
            detail="The inference pipeline is unavailable. Check the configured model files and restart the app.",
        )

    try:
        return await run_in_threadpool(
            _run_text_analysis,
            request.app,
            payload.text,
            payload.classifier_threshold,
            payload.ner_threshold,
            payload.temperature,
        )
    except GroqUnavailableError as exc:
        LOGGER.warning("Explanation model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except _out_of_memory_errors() as exc:
        LOGGER.warning("Inference ran out of GPU memory: %s", exc)
        _release_gpu_memory()
        raise HTTPException(
            status_code=503,
            detail="The GPU ran out of memory for this review. Try a shorter review or retry in a moment.",
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        LOGGER.warning("Inference resource unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="A required model checkpoint or calibration resource is unavailable.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected inference failure")
        raise HTTPException(
            status_code=500,
            detail="Analysis could not be completed. Review the server log for details.",
        ) from exc


@router.post("/api/analyze-image")
async def analyze_image(
    request: Request,
    image: UploadFile = File(...),
    classifier_threshold: float = Form(0.90, ge=0.50, le=0.99),
    ner_threshold: float = Form(0.50, ge=0.05, le=0.95),
    temperature: float = Form(0.70, ge=0.10, le=1.50),
) -> dict:
    ocr_service = request.app.state.ocr_service
    if request.app.state.pipeline_service is None:
        raise HTTPException(status_code=503, detail="The inference pipeline is unavailable.")

    allowed_types = {"image/jpeg", "image/png", "image/webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=422, detail="Upload a PNG, JPEG, or WebP image.")

    contents = await image.read(ocr_service.max_upload_bytes + 1)
    if len(contents) > ocr_service.max_upload_bytes:
        limit_mb = ocr_service.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=422, detail=f"Image must be {limit_mb} MB or smaller.")
    try:
        with Image.open(BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        # DecompressionBombError is not an OSError, so it needs naming explicitly:
        # a small file declaring enormous dimensions would otherwise escape as a 500.
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from exc

    try:
        result = await run_in_threadpool(
            _run_image_analysis,
            request.app,
            contents,
            image.content_type,
            classifier_threshold,
            ner_threshold,
            temperature,
        )
        result["ocr"] = {
            "model": ocr_service.model_name,
            "filename": Path(image.filename or "image").name,
            "text": result["input_text"],
        }
        return result
    except (OCRUnavailableError, GroqUnavailableError) as exc:
        LOGGER.warning("Transcription model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRExtractionError as exc:
        LOGGER.warning("Transcription failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except _out_of_memory_errors() as exc:
        LOGGER.warning("Image analysis ran out of GPU memory: %s", exc)
        _release_gpu_memory()
        raise HTTPException(
            status_code=503,
            detail="The GPU ran out of memory for this review. Try a shorter review or retry in a moment.",
        ) from exc
    except (FileNotFoundError, OSError) as exc:
        # Same failure as the text route: report it the same way rather than as a 500.
        LOGGER.warning("Inference resource unavailable: %s", exc)
        raise HTTPException(
            status_code=503,
            detail="A required model checkpoint or calibration resource is unavailable.",
        ) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected image-analysis failure")
        raise HTTPException(status_code=500, detail="Image analysis could not be completed.") from exc


def _run_text_analysis(application, text, classifier_threshold, ner_threshold, temperature):
    service = application.state.pipeline_service
    if service is None:
        raise RuntimeError("The inference pipeline is temporarily unavailable.")
    return service.analyze(text, classifier_threshold, ner_threshold, temperature)


def _run_image_analysis(
    application,
    image_bytes,
    content_type,
    classifier_threshold,
    ner_threshold,
    temperature,
):
    """Transcribe the upload, then run the ordinary text pipeline over the result.

    Transcription is a network call to Groq and holds no GPU memory, so unlike the
    previous local-OCR design it needs no teardown of the text pipeline.
    """
    extracted_text = application.state.ocr_service.extract_text(image_bytes, content_type)
    extracted_text = extracted_text[: application.state.settings.max_chars]
    return _run_text_analysis(
        application,
        extracted_text,
        classifier_threshold,
        ner_threshold,
        temperature,
    )


def _release_gpu_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _out_of_memory_errors() -> tuple[type[BaseException], ...]:
    """Torch may be absent in test runs, so resolve the exception type lazily."""
    try:
        import torch

        return (getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError),)
    except Exception:
        return ()
