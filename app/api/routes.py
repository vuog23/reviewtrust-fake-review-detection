from __future__ import annotations

import gc
import logging
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.templating import Jinja2Templates

from app.schemas import AnalyzeRequest
from app.services.ocr_service import OCRExtractionError, OCRUnavailableError


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
    return {"status": "ok" if ready else "degraded", "pipeline_ready": ready}


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

    allowed_types = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    if image.content_type not in allowed_types:
        raise HTTPException(status_code=422, detail="Upload a PNG, JPEG, or WebP image.")

    contents = await image.read(ocr_service.max_upload_bytes + 1)
    if len(contents) > ocr_service.max_upload_bytes:
        limit_mb = ocr_service.max_upload_bytes // (1024 * 1024)
        raise HTTPException(status_code=422, detail=f"Image must be {limit_mb} MB or smaller.")
    try:
        with Image.open(BytesIO(contents)) as uploaded_image:
            uploaded_image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="The uploaded file is not a valid image.") from exc

    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(delete=False, suffix=allowed_types[image.content_type]) as temp_file:
            temp_file.write(contents)
            temporary_path = Path(temp_file.name)
        result = await run_in_threadpool(
            _run_image_analysis,
            request.app,
            temporary_path,
            classifier_threshold,
            ner_threshold,
            temperature,
        )
        extracted_text = result["input_text"]
        result["ocr"] = {
            "model": ocr_service.model_name,
            "filename": Path(image.filename or "image").name,
            "text": extracted_text,
        }
        return result
    except OCRUnavailableError as exc:
        LOGGER.warning("OCR model unavailable: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except OCRExtractionError as exc:
        LOGGER.warning("OCR extraction failed: %s", exc)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected image-analysis failure")
        raise HTTPException(status_code=500, detail="Image analysis could not be completed.") from exc
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _run_text_analysis(application, text, classifier_threshold, ner_threshold, temperature):
    with application.state.model_lifecycle_lock:
        service = application.state.pipeline_service
        if service is None:
            raise RuntimeError("The inference pipeline is temporarily unavailable.")
        return service.analyze(text, classifier_threshold, ner_threshold, temperature)


def _run_image_analysis(application, image_path, classifier_threshold, ner_threshold, temperature):
    """Swap large GPU models so OCR and text inference can run on low-VRAM systems."""
    with application.state.model_lifecycle_lock:
        old_pipeline = application.state.pipeline_service
        application.state.pipeline_service = None
        del old_pipeline
        _release_gpu_memory()

        extracted_text = ""
        try:
            extracted_text = application.state.ocr_service.extract_text(image_path)
            extracted_text = extracted_text[: application.state.settings.max_chars]
        finally:
            application.state.ocr_service.unload()
            _release_gpu_memory()
            LOGGER.info("Restoring text inference pipeline after OCR")
            application.state.pipeline_service = application.state.pipeline_factory(
                application.state.project_root
            )

        return application.state.pipeline_service.analyze(
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
