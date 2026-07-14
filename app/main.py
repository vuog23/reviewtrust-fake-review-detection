from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

from app.runtime_environment import configure_windows_cuda_dlls, isolate_web_runtime


PROJECT_DIRECTORY = Path(__file__).resolve().parent.parent
REMOVED_OCR_PATHS = isolate_web_runtime(PROJECT_DIRECTORY)
REGISTERED_CUDA_PATHS = configure_windows_cuda_dlls()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import transformers

from app.api.routes import router
from app.dependencies import PROJECT_ROOT, load_settings
from app.services.pipeline_service import PipelineService
from app.services.ocr_service import DeepSeekOCRService


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if REMOVED_OCR_PATHS:
    LOGGER.warning("Removed OCR-only paths from web runtime: %s", REMOVED_OCR_PATHS)
if REGISTERED_CUDA_PATHS:
    LOGGER.info("Registered Windows CUDA DLL paths: %s", REGISTERED_CUDA_PATHS)
LOGGER.info(
    "Text Transformers runtime %s from %s",
    transformers.__version__,
    transformers.__file__,
)


def create_app(
    service_factory: Callable[[Path], object] | None = None,
    ocr_service_factory: Callable[[Path], object] | None = None,
    project_root: Path | None = None,
) -> FastAPI:
    root = (project_root or PROJECT_ROOT).resolve()
    factory = service_factory or PipelineService
    ocr_factory = ocr_service_factory or DeepSeekOCRService

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.project_root = root
        application.state.settings = load_settings(root)
        application.state.pipeline_service = None
        application.state.ocr_service = ocr_factory(root)
        ensure_ocr_worker = getattr(
            application.state.ocr_service,
            "ensure_worker_environment",
            None,
        )
        if callable(ensure_ocr_worker):
            try:
                ensure_ocr_worker()
            except Exception as exc:
                LOGGER.warning(
                    "OCR worker setup could not be completed; text analysis remains available: %s",
                    exc,
                )
        application.state.pipeline_factory = factory
        application.state.model_lifecycle_lock = threading.Lock()
        LOGGER.info("Starting ReviewTrust inference pipeline")
        try:
            application.state.pipeline_service = factory(root)
            LOGGER.info("Inference pipeline ready")
        except Exception as exc:
            LOGGER.exception("Inference pipeline initialization failed: %s", exc)
        yield
        LOGGER.info("ReviewTrust shutdown complete")

    application = FastAPI(
        title="ReviewTrust",
        description="Confidence-calibrated fake product review analysis",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.mount(
        "/static",
        StaticFiles(directory=Path(__file__).resolve().parent / "static"),
        name="static",
    )
    application.include_router(router)
    return application


app = create_app()
