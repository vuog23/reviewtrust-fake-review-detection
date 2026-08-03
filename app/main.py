from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Callable

# These two run before torch or transformers is imported: .env supplies the API key,
# and the CUDA DLL directories have to be registered while the process can still add
# them. Only these imports may precede that; app.dependencies pulls in nothing heavier
# than yaml, so PROJECT_ROOT can come from there rather than being computed twice.
from app.dependencies import PROJECT_ROOT, load_settings
from app.runtime_environment import configure_windows_cuda_dlls, load_dotenv


LOADED_ENV_NAMES = load_dotenv(PROJECT_ROOT)
REGISTERED_CUDA_PATHS = configure_windows_cuda_dlls()

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import transformers

from app.api.routes import router
from app.services.pipeline_service import PipelineService
from app.services.ocr_service import GroqOCRService


LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

if LOADED_ENV_NAMES:
    LOGGER.info("Loaded from .env: %s", ", ".join(LOADED_ENV_NAMES))
if REGISTERED_CUDA_PATHS:
    LOGGER.info("Registered Windows CUDA DLL paths: %s", REGISTERED_CUDA_PATHS)
LOGGER.info(
    "Transformers runtime %s from %s",
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
    ocr_factory = ocr_service_factory or GroqOCRService

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.project_root = root
        application.state.settings = load_settings(root)
        application.state.pipeline_service = None
        application.state.ocr_service = ocr_factory(root)
        LOGGER.info("Starting ReviewTrust inference pipeline")
        try:
            application.state.pipeline_service = factory(root)
            LOGGER.info("Inference pipeline ready")
        except Exception as exc:
            LOGGER.exception("Inference pipeline initialization failed: %s", exc)
        _log_hosted_model_readiness(application)
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


def _log_hosted_model_readiness(application: FastAPI) -> None:
    """Say up front whether the hosted stages can run, rather than at first request."""
    configured = getattr(application.state.ocr_service, "is_configured", None)
    if configured is None:
        return
    if configured:
        LOGGER.info("Groq credentials found; explanation and image transcription are available")
    else:
        LOGGER.warning(
            "No Groq API key found. Classification and entity extraction will run, but the "
            "explanation and image-transcription stages will return 503 until a key is set."
        )


app = create_app()
