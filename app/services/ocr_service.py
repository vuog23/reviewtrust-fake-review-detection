from __future__ import annotations

import io
import gc
import json
import logging
import os
import re
import subprocess
import sys
import threading
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import yaml


LOGGER = logging.getLogger(__name__)
WORKER_RESULT_PREFIX = "REVIEWTRUST_OCR_RESULT="


class OCRUnavailableError(RuntimeError):
    """Raised when the configured OCR model cannot be initialized."""


class OCRExtractionError(RuntimeError):
    """Raised when OCR returns no usable review text."""


class DeepSeekOCRService:
    """Lazy, serialized adapter for the official DeepSeek-OCR model."""

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()
        config_path = project_root / "config.yaml"
        with config_path.open("r", encoding="utf-8") as config_file:
            config = yaml.safe_load(config_file) or {}
        settings = config.get("ocr_inference", {})

        self.model_name = str(settings.get("model_name", "deepseek-ai/DeepSeek-OCR"))
        self.cache_dir = project_root / str(settings.get("cache_dir", "cache/deepseek-ocr"))
        self.worker_python = project_root / str(
            settings.get("worker_python", ".venv-ocr/Scripts/python.exe")
        )
        self.prompt = str(settings.get("prompt", "<image>\nFree OCR. "))
        self.trust_remote_code = bool(settings.get("trust_remote_code", True))
        self.use_safetensors = bool(settings.get("use_safetensors", True))
        self.load_in_4bit = bool(settings.get("load_in_4bit", True))
        self.attention_implementation = str(settings.get("attention_implementation", "eager"))
        self.base_size = int(settings.get("base_size", 1024))
        self.image_size = int(settings.get("image_size", 640))
        self.crop_mode = bool(settings.get("crop_mode", True))
        self.max_upload_bytes = int(settings.get("max_upload_mb", 10)) * 1024 * 1024
        self.auto_setup_worker = bool(settings.get("auto_setup_worker", True))

        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._tokenizer: Any = None
        self._model: Any = None
        self._lock = threading.Lock()
        self._environment_lock = threading.Lock()
        self._worker_verified = False
        self._dll_handles: list[Any] = []
        self._vision_hooks: list[Any] = []

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def extract_text(self, image_path: Path) -> str:
        """Run OCR in an isolated Transformers 4.46 subprocess."""
        if not self.worker_python.is_file() or not self._worker_verified:
            self.ensure_worker_environment()
        if not self.worker_python.is_file():
            raise OCRUnavailableError(
                "The DeepSeek-OCR worker environment could not be created automatically."
            )
        command = [
            str(self.worker_python),
            "-m",
            "app.ocr_worker",
            str(self.project_root),
            str(image_path),
        ]
        environment = os.environ.copy()
        environment["PYTHONUTF8"] = "1"
        completed = subprocess.run(
            command,
            cwd=self.project_root,
            env=environment,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=900,
            check=False,
        )
        if completed.returncode != 0:
            LOGGER.error("DeepSeek-OCR worker stderr:\n%s", completed.stderr.strip())
            raise OCRExtractionError(
                "The DeepSeek-OCR worker failed. Review the server log for the exact error."
            )
        for line in reversed(completed.stdout.splitlines()):
            if line.startswith(WORKER_RESULT_PREFIX):
                payload = json.loads(line[len(WORKER_RESULT_PREFIX):])
                text = _clean_ocr_text(str(payload.get("text", "")))
                if text:
                    return text
        LOGGER.error("DeepSeek-OCR worker returned no result marker:\n%s", completed.stdout)
        raise OCRExtractionError("DeepSeek-OCR did not return readable text.")

    def ensure_worker_environment(self) -> None:
        """Create or repair the isolated OCR environment when required."""
        if self._worker_verified and self.worker_python.is_file():
            return
        with self._environment_lock:
            if self._worker_is_compatible():
                self._worker_verified = True
                return
            if not self.auto_setup_worker:
                raise OCRUnavailableError(
                    "The DeepSeek-OCR worker is missing and automatic setup is disabled."
                )

            setup_script = self.project_root / "scripts" / "setup_ocr_env.py"
            if not setup_script.is_file():
                raise OCRUnavailableError(
                    "The OCR setup script is missing from scripts/setup_ocr_env.py."
                )
            LOGGER.info("Creating or repairing the DeepSeek-OCR worker environment")
            try:
                completed = subprocess.run(
                    [sys.executable, str(setup_script)],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=600,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise OCRUnavailableError(
                    "DeepSeek-OCR setup could not start or timed out."
                ) from exc
            if completed.returncode != 0 or not self._worker_is_compatible():
                LOGGER.error(
                    "OCR environment setup failed. stdout:\n%s\nstderr:\n%s",
                    completed.stdout.strip(),
                    completed.stderr.strip(),
                )
                raise OCRUnavailableError(
                    "DeepSeek-OCR setup failed. Check network access, disk space, and the server log."
                )
            self._worker_verified = True
            LOGGER.info("DeepSeek-OCR worker environment ready")

    def _worker_is_compatible(self) -> bool:
        if not self.worker_python.is_file():
            return False
        try:
            completed = subprocess.run(
                [
                    str(self.worker_python),
                    "-c",
                    (
                        "import transformers, tokenizers, huggingface_hub; "
                        "assert transformers.__version__ == '4.46.3'; "
                        "assert tokenizers.__version__ == '0.20.3'; "
                        "assert huggingface_hub.__version__ == '0.36.2'"
                    ),
                ],
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            return completed.returncode == 0
        except (OSError, subprocess.SubprocessError):
            return False

    def extract_text_in_process(self, image_path: Path) -> str:
        with self._lock:
            self._ensure_model()
            try:
                with TemporaryDirectory(prefix="reviewtrust-ocr-") as output_dir:
                    captured = io.StringIO()
                    with redirect_stdout(captured), redirect_stderr(captured):
                        result = self._model.infer(
                            self._tokenizer,
                            prompt=self.prompt,
                            image_file=str(image_path),
                            output_path=output_dir,
                            base_size=self.base_size,
                            image_size=self.image_size,
                            crop_mode=self.crop_mode,
                            save_results=False,
                            test_compress=False,
                            eval_mode=True,
                        )
                    if captured.getvalue().strip():
                        LOGGER.debug("DeepSeek-OCR output: %s", captured.getvalue().strip())
            except Exception as exc:
                self._clear_cuda_cache()
                raise OCRExtractionError(
                    "DeepSeek-OCR could not process this image. Check GPU memory and the server log."
                ) from exc

        text = _clean_ocr_text(str(result or ""))
        if not text:
            raise OCRExtractionError("DeepSeek-OCR did not find readable text in the image.")
        return text

    def unload(self) -> None:
        """Release OCR weights so the text-analysis models can return to the GPU."""
        with self._lock:
            self._model = None
            self._tokenizer = None
            self._vision_hooks.clear()
            gc.collect()
            self._clear_cuda_cache()
            LOGGER.info("DeepSeek-OCR unloaded")

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            self._configure_windows_cuda_dlls()
            import torch
            from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

            if not torch.cuda.is_available():
                raise OCRUnavailableError(
                    "The official DeepSeek-OCR inference implementation requires an NVIDIA CUDA GPU."
                )

            LOGGER.info("Loading DeepSeek-OCR model %s", self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=str(self.cache_dir),
                trust_remote_code=self.trust_remote_code,
            )
            model_arguments: dict[str, Any] = {
                "cache_dir": str(self.cache_dir),
                "trust_remote_code": self.trust_remote_code,
                "use_safetensors": self.use_safetensors,
                "_attn_implementation": self.attention_implementation,
                "device_map": "auto",
            }
            if self.load_in_4bit:
                model_arguments["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                model_arguments["torch_dtype"] = torch.bfloat16

            self._model = AutoModel.from_pretrained(self.model_name, **model_arguments)
            self._model.eval()
            if self.load_in_4bit:
                self._align_vision_dtype()
            LOGGER.info("DeepSeek-OCR ready")
        except OCRUnavailableError:
            raise
        except Exception as exc:
            self._model = None
            self._tokenizer = None
            self._clear_cuda_cache()
            raise OCRUnavailableError(
                "DeepSeek-OCR could not be loaded. Verify the pinned Transformers "
                "dependencies and available GPU memory; see the server log for the exact cause."
            ) from exc

    def _configure_windows_cuda_dlls(self) -> None:
        if os.name != "nt" or self._dll_handles:
            return
        candidates = [
            Path(sys.prefix) / "bin",
            Path(sys.prefix) / "Library" / "bin",
            Path(sys.base_prefix) / "bin",
            Path(sys.base_prefix) / "Library" / "bin",
        ]
        try:
            import torch

            candidates.append(Path(torch.__file__).resolve().parent / "lib")
        except Exception:
            pass
        for directory in candidates:
            if directory.is_dir():
                self._dll_handles.append(os.add_dll_directory(str(directory)))
                os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"

    def _align_vision_dtype(self) -> None:
        """Match unquantized vision outputs to quantized language embeddings."""
        core_model = getattr(self._model, "model", None)
        embeddings = getattr(core_model, "embed_tokens", None)
        if core_model is None or embeddings is None:
            return
        target_dtype = embeddings.weight.dtype
        for attribute in ("sam_model", "vision_model", "projector"):
            module = getattr(core_model, attribute, None)
            if module is not None:
                module.to(dtype=target_dtype)
        projector = getattr(core_model, "projector", None)
        if projector is not None:
            self._vision_hooks.append(
                projector.register_forward_hook(
                    lambda _module, _inputs, output: output.to(dtype=target_dtype)
                )
            )
        for attribute in ("image_newline", "view_seperator"):
            parameter = getattr(core_model, attribute, None)
            if parameter is not None:
                parameter.data = parameter.data.to(dtype=target_dtype)
        LOGGER.info("DeepSeek-OCR vision modules aligned to %s", target_dtype)

    @staticmethod
    def _clear_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _clean_ocr_text(value: str) -> str:
    value = re.sub(r"<\|[^>]+\|>", "", value)
    value = re.sub(r"!\[[^]]*]\([^)]*\)", "", value)
    value = value.replace("```markdown", "").replace("```", "")
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
