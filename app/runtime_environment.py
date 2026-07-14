from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any


_OCR_DEPENDENCY_PREFIXES = ("transformers", "tokenizers", "huggingface_hub")
_CUDA_DLL_HANDLES: list[Any] = []


def isolate_web_runtime(project_root: Path) -> list[str]:
    """Keep the OCR-only site-packages directory out of the web process.

    The OCR worker intentionally pins an older Transformers release. A shell with
    that virtual environment on PATH (or PYTHONPATH) must not leak those packages
    into the main ModernBERT process.
    """
    ocr_environment = (project_root / ".venv-ocr").resolve()
    retained_paths: list[str] = []
    removed_paths: list[str] = []

    for entry in sys.path:
        if entry and _is_within(Path(entry), ocr_environment):
            removed_paths.append(entry)
        else:
            retained_paths.append(entry)
    sys.path[:] = retained_paths

    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith(_OCR_DEPENDENCY_PREFIXES):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file and _is_within(Path(module_file), ocr_environment):
            del sys.modules[module_name]

    if removed_paths:
        importlib.invalidate_caches()
    return removed_paths


def configure_windows_cuda_dlls() -> list[str]:
    """Keep Conda/PyTorch CUDA DLL directories registered for this process."""
    if os.name != "nt" or _CUDA_DLL_HANDLES:
        return []

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

    registered: list[str] = []
    seen: set[Path] = set()
    for directory in candidates:
        directory = directory.resolve()
        if directory in seen or not directory.is_dir():
            continue
        seen.add(directory)
        _CUDA_DLL_HANDLES.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"
        registered.append(str(directory))
    return registered


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent)
        return True
    except (OSError, ValueError):
        return False
