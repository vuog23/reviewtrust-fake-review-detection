from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any


_CUDA_DLL_HANDLES: list[Any] = []


def load_dotenv(project_root: Path) -> list[str]:
    """Load KEY=VALUE pairs from a gitignored .env into the environment.

    The project file deliberately wins over an existing environment variable: a
    stale key exported in some shell months ago should not silently shadow the one
    sitting in the project. Callers log which names were applied so the source of a
    credential is never a mystery.
    """
    env_path = project_root / ".env"
    if not env_path.is_file():
        return []

    applied: list[str] = []
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        value = value.strip().strip('"').strip("'")
        if not name:
            continue
        os.environ[name] = value
        applied.append(name)
    return applied


def configure_windows_cuda_dlls() -> list[str]:
    """Keep Conda/PyTorch CUDA DLL directories registered for this process.

    The classifier and NER models still run locally on the GPU, so the DLL search
    path has to include the Conda and Torch library directories on Windows.
    """
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
