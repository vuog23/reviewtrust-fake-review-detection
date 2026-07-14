from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

from app.services.ocr_service import DeepSeekOCRService
from scripts.setup_ocr_env import load_ocr_requirements


def _service(tmp_path):
    (tmp_path / "config.yaml").write_text(
        "ocr_inference:\n"
        "  worker_python: .venv-ocr/Scripts/python.exe\n"
        "  auto_setup_worker: true\n",
        encoding="utf-8",
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "setup_ocr_env.py").write_text("", encoding="utf-8")
    return DeepSeekOCRService(tmp_path)


def test_compatible_ocr_worker_skips_setup(tmp_path, monkeypatch):
    service = _service(tmp_path)
    monkeypatch.setattr(service, "_worker_is_compatible", lambda: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("setup must not run")),
    )

    service.ensure_worker_environment()

    assert service._worker_verified is True


def test_missing_ocr_worker_runs_automatic_setup(tmp_path, monkeypatch):
    service = _service(tmp_path)
    compatibility = iter((False, True))
    monkeypatch.setattr(service, "_worker_is_compatible", lambda: next(compatibility))
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout="ready", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    service.ensure_worker_environment()

    assert service._worker_verified is True
    assert calls[0][0][0] == sys.executable
    assert calls[0][0][1].endswith("setup_ocr_env.py")


def test_ocr_worker_pins_are_embedded_in_main_requirements():
    assert load_ocr_requirements() == [
        "transformers==4.46.3",
        "tokenizers==0.20.3",
        "huggingface-hub==0.36.2",
    ]
