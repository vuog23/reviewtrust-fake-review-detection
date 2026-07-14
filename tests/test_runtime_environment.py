from __future__ import annotations

import sys

from app.runtime_environment import isolate_web_runtime


def test_isolate_web_runtime_removes_ocr_environment_paths(tmp_path, monkeypatch):
    ocr_root = tmp_path / ".venv-ocr"
    ocr_site_packages = ocr_root / "Lib" / "site-packages"
    ocr_site_packages.mkdir(parents=True)
    ordinary_path = tmp_path / "ordinary"
    ordinary_path.mkdir()
    monkeypatch.setattr(
        sys,
        "path",
        [str(ordinary_path), str(ocr_root), str(ocr_site_packages)],
    )

    removed = isolate_web_runtime(tmp_path)

    assert sys.path == [str(ordinary_path)]
    assert removed == [str(ocr_root), str(ocr_site_packages)]
