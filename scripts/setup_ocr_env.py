from __future__ import annotations

import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ENVIRONMENT = ROOT / ".venv-ocr"
PYTHON = ENVIRONMENT / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
REQUIREMENTS = ROOT / "requirements.txt"
OCR_REQUIREMENT_PREFIX = "# ocr-worker:"


def load_ocr_requirements() -> list[str]:
    requirements = [
        line[len(OCR_REQUIREMENT_PREFIX):].strip()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.startswith(OCR_REQUIREMENT_PREFIX)
    ]
    if not requirements:
        raise RuntimeError("No OCR worker requirements were found in requirements.txt")
    return requirements


def main() -> None:
    if not PYTHON.is_file():
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(ENVIRONMENT)
    ocr_requirements = load_ocr_requirements()
    subprocess.run(
        [
            str(PYTHON), "-m", "pip", "install", "--upgrade", "--ignore-installed",
            "--no-deps", *ocr_requirements,
        ],
        check=True,
        cwd=ROOT,
    )
    subprocess.run(
        [
            str(PYTHON), "-c",
            "import torch, transformers, tokenizers, huggingface_hub; "
            "assert transformers.__version__ == '4.46.3'; "
            "assert tokenizers.__version__ == '0.20.3'; "
            "assert huggingface_hub.__version__ == '0.36.2'; "
            "print('OCR environment ready:', torch.__version__, "
            "transformers.__version__, tokenizers.__version__, huggingface_hub.__version__)",
        ],
        check=True,
        cwd=ROOT,
    )


if __name__ == "__main__":
    main()
