from __future__ import annotations

import json
import sys
from pathlib import Path

from app.services.ocr_service import DeepSeekOCRService


RESULT_PREFIX = "REVIEWTRUST_OCR_RESULT="


def main() -> int:
    if len(sys.argv) != 3:
        print("Usage: python -m app.ocr_worker PROJECT_ROOT IMAGE_PATH", file=sys.stderr)
        return 2
    project_root = Path(sys.argv[1]).resolve()
    image_path = Path(sys.argv[2]).resolve()
    service = DeepSeekOCRService(project_root)
    try:
        text = service.extract_text_in_process(image_path)
        print(f"{RESULT_PREFIX}{json.dumps({'text': text}, ensure_ascii=False)}")
        return 0
    finally:
        service.unload()


if __name__ == "__main__":
    raise SystemExit(main())
