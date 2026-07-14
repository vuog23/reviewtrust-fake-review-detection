import json
from pathlib import Path
import pandas as pd

def preprocessing(root_path: str, max_chars: int=2000):
    root_path = Path(root_path)
    dataset_path = root_path / "product_reviews"
    output_path = root_path / "preprocessed"

    output_path.mkdir(parents=True, exist_ok=True)

    preprocessed_dict = {}

    for split in dataset_path.iterdir():
        split_name = split.stem
        preprocessed_samples = []

        df = pd.read_json(split, lines=True)
        for text, label in zip(df['text'], df['label']):
            preprocessed_samples.append(
                {
                    "text": str(text.strip()[:max_chars]),
                    "label": int(label)
                }
            )
        preprocessed_dict[split_name] = preprocessed_samples

        save_path = output_path / f"{split_name}.jsonl"

        with save_path.open("w", encoding="utf-8") as file:
            for sample in preprocessed_samples:
                file.write(json.dumps(sample, ensure_ascii=False) + "\n")

        print(f"Saved {len(preprocessed_samples)} preprocessed samples to {save_path}")