import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DataCollatorWithPadding
import json

class MyDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        jsonl_path: str,
        max_length: int
    ):
        self.tokenizer = tokenizer

        samples = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()

                if not line:
                    continue

                sample = json.loads(line)

                if "text" not in sample or "label" not in sample:
                    raise ValueError(
                        "Every JSONL sample must contain 'text' and 'label'."
                    )

                samples.append(sample)

        texts = [str(sample["text"]) for sample in samples]
        self.labels = [int(sample["label"]) for sample in samples]

        self.tokenized = self.tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding=False
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {
            key: torch.tensor(values[idx], dtype=torch.long)
            for key, values in self.tokenized.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


def dataloader(
    dataset: MyDataset,
    batch_size: int=32,
    num_workers: int=0,
    shuffle: bool=False,
):
    data_collator = DataCollatorWithPadding(
        tokenizer=dataset.tokenizer,
        return_tensors="pt"
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=data_collator
    )