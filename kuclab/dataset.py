from __future__ import annotations

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .tokenizer import KucTokenizer


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _typo(text: str, rng: random.Random) -> str:
    """One random single-character edit: deletion, substitution or duplication."""
    if len(text) < 5:
        return text
    i = rng.randrange(len(text))
    op = rng.random()
    if op < 0.4:
        return text[:i] + text[i + 1 :]
    if op < 0.8:
        return text[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + text[i + 1 :]
    return text[: i + 1] + text[i] + text[i + 1 :]


class SafetyDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tok: KucTokenizer,
        max_len: int,
        augment_typo: float = 0.0,
        seed: int = 0,
        soft: "list[float] | None" = None,
        augment_adversarial: float = 0.0,
    ) -> None:
        self.rows = read_jsonl(path)
        self.tok = tok
        self.max_len = max_len
        self.augment_typo = augment_typo
        self.seed = seed
        self.soft = soft
        self.augment_adversarial = augment_adversarial

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        text = row["text"]
        if self.augment_typo or self.augment_adversarial:
            rng = random.Random(f"{self.seed}:{index}")
            if self.augment_typo and rng.random() < self.augment_typo:
                text = _typo(text, rng)
            if self.augment_adversarial and int(row["label"]) == 1 and rng.random() < self.augment_adversarial:
                from .multilingual import adversarial_variants

                text = adversarial_variants(text, index, n=1)[0]
        ids = self.tok.encode_prompt(text, self.max_len)
        label = int(row["label"])
        cats = row.get("cats")
        label0 = int(row["label"])
        n_cat = 10
        if label0 == 0:
            # safe rows: all category targets are explicit zeros (full mask)
            cats_vec = [0.0] * n_cat
            mask_vec = [1.0] * n_cat
        elif cats is None:
            cats_vec = [0.0] * n_cat
            mask_vec = [0.0] * n_cat
        else:
            # cats is a list of category INDICES present in the text -> multi-hot
            cats_vec = [0.0] * n_cat
            for c in cats:
                ci = int(c)
                if 0 <= ci < n_cat:
                    cats_vec[ci] = 1.0
            mask_vec = [1.0] * n_cat
        return {
            "input_ids": ids,
            "answer_id": self.tok.answer_ids[label],
            "label": label,
            "source": row.get("source", ""),
            "soft": float(self.soft[index]) if self.soft is not None else float(label),
            "cats": cats_vec,
            "cats_mask": mask_vec,
        }


def collate(batch: list[dict], pad_id: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
    answer_ids = torch.tensor([item["answer_id"] for item in batch], dtype=torch.long)
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    return input_ids, answer_ids, labels


class Batcher:
    """Token-budget aware batches so long inputs do not blow up VRAM."""

    def __init__(self, dataset: SafetyDataset, batch_size: int, max_tokens: int, shuffle: bool, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.rng = random.Random(seed)

    def __iter__(self):
        order = list(range(len(self.dataset)))
        if self.shuffle:
            self.rng.shuffle(order)
        batch: list[dict] = []
        tokens = 0
        longest = 0
        for index in order:
            item = self.dataset[index]
            length = len(item["input_ids"])
            new_longest = max(longest, length)
            if batch and ((len(batch) + 1) * new_longest > self.max_tokens or len(batch) >= self.batch_size):
                yield batch
                batch, tokens, longest = [], 0, 0
                new_longest = length
            batch.append(item)
            longest = new_longest
            tokens = len(batch) * longest
        if batch:
            yield batch

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
