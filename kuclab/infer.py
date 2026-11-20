from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .evaluate import load_checkpoint, predict_full, predict_proba_chunked


@dataclass
class Verdict:
    label: str
    probability: float
    threshold: float
    categories: list[str] | None = None
    chunks: int = 1

    @property
    def is_unsafe(self) -> bool:
        return self.label == "1"

    def as_dict(self) -> dict:
        d = {"label": self.label, "unsafe": self.is_unsafe, "score": round(self.probability, 4)}
        if self.categories:
            d["categories"] = self.categories
        if self.chunks > 1:
            d["chunks"] = self.chunks
        return d


class SafetyModel:
    """Inference wrapper: text in, '0' or '1' out."""

    def __init__(self, checkpoint: str | Path = "checkpoints/best.pt", threshold: float = 0.5, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.tokenizer, self.model_config, self.state = load_checkpoint(checkpoint, self.device)
        self.max_chars = int(self.state.get("max_chars", 500))
        self.max_len = self.model_config.max_seq_len
        self.threshold = threshold
        self.temperature = float(self.state.get("temperature", 1.0) or 1.0)

    def _cats(self) -> list[str]:
        from .categories import CATEGORIES

        return CATEGORIES if len(self.model_config.answer_ids) > 2 else []

    def predict(self, text: str) -> Verdict:
        probs = predict_proba_chunked(self.model, self.tokenizer, [text[: self.max_chars * 8]], self.max_len, self.device, temperature=self.temperature)[0]
        cats = self._cats()
        row = predict_full(self.model, self.tokenizer, [text[: self.max_chars]], self.max_len, self.device, temperature=self.temperature)[0]
        flagged = [c for c, p in zip(cats, row[2:]) if p >= self.threshold] if (cats and probs >= self.threshold) else None
        return Verdict(label="1" if probs >= self.threshold else "0", probability=probs,
                       threshold=self.threshold, categories=flagged)

    def predict_batch(self, texts: list[str]) -> list[Verdict]:
        trimmed = [t[: self.max_chars * 8] for t in texts]
        probs = predict_proba_chunked(self.model, self.tokenizer, trimmed, self.max_len, self.device, temperature=self.temperature)
        return [
            Verdict(label="1" if p >= self.threshold else "0", probability=p, threshold=self.threshold)
            for p in probs
        ]
