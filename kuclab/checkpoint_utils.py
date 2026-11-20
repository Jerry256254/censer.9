"""Helpers for building throwaway checkpoints in tests."""

from __future__ import annotations

from pathlib import Path

import torch

from .model import KucLabCneser, ModelConfig
from .tokenizer import KucTokenizer, train_tokenizer

CORPUS = [
    "hello how are you today my friend",
    "please explain how to bake a chocolate cake",
    "the weather is nice and sunny outside",
    "suck my dick is a rude phrase",
    "how do I kill a process in linux",
    "the quick brown fox jumps over the lazy dog",
] * 60


def make_test_checkpoint(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    tok_dir = out / "tokenizer"
    tok = train_tokenizer(CORPUS, vocab_size=256, min_frequency=1, out_dir=tok_dir)

    cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        n_layers=2,
        n_heads=2,
        d_model=32,
        d_ff=64,
        max_seq_len=64,
        dropout=0.0,
        pad_id=tok.pad_id,
        answer_ids=tuple(tok.answer_ids),
    )
    torch.manual_seed(0)
    model = KucLabCneser(cfg)

    # keep the test checkpoint's temperature in sync with reports/calibration.json
    # (build_bundle writes that temperature into the bundle config.json)
    temperature = 1.0
    try:
        import json

        cal = Path(__file__).resolve().parent.parent / "reports" / "calibration.json"
        if cal.exists():
            temperature = float(json.loads(cal.read_text()).get("temperature", 1.0))
    except Exception:
        pass

    path = out / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_config": cfg.__dict__,
            "metrics": {},
            "step": 0,
            "epoch": 0,
            "best_f1": 0.0,
            "max_chars": 500,
            "name": "Censer.9",
            "temperature": temperature,
        },
        path,
    )
    return path
