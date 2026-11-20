"""Precompute teacher soft probabilities for the training split (knowledge distillation).

Runs one or more reference checkpoints over data/processed/train.jsonl and stores the
(weighted) mean probability of "unsafe" per row as a .npy file aligned with the rows.

    .venv/bin/python -m kuclab.distill \
        --teachers checkpoints_ref/censer3.pt:checkpoints_ref/tokenizer_censer3,...

Writes data/processed/train_soft.npy by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from .model import remap_legacy_state

from .dataset import read_jsonl
from .model import remap_legacy_state
from .evaluate import predict_proba
from .model import KucLabCneser, ModelConfig
from .tokenizer import KucTokenizer


def load_ref(ckpt: str | Path, tok_dir: str | Path, device: torch.device):
    state = torch.load(ckpt, map_location=device, weights_only=False)
    mcfg = ModelConfig(**state["model_config"])
    model = KucLabCneser(mcfg).to(device)
    model.load_state_dict(remap_legacy_state(state)["model"])
    model.eval()
    tok = KucTokenizer.load(tok_dir)
    return model, tok, mcfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute teacher soft labels for distillation.")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--out", default=None, help="default: <data>/train_soft.npy")
    parser.add_argument(
        "--teachers",
        default=(
            "checkpoints_ref/censer5.pt:checkpoints_ref/tokenizer_censer5,"
            "checkpoints_ref/censer4.pt:checkpoints_ref/tokenizer_censer4,"
            "checkpoints_ref/censer3.pt:checkpoints_ref/tokenizer_censer3,"
            "checkpoints_ref/censer2.pt:checkpoints_ref/tokenizer_censer2,"
            "checkpoints_ref/censer1.pt:checkpoints_ref/tokenizer_censer1"
        ),
        help="comma-separated ckpt:tokenizer_dir list",
    )
    parser.add_argument("--weights", default="", help="comma-separated floats, default uniform")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = Path(args.data)
    rows = read_jsonl(data_dir / "train.jsonl")
    texts = [r["text"] for r in rows]
    print(f"device={device} rows={len(rows)} teachers={args.teachers}")

    specs = [s.split(":") for s in args.teachers.split(",") if s.strip()]
    weights = [float(w) for w in args.weights.split(",")] if args.weights else [1.0] * len(specs)
    assert len(weights) == len(specs)
    total_w = sum(weights)

    soft = np.zeros(len(rows), dtype=np.float64)
    for (ck, tk), w in zip(specs, weights):
        print(f"teacher {ck} (weight {w}) ...", flush=True)
        model, tok, mcfg = load_ref(ck, tk, device)
        probs = predict_proba(model, tok, texts, mcfg.max_seq_len, device, batch_size=args.batch_size)
        soft += w * np.asarray(probs, dtype=np.float64)
        del model
        torch.cuda.empty_cache()
    soft /= total_w

    out = Path(args.out) if args.out else data_dir / "train_soft.npy"
    np.save(out, soft.astype(np.float32))
    print(f"wrote {out} shape={soft.shape} mean={soft.mean():.4f} "
          f"min={soft.min():.4f} max={soft.max():.4f}")


if __name__ == "__main__":
    main()
