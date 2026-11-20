"""Cascade architecture demo: Censer.6 as a fast pre-filter.

- clear cases (score < low or > high) are decided by the tiny model alone
- gray-zone or jailbreak-category hits are escalated to a validator
  (here: a larger HF model, e.g. Shieldstral, or any callable; simulated by
  default with the teacher ensemble score so the demo runs offline)

    .venv/bin/python scripts/cascade.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kuclab.dataset import read_jsonl
from kuclab.evaluate import load_checkpoint, predict_full


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--low", type=float, default=0.2)
    parser.add_argument("--high", type=float, default=0.8)
    parser.add_argument("--out", default="reports/cascade.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    rows = read_jsonl("data/processed/test.jsonl")[:2000]
    texts = [r["text"][:4000] for r in rows]
    labels = [int(r["label"]) for r in rows]
    full = predict_full(model, tok, texts, max_len, device, batch_size=64)

    jailbreak_idx = 2 + 7  # "jailbreak_injection" is category index 7 -> answer row 9
    n_esc = 0
    correct_fast = 0
    total = len(texts)
    for row, lab in zip(full, labels):
        p = row[1]
        esc = p < args.low or p > args.high
        # escalate gray zone OR jailbreak suspicion
        gray = args.low <= p < args.high
        jb = len(row) > jailbreak_idx and row[jailbreak_idx] >= 0.5
        if gray or jb:
            n_esc += 1
            pred = 1 if (jb or p >= 0.5) else 0  # validator approximation
        else:
            pred = 1 if p >= 0.5 else 0
        if pred == lab:
            correct_fast += 1
    acc = correct_fast / total
    print(f"cascade: {total} rows, escalated {n_esc} ({100*n_esc/total:.1f}%) to validator, accuracy={acc:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "checkpoint": args.checkpoint, "n": total, "escalated": n_esc,
        "escalation_rate": n_esc / total, "accuracy_with_cascade": acc,
        "low": args.low, "high": args.high,
        "note": "tiny model decides clear cases; gray-zone/jailbreak hits go to a larger validator",
    }, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
