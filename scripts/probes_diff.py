"""Probe comparison across versions: which of the 97 hand-written probes each model gets wrong.

    .venv/bin/python scripts/probes_diff.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from kuclab.evaluate import load_checkpoint, predict_proba
from kuclab.probes import PROBES

CKPTS = [
    ("Censer.5", "/home/tvadmin/censer.5/checkpoints/best.pt", "/home/tvadmin/censer.5/checkpoints/tokenizer"),
    ("Censer.6", "/home/tvadmin/censer.6/checkpoints/best.pt", "/home/tvadmin/censer.6/checkpoints/tokenizer"),
    ("Censer.9", "checkpoints8/best.pt", "checkpoints8/tokenizer"),
    ("Censer.7", "/home/tvadmin/censer.7/checkpoints/best.pt", "/home/tvadmin/censer.7/checkpoints/tokenizer"),
]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = [t for t, _ in PROBES]
    labels = [l for _, l in PROBES]
    out = {}
    misses = {}
    for name, ck, tk in CKPTS:
        if not Path(ck).exists():
            continue
        model, tok, mcfg, state = load_checkpoint(ck, device, tk)
        probs = predict_proba(model, tok, texts, mcfg.max_seq_len, device, batch_size=64)
        wrong = [texts[i] for i in range(len(texts)) if (probs[i] >= 0.5) != bool(labels[i])]
        acc = 1 - len(wrong) / len(texts)
        out[name] = {"correct": len(texts) - len(wrong), "total": len(texts), "accuracy": acc}
        misses[name] = wrong
        print(f"{name}: {len(texts)-len(wrong)}/{len(texts)} ({acc:.1%})")
        for w in wrong:
            print(f"   MISS: {w!r}")
    if "Censer.9" in misses and "Censer.5" in misses:
        fixed = [m for m in misses["Censer.5"] if m not in misses["Censer.9"]]
        regressed = [m for m in misses["Censer.9"] if m not in misses["Censer.5"]]
        out["diff_5_vs_8"] = {"fixed_by_7": fixed, "regressed_in_7": regressed}
        print("fixed by .8 vs .5:", fixed)
        print("regressed in .8 vs .5:", regressed)
    Path("reports/probes_diff.json").write_text(json.dumps({"results": out, "misses": misses}, indent=2, ensure_ascii=False))
    print("wrote reports/probes_diff.json")


if __name__ == "__main__":
    main()
