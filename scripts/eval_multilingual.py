"""Multilingual evaluation: per-language binary F1.

Two sources:
1. textdetox/multilingual_toxicity_dataset — one split per language (de, es, fr, it,
   pt, ru, zh, ar, hi, ...); we sample up to --per-lang rows per language. This set is
   NEVER used for training.
2. Held-out synthetic cs/de/es/fr probes generated with a different seed than training.

Writes reports/multilingual.json + .md.

    .venv/bin/python scripts/eval_multilingual.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from kuclab.evaluate import load_checkpoint, metrics_from, predict_proba
from kuclab.multilingual import generate_ml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--per-lang", type=int, default=800)
    parser.add_argument("--out", default="reports/multilingual.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    results = {}

    # 1. textdetox per-language splits
    try:
        from datasets import load_dataset

        ds = load_dataset("textdetox/multilingual_toxicity_dataset")
        for lang in sorted(ds.keys()):
            split = ds[lang]
            if len(split) > args.per_lang:
                idx = random.Random(13).sample(range(len(split)), args.per_lang)
                split = split.select(idx)
            texts = [str(r["text"])[:4000] for r in split]
            labels = [int(r["toxic"]) for r in split]
            if sum(labels) == 0:
                continue
            probs = predict_proba(model, tok, texts, max_len, device, batch_size=64)
            m = metrics_from(labels, probs, 0.5)
            best = (0.5, m)
            for i in range(5, 96):
                mm = metrics_from(labels, probs, i / 100)
                if mm["f1"] > best[1]["f1"]:
                    best = (i / 100, mm)
            results[f"textdetox_{lang}"] = {"n": len(labels), "unsafe": sum(labels),
                                             "f1@0.5": m["f1"], "acc@0.5": m["accuracy"],
                                             "best_f1": best[1]["f1"], "best_thr": best[0]}
            print(f"textdetox_{lang}: n={len(labels)} u={sum(labels)} f1@0.5={m['f1']:.4f} best={best[1]['f1']:.4f}@{best[0]:.2f}", flush=True)
    except Exception as exc:
        print("textdetox skipped:", str(exc)[:150])

    # 2. held-out synthetic cs/de/es/fr (different seed than training data)
    rows = generate_ml(count_per_lang=150, seed=999)  # seed 999 != training seed; rows are (text, label, source)
    for lang in ["cs", "de", "es", "fr"]:
        sel = [(t, l) for t, l, src in rows if src.endswith("_" + lang)]
        texts = [t for t, _ in sel]
        labels = [l for _, l in sel]
        if not texts:
            continue
        probs = predict_proba(model, tok, texts, max_len, device, batch_size=64)
        m = metrics_from(labels, probs, 0.5)
        results[f"synth_{lang}"] = {"n": len(labels), "unsafe": sum(labels), "f1@0.5": m["f1"], "acc@0.5": m["accuracy"]}
        print(f"synth_{lang}: n={len(labels)} f1@0.5={m['f1']:.4f} acc={m['accuracy']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=2))
    md = ["# Multilingual eval", "", "| set | n | unsafe | F1@0.5 | acc@0.5 | best F1 |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in results.items():
        md.append(f"| {k} | {v['n']} | {v['unsafe']} | {v.get('f1@0.5',0):.4f} | {v.get('acc@0.5',0):.4f} | {v.get('best_f1','—')} |")
    (Path("reports/multilingual.md")).write_text("\n".join(md) + "\n")
    print("wrote", out, "and reports/multilingual.md")


if __name__ == "__main__":
    main()
