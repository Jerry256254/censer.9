"""Per-category threshold tuning + calibration (ECE) on the held-out catval set.

For each of the 10 categories, sweep thresholds on catval.jsonl and pick the one
maximizing F1; compute ECE and draw a reliability diagram (figs/reliability.png).
Writes reports/calibration.json with per-category thresholds — these go into the
bundle config as `cat_thresholds` so production inference uses calibrated cutoffs.

    .venv/bin/python scripts/calibrate_categories.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kuclab.categories import CATEGORIES
from kuclab.dataset import read_jsonl
from kuclab.evaluate import load_checkpoint, predict_full


def ece(probs, labels, n_bins=10):
    import numpy as np

    probs = np.asarray(probs)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    ece_v = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if m.sum() == 0:
            continue
        ece_v += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(ece_v)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--catval", default="data/processed/catval.jsonl")
    parser.add_argument("--out", default="reports/calibration.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    rows = read_jsonl(args.catval)
    texts = [r["text"][:4000] for r in rows]
    bin_labels = [int(r["label"]) for r in rows]
    # cat ground truth: multi-hot from row["cats"] (indices)
    n_cat = len(CATEGORIES)
    cat_true = []
    for r in rows:
        v = [0.0] * n_cat
        for c in r.get("cats") or []:
            v[int(c)] = 1.0
        cat_true.append(v)

    full = predict_full(model, tok, texts, mcfg.max_seq_len, device, batch_size=64)
    # full rows: [safe, unsafe, cat0..cat9] probabilities (sigmoid for >2 answers)
    cat_probs = [[row[2 + i] for row in full] for i in range(n_cat)]

    result = {"checkpoint": args.checkpoint, "n": len(rows), "categories": {}, "binary_ece": ece([r[1] for r in full], bin_labels)}
    thresholds = {}
    for i, cat in enumerate(CATEGORIES):
        y = cat_true and [c[i] for c in cat_true]
        probs = cat_probs[i]
        best = (0.5, -1.0)
        for t in [x / 100 for x in range(20, 96)]:
            tp = sum(1 for p, l in zip(probs, y) if p >= t and l == 1)
            fp = sum(1 for p, l in zip(probs, y) if p >= t and l == 0)
            fn = sum(1 for p, l in zip(probs, y) if p < t and l == 1)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best[1]:
                best = (t, f1)
        thresholds[cat] = best[0]
        result["categories"][cat] = {"best_threshold": best[0], "f1_at_best": best[1],
                                     "ece": ece(probs, y), "prevalence": sum(y) / len(y)}
        print(f"{cat:22s} thr={best[0]:.2f} f1={best[1]:.4f} ece={result['categories'][cat]['ece']:.4f} prev={sum(y)/len(y):.3f}")

    result["cat_thresholds"] = thresholds
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # reliability diagram for binary unsafe score
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        probs = np.array([r[1] for r in full])
        labels = np.array(bin_labels)
        bins = np.linspace(0, 1, 11)
        xs, ys, ws = [], [], []
        for i in range(10):
            m = (probs > bins[i]) & (probs <= bins[i + 1])
            if m.sum() == 0:
                continue
            xs.append(probs[m].mean())
            ys.append(labels[m].mean())
            ws.append(m.mean())
        fig, ax = plt.subplots(figsize=(4.5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        ax.scatter(xs, ys, s=[w * 400 for w in ws], c="tab:red", zorder=3)
        ax.set_xlabel("predicted P(unsafe)")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Censer.7 reliability (ECE={result['binary_ece']:.4f})\ncatval n={len(rows)}")
        figs = Path("reports/figs")
        figs.mkdir(parents=True, exist_ok=True)
        fig.savefig(figs / "reliability.png", dpi=130)
        plt.close(fig)
        print("wrote reports/figs/reliability.png")
    except Exception as exc:
        print("reliability plot skipped:", exc)

    print("wrote", out)


if __name__ == "__main__":
    main()
