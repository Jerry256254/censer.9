"""Full calibration for Censer.9: binary temperature scaling (on val) + per-category
thresholds (on catval, which has real per-category labels from Aegis/Jigsaw/BeaverTails/
HateXplain-train) + ECE + reliability diagram.

Writes reports/calibration.json (with `temperature` and `cat_thresholds`) and
reports/figs/reliability.png. The bundle's config.json picks these up via upload_hf.py.

    .venv/bin/python scripts/calibrate_full.py --checkpoint checkpoints8/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kuclab.categories import CATEGORIES
from kuclab.dataset import read_jsonl
from kuclab.evaluate import load_checkpoint, predict_full


def ece(probs, labels, n_bins=10):
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for i in range(n_bins):
        m = (probs > bins[i]) & (probs <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += m.mean() * abs(probs[m].mean() - labels[m].mean())
    return float(e)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--val", default="data/processed/val.jsonl")
    parser.add_argument("--catval", default="data/processed/catval.jsonl")
    parser.add_argument("--out", default="reports/calibration.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    # ---- binary temperature scaling on val ----
    val = read_jsonl(args.val)
    vtexts = [r["text"][:4000] for r in val]
    vlabels = [int(r["label"]) for r in val]
    logits = predict_full(model, tok, vtexts, max_len, device, batch_size=128)
    # predict_full returns probabilities; recompute raw binary logit via model
    import torch as T
    bin_logit = []
    with T.no_grad():
        for i in range(0, len(vtexts), 128):
            ids = [tok.encode_prompt(t, max_len) for t in vtexts[i:i + 128]]
            width = max(len(x) for x in ids)
            x = T.full((len(ids), width), tok.pad_id, dtype=T.long)
            for j, row in enumerate(ids):
                x[j, : len(row)] = T.tensor(row)
            bin_logit.extend(model.answer_logits(x.to(device))[:, 1].float().tolist())
    bin_logit = T.tensor(bin_logit)
    y = T.tensor(vlabels, dtype=T.float32)

    def nll_at_T(Tv):
        z = bin_logit / Tv
        return T.nn.functional.binary_cross_entropy_with_logits(z, y).item()

    # golden-section search on log T in [0.05, 5]
    import math
    lo, hi = math.log(0.05), math.log(5.0)
    gr = (math.sqrt(5) - 1) / 2
    for _ in range(60):
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        if nll_at_T(math.exp(c)) < nll_at_T(math.exp(d)):
            hi = d
        else:
            lo = c
    temperature = math.exp((lo + hi) / 2)
    p_raw = T.sigmoid(bin_logit).numpy()
    p_cal = T.sigmoid(bin_logit / temperature).numpy()
    ece_raw = ece(p_raw, vlabels)
    ece_cal = ece(p_cal, vlabels)
    print(f"temperature: {temperature:.4f}  val ECE raw={ece_raw:.4f} -> calibrated={ece_cal:.4f}")

    # ---- per-category thresholds on catval ----
    catval = read_jsonl(args.catval)
    ctexts = [r["text"][:4000] for r in catval]
    cfull = predict_full(model, tok, ctexts, max_len, device, batch_size=128)
    n_cat = len(CATEGORIES)
    cat_true = []
    for r in catval:
        v = [0.0] * n_cat
        for c in r.get("cats") or []:
            v[int(c)] = 1.0
        cat_true.append(v)
    cat_probs = [[row[2 + i] for row in cfull] for i in range(n_cat)]

    categories = {}
    thresholds = {}
    for i, cat in enumerate(CATEGORIES):
        yv = [c[i] for c in cat_true]
        probs = cat_probs[i]
        if sum(yv) == 0:
            thresholds[cat] = 0.5
            categories[cat] = {"best_threshold": 0.5, "f1_at_best": 0.0, "ece": ece(probs, yv),
                                "prevalence": 0.0, "experimental": True,
                                "note": "no positives in catval; default 0.5"}
            continue
        best = (0.5, -1.0)
        for t in [x / 100 for x in range(20, 96)]:
            tp = sum(1 for p, l in zip(probs, yv) if p >= t and l == 1)
            fp = sum(1 for p, l in zip(probs, yv) if p >= t and l == 0)
            fn = sum(1 for p, l in zip(probs, yv) if p < t and l == 1)
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            if f1 > best[1]:
                best = (t, f1)
        thresholds[cat] = best[0]
        categories[cat] = {"best_threshold": best[0], "f1_at_best": best[1], "ece": ece(probs, yv),
                           "prevalence": sum(yv) / len(yv),
                           "experimental": best[1] < 0.3}  # weak cats are flagged experimental
        print(f"{cat:22s} thr={best[0]:.2f} f1={best[1]:.4f} prev={sum(yv)/len(yv):.3f}")

    result = {
        "checkpoint": args.checkpoint,
        "temperature": temperature,
        "val_n": len(vlabels),
        "binary_ece_raw": ece_raw,
        "binary_ece": ece_cal,
        "catval_n": len(catval),
        "categories": categories,
        "cat_thresholds": thresholds,
        "cat_experimental": [c for c, v in categories.items() if v.get("experimental")],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))

    # reliability diagram (raw vs calibrated)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        bins = np.linspace(0, 1, 11)
        fig, ax = plt.subplots(figsize=(5, 4.5))
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        for probs, name, color in ((p_raw, "raw", "tab:red"), (p_cal, f"T={temperature:.2f}", "tab:blue")):
            xs, ys, ws = [], [], []
            for i in range(10):
                m = (probs > bins[i]) & (probs <= bins[i + 1])
                if m.sum() == 0:
                    continue
                xs.append(probs[m].mean())
                ys.append(np.asarray(vlabels)[m].mean())
                ws.append(m.mean())
            ax.scatter(xs, ys, s=[w * 400 for w in ws], label=name, alpha=0.8)
        ax.set_xlabel("predicted P(unsafe)")
        ax.set_ylabel("observed frequency")
        ax.set_title(f"Censer.9 reliability on val (ECE {ece_raw:.3f} -> {ece_cal:.3f})")
        ax.legend()
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
