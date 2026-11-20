"""External benchmarks (never seen in training): ToxiGen + HateXplain.

Maps both to binary unsafe and reports accuracy/F1/precision/recall at 0.5 plus
best-threshold F1, on the same hardware as the internal comparison.

    .venv/bin/python scripts/eval_external.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from kuclab.evaluate import load_checkpoint, metrics_from, predict_proba


def load_toxigen() -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    ds = load_dataset("Intuit-GenSRF/toxigen-test-annotated", split="train")  # only split available
    texts, labels = [], []
    for row in ds:
        t = str(row.get("text") or "").strip()
        if len(t) < 3:
            continue
        lab = row.get("labels", row.get("label"))
        # ToxiGen annotated: labels is a list of category strings ('toxic' = unsafe,
        # empty/'benign' = safe); older revisions use a numeric label (0 = benign, 1..4 = intensity)
        if isinstance(lab, (list, tuple)):
            y = 1 if any(str(l).strip().lower() == "toxic" for l in lab) else 0
        else:
            try:
                y = 1 if float(lab) > 0 else 0
            except Exception:
                continue
        texts.append(t[:4000])
        labels.append(y)
    return texts, labels


def load_hatexplain() -> tuple[list[str], list[int]]:
    from datasets import load_dataset

    # script-based dataset: use the auto-converted parquet revision
    ds = load_dataset("Hate-speech-CNERG/hatexplain", revision="refs/convert/parquet", split="test")
    texts, labels = [], []
    for row in ds:
        t = " ".join(str(p) for p in row["post_tokens"]).strip()
        if len(t) < 3:
            continue
        # labels: 0 = hatespeech, 1 = offensive, 2 = normal (majority vote list)
        votes = row["annotators"]["label"]
        from collections import Counter

        maj = Counter(votes).most_common(1)[0][0]
        texts.append(t[:4000])
        labels.append(1 if maj in (0, 1) else 0)
    return texts, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="reports/external_eval.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    results = {}
    for name, loader in [("toxigen_test", load_toxigen), ("hatexplain_test", load_hatexplain)]:
        try:
            texts, labels = loader()
        except Exception as exc:
            print(f"{name}: skipped ({exc})")
            continue
        probs = predict_proba(model, tok, texts, max_len, device, batch_size=64)
        m05 = metrics_from(labels, probs, 0.5)
        best = (0.5, m05)
        for i in range(5, 96):
            th = i / 100
            m = metrics_from(labels, probs, th)
            if m["f1"] > best[1]["f1"]:
                best = (th, m)
        results[name] = {"n": len(labels), "unsafe": sum(labels), "at_0.5": m05,
                         "best": {"threshold": best[0], **best[1]}}
        print(f"{name}: n={len(labels)} unsafe={sum(labels)}  @0.5 f1={m05['f1']:.4f} acc={m05['accuracy']:.4f} "
              f"| best@{best[0]:.2f} f1={best[1]['f1']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "device": str(device), "results": results}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
