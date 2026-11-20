"""Long-context evaluation: build long documents from held-out test rows and
compare (a) 192-token truncation (old behaviour), (b) 4096-token native
context with linear RoPE scaling, (c) chunked inference beyond the context.

Label of a long doc = OR of its row labels. Writes reports/long_context.json.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from kuclab.dataset import read_jsonl
from kuclab.evaluate import load_checkpoint, predict_proba, predict_proba_chunked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--docs", type=int, default=300)
    parser.add_argument("--rows-per-doc", type=int, default=25)
    parser.add_argument("--out", default="reports/long_context.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    ctx = mcfg.max_seq_len

    rows = read_jsonl("data/processed/test.jsonl")
    rng = random.Random(7)
    docs, labels = [], []
    for _ in range(args.docs):
        pick = rng.sample(rows, args.rows_per_doc)
        # put an unsafe row at a random position in half the docs
        text = "\n\n".join(r["text"] for r in pick)
        docs.append(text)
        labels.append(1 if any(int(r["label"]) == 1 for r in pick) else 0)

    def f1(probs):
        tp = fp = tn = fn = 0
        for l, p in zip(labels, probs):
            pr = 1 if p >= 0.5 else 0
            if pr == 1 and l == 1: tp += 1
            elif pr == 1: fp += 1
            elif l == 0: tn += 1
            else: fn += 1
        P = tp / (tp + fp) if tp + fp else 0
        R = tp / (tp + fn) if tp + fn else 0
        return (tp + tn) / len(labels), 2 * P * R / (P + R) if P + R else 0

    avg_chars = sum(len(d) for d in docs) / len(docs)
    avg_tokens = sum(len(tok.encode_prompt(d, 100000)) for d in docs) / len(docs)
    print(f"docs={len(docs)} avg_chars={avg_chars:.0f} avg_tokens={avg_tokens:.0f} ctx={ctx}")

    res = {}
    # (a) old behaviour: truncate at 192 tokens
    p192 = predict_proba(model, tok, docs, 192, device, batch_size=16)
    res["truncate_192_tokens"] = {"acc": f1(p192)[0], "f1": f1(p192)[1]}
    # (b) native full context
    pfull = predict_proba(model, tok, docs, ctx, device, batch_size=16)
    res[f"native_{ctx}_tokens"] = {"acc": f1(pfull)[0], "f1": f1(pfull)[1]}
    # (c) chunked beyond context
    pchunk = predict_proba_chunked(model, tok, docs, ctx, device, batch_size=16)
    res["chunked_256_192_max"] = {"acc": f1(pchunk)[0], "f1": f1(pchunk)[1]}

    for k, v in res.items():
        print(f"{k:24s} acc={v['acc']:.4f} f1={v['f1']:.4f}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "ctx": ctx, "n_docs": len(docs),
                               "avg_chars": avg_chars, "avg_tokens": avg_tokens, "results": res}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
