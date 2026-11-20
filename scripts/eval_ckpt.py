"""Evaluate one checkpoint: val/test metrics @0.5 + best threshold, probes, quick latency.

    PYTHONPATH=. .venv/bin/python scripts/eval_ckpt.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kuclab.dataset import read_jsonl
from kuclab.evaluate import best_threshold, load_checkpoint, metrics_from, predict_proba
from kuclab.probes import PROBES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tok", default=None)
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device, args.tok)
    max_len = mcfg.max_seq_len
    print(f"params={model.num_parameters():,} ckpt_mb={Path(args.checkpoint).stat().st_size/1e6:.2f} "
          f"layers={mcfg.n_layers} d={mcfg.d_model} vocab={mcfg.vocab_size} ctx={max_len}")

    out: dict = {"checkpoint": args.checkpoint, "params": model.num_parameters()}
    for split in ("val", "test"):
        rows = read_jsonl(Path(args.data) / f"{split}.jsonl")
        texts = [r["text"][:500] for r in rows]
        labels = [int(r["label"]) for r in rows]
        probs = predict_proba(model, tok, texts, max_len, device, batch_size=args.batch_size)
        m05 = metrics_from(labels, probs, 0.5)
        bt, bm = best_threshold(labels, probs)
        out[split] = {"at_0.5": m05, "best": {"threshold": bt, **bm}}
        print(f"{split}: @0.5 acc={m05['accuracy']:.4f} f1={m05['f1']:.4f} p={m05['precision']:.4f} "
              f"r={m05['recall']:.4f} | best@{bt:.2f} f1={bm['f1']:.4f}")

    probe_texts = [p[0] for p in PROBES]
    probe_labels = [p[1] for p in PROBES]
    probe_probs = predict_proba(model, tok, probe_texts, max_len, device, batch_size=len(probe_texts))
    pm = metrics_from(probe_labels, probe_probs, 0.5)
    out["probes"] = pm
    misses = [t for t, l, p in zip(probe_texts, probe_labels, probe_probs) if (p >= 0.5) != bool(l)]
    print(f"probes: {pm['accuracy']*100:.1f}% ({len(probe_texts)-len(misses)}/{len(probe_texts)})")
    for m in misses[:15]:
        print(f"  MISS: {m!r}")

    # quick latency / throughput
    sample = [r["text"][:500] for r in read_jsonl(Path(args.data) / "test.jsonl")[:200]]
    for t in sample[:20]:
        predict_proba(model, tok, [t], max_len, device, batch_size=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for t in sample:
        predict_proba(model, tok, [t], max_len, device, batch_size=1)
    torch.cuda.synchronize()
    lat = (time.perf_counter() - t0) / len(sample) * 1000
    bench = [r["text"][:500] for r in read_jsonl(Path(args.data) / "test.jsonl")[:1024]]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for start in range(0, len(bench), 8):
        predict_proba(model, tok, bench[start:start + 8], max_len, device, batch_size=8)
    torch.cuda.synchronize()
    thr = len(bench) / (time.perf_counter() - t0)
    out["latency_ms"] = lat
    out["throughput_b8"] = thr
    print(f"latency={lat:.2f} ms  throughput(b8)={thr:.0f} texts/s")
    print(json.dumps(out, indent=2)[:400])


if __name__ == "__main__":
    main()
