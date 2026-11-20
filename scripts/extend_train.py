"""Append extra TRAIN-ONLY rows to data/processed/train.jsonl.

The val/test splits are content-hash based and must stay byte-identical to the
previous versions, so this script only appends rows whose source is curated
(hard_negative / borderline / curated_unsafe / synthetic_safe / synthetic_unsafe)
— prepare_data.py always routes those sources to the train split.

    .venv/bin/python scripts/extend_train.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuclab.data import (
    BORDERLINE,
    EXTRA_BORDERLINE,
    EXTRA_HARD_NEGATIVES,
    EXTRA_HANDWRITTEN_UNSAFE,
    HANDWRITTEN_UNSAFE,
    HARD_NEGATIVES,
    fingerprint,
)
from kuclab.dataset import read_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--synthetic-unsafe", type=int, default=8000)
    parser.add_argument("--synthetic-safe", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_path = data_dir / "train.jsonl"
    rows = read_jsonl(train_path)
    seen = {fingerprint(r["text"]) for r in rows}
    print(f"existing train rows: {len(rows)}")

    new_rows: list[dict] = []

    def add(text: str, label: int, source: str) -> None:
        fp = fingerprint(text)
        if fp in seen or len(text) < 3:
            return
        seen.add(fp)
        new_rows.append({"text": text, "label": label, "source": source})

    from kuclab import synth_unsafe, synthetic

    for t in synth_unsafe.generate(args.synthetic_unsafe, args.seed):
        add(t, 1, "synthetic_unsafe")
    for t in synthetic.generate(args.synthetic_safe, args.seed + 1):
        add(t, 0, "synthetic_safe")
    for t in HARD_NEGATIVES + EXTRA_HARD_NEGATIVES:
        add(t, 0, "hard_negative")
    for t in BORDERLINE + EXTRA_BORDERLINE:
        add(t, 0, "borderline")
    for t in HANDWRITTEN_UNSAFE + EXTRA_HANDWRITTEN_UNSAFE:
        add(t, 1, "curated_unsafe")

    print(f"new rows: {len(new_rows)} (unsafe={sum(r['label'] for r in new_rows)})")
    rows.extend(new_rows)
    rows.sort(key=lambda r: r["text"])
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {train_path}")

    # invalidate stale soft labels: they must be recomputed for the new row order
    soft = data_dir / "train_soft.npy"
    if soft.exists():
        soft.unlink()
        print(f"removed stale {soft} (re-run kuclab.distill)")


if __name__ == "__main__":
    main()
