"""Duplicate scraped TRAIN rows with text augmentations, as train-only rows.

Each augmented copy gets source="aug_scraped", which prepare_data.py routes to
the train split only — so the val/test splits stay byte-identical to the
previous versions while the effective training set roughly doubles.

    .venv/bin/python scripts/augment_train.py [--copies 1]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from kuclab.data import fingerprint
from kuclab.dataset import read_jsonl

SCRAPED = {"aegis", "jigsaw", "civil_comments", "beavertails", "toxic_chat", "dolly"}

_RISK = ("kill", "bomb", "drug", "shoot", "hack", "attack", "weapon", "poison", "gun", "rape", "murder", "gun", "meth", "fentanyl", "virus", "malware", "suicide")


def _augment(text: str, rng: random.Random) -> str:
    op = rng.random()
    if op < 0.35 and len(text) >= 5:
        # one random character edit (same as training-time typo aug)
        i = rng.randrange(len(text))
        r = rng.random()
        if r < 0.4:
            text = text[:i] + text[i + 1 :]
        elif r < 0.8:
            text = text[:i] + rng.choice("abcdefghijklmnopqrstuvwxyz") + text[i + 1 :]
        else:
            text = text[: i + 1] + text[i] + text[i + 1 :]
        return text
    if op < 0.55:
        # random case flip on a few words
        words = text.split(" ")
        for _ in range(rng.randint(1, 3)):
            j = rng.randrange(len(words))
            words[j] = words[j].upper() if rng.random() < 0.5 else words[j].lower()
        return " ".join(words)
    if op < 0.75:
        # drop one non-risk word (keeps the label-relevant words)
        words = text.split(" ")
        if len(words) > 6:
            candidates = [j for j, w in enumerate(words) if not any(k in w.lower() for k in _RISK)]
            if candidates:
                j = rng.choice(candidates)
                del words[j]
        return " ".join(words)
    # punctuation / whitespace jitter
    text = text.replace("  ", " ")
    if rng.random() < 0.5:
        text = text.rstrip(".!?") 
    if rng.random() < 0.3:
        text += rng.choice([".", "!", "?", "..."])
    return text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--copies", type=int, default=1, help="augmented copies per scraped row")
    parser.add_argument("--seed", type=int, default=777)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_path = data_dir / "train.jsonl"
    rows = read_jsonl(train_path)
    seen = {fingerprint(r["text"]) for r in rows}
    n0 = len(rows)
    scraped = [i for i, r in enumerate(rows) if r.get("source") in SCRAPED]
    print(f"train rows: {n0}, scraped rows: {len(scraped)}, copies: {args.copies}")

    added = 0
    for i in scraped:
        row = rows[i]
        for c in range(args.copies):
            rng = random.Random(f"aug:{args.seed}:{i}:{c}")
            text = _augment(row["text"], rng)
            if len(text) < 3:
                continue
            fp = fingerprint(text)
            if fp in seen:
                continue
            seen.add(fp)
            rows.append({"text": text, "label": row["label"], "source": "aug_scraped"})
            added += 1
    print(f"added {added} augmented rows")
    rows.sort(key=lambda r: r["text"])
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {train_path}")

    soft = data_dir / "train_soft.npy"
    if soft.exists():
        soft.unlink()
        print(f"removed stale {soft} (re-run kuclab.distill)")


if __name__ == "__main__":
    main()
