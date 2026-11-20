"""Extend Censer.7 training data with external, distribution-shifted sources:

- HateXplain train split (parquet revision; the *test* split stays pure external eval)
- Jigsaw multilingual train (Intuit-GenSRF/jigsaw-multilingual-train-unique)
- catval.jsonl: HateXplain *val* split -> held-out set with category labels for calibration

val/test splits of the internal data stay byte-identical (append-only to train).

    .venv/bin/python scripts/extend_train7.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kuclab.categories import categorize_text

DATA = Path(__file__).resolve().parent.parent / "data" / "processed"

# jigsaw label -> our category index
JIGSAW_CAT = {
    "toxic": 3,  # hate_harassment
    "severe_toxic": 3,
    "obscene": 4,  # sexual_abuse-ish -> hate? keep hate_harassment; obscene -> sexual_abuse
    "threat": 0,  # violence
    "insult": 3,
    "identity_hate": 3,
}
JIGSAW_CAT["obscene"] = 4


def load_hatexplain(split: str):
    from datasets import load_dataset

    ds = load_dataset("Hate-speech-CNERG/hatexplain", revision="refs/convert/parquet", split=split)
    rows = []
    for r in ds:
        text = " ".join(str(p) for p in r["post_tokens"]).strip()
        if len(text) < 3:
            continue
        votes = r["annotators"]["label"]
        maj = Counter(votes).most_common(1)[0][0]
        label = 1 if maj in (0, 1) else 0  # 0 hatespeech, 1 offensive -> unsafe; 2 normal
        cats = [3] if label == 1 else []
        rows.append({"text": text, "label": label, "source": f"hatexplain_{split}", **({"cats": cats} if cats else {})})
    return rows


def load_jigsaw_ml():
    from datasets import load_dataset

    ds = load_dataset("Intuit-GenSRF/jigsaw-multilingual-train-unique", split="train")
    rows = []
    for r in ds:
        text = str(r["text"]).strip()
        if len(text) < 3:
            continue
        labels = [str(l) for l in (r["labels"] or [])]
        label = 1 if labels else 0
        cats = sorted({JIGSAW_CAT.get(l) for l in labels} - {None})
        if label == 1 and not cats:
            cats = categorize_text(text)
        rows.append({"text": text, "label": label, "source": "jigsaw_ml", **({"cats": cats} if cats else {})})
    return rows


def main() -> None:
    train_path = DATA / "train.jsonl"
    rows = [json.loads(l) for l in train_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    n0 = len(rows)
    print(f"existing train rows: {n0}")

    hx_train = load_hatexplain("train")
    hx_val = load_hatexplain("validation")
    jml = load_jigsaw_ml()
    print(f"hatexplain train={len(hx_train)} val={len(hx_val)} jigsaw_ml={len(jml)}")

    rows.extend(hx_train)
    rows.extend(jml)
    train_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"train.jsonl now {len(rows)} rows (+{len(rows)-n0})")

    catval = DATA / "catval.jsonl"
    catval.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in hx_val), encoding="utf-8")
    print(f"catval.jsonl: {len(hx_val)} rows (hatexplain val, held out)")

    # sanity: val/test untouched
    import hashlib

    for name, want in (("val.jsonl", "603ecc414d522b95dce9905ec293a037"), ("test.jsonl", "47dc390771bbed5ef46b1243bfce4516")):
        got = hashlib.md5((DATA / name).read_bytes()).hexdigest()
        print(name, "md5", got, "OK" if got == want else "MISMATCH!")


if __name__ == "__main__":
    main()
