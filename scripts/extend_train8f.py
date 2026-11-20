#!/usr/bin/env python3
"""Balance the category labels: duplicate rare-category unsafe rows (x8) as
must_catch so the category head sees a balanced distribution. Append-only;
extends train_soft8.npy with hard 1.0 labels."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RNG = random.Random(8802)

RARE = {0, 1, 2, 4, 5, 6, 7, 8}  # all except hate(3) and other(9) get boosted; violence(0) kept too
PER_CAT = 200
TIMES = 8

train_path = DATA / "train.jsonl"
soft_path = DATA / "train_soft8.npy"

rows = [json.loads(l) for l in open(train_path, encoding="utf-8")]
existing = {r["text"] for r in rows}

by_cat = {}
for r in rows:
    if r["label"] != 1:
        continue
    for c in r.get("cats", []):
        if int(c) in RARE:
            by_cat.setdefault(int(c), []).append(r)

soft = np.load(soft_path)
assert len(soft) == len(rows)

added = 0
with open(train_path, "a", encoding="utf-8") as f:
    for cat, items in sorted(by_cat.items()):
        RNG.shuffle(items)
        for r in items[:PER_CAT]:
            for k in range(TIMES):
                # slight surface variation to avoid pure memorization
                text = r["text"]
                if k > 0:
                    text = RNG.choice(["hey, ", "please: ", "quick question: ", "", "hi, "]) + text
                if text in existing:
                    continue
                existing.add(text)
                nr = {"text": text[:4000], "label": 1, "source": "must_catch", "cats": r.get("cats", [])}
                f.write(json.dumps(nr, ensure_ascii=False) + "\n")
                added += 1

soft_new = np.concatenate([soft, np.ones(added, dtype=soft.dtype)])
np.save(soft_path, soft_new)
print(f"appended {added} cat-boost rows; train={len(soft_new)} soft aligned OK")
