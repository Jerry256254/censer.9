from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from .data import (
    BORDERLINE,
    HANDWRITTEN_UNSAFE,
    HARD_NEGATIVES,
    SAFE,
    UNSAFE,
    Example,
    clean_text,
    fingerprint,
    has_risk_keyword,
)

AEGIS_SAFE = {"Safe"}
AEGIS_SKIP = {"Needs Caution", None, ""}


def _balanced(yes_ratio: float) -> float:
    return yes_ratio


def load_aegis(max_per_source: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    out: list[Example] = []
    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split="train")
    for row in ds:
        labels = [row[f"labels_{i}"] for i in range(row.get("num_annotations") or 5)]
        labels = [l for l in labels if l not in AEGIS_SKIP]
        if not labels:
            continue
        safe_votes = sum(1 for l in labels if l in AEGIS_SAFE)
        label = SAFE if safe_votes * 2 > len(labels) else UNSAFE
        out.append(Example(text=row["text"], label=label, source="aegis"))
    rng.shuffle(out)
    return out[:max_per_source]


def load_jigsaw(max_per_source: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    out: list[Example] = []
    ds = load_dataset(
        "thesofakillers/jigsaw-toxic-comment-classification-challenge", split="train"
    )
    for row in ds:
        text = row["comment_text"] or ""
        label = UNSAFE if (row["toxic"] or row["severe_toxic"] or row["threat"] or row["identity_hate"]) else SAFE
        out.append(Example(text=text, label=label, source="jigsaw"))
    rng.shuffle(out)
    return out[:max_per_source]


def load_civil_comments(max_per_source: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    out: list[Example] = []
    try:
        ds = load_dataset("google/civil_comments", split="train")
    except Exception as exc:
        print(f"  civil_comments skipped: {exc}")
        return out
    for row in ds:
        score = float(row.get("toxicity") or 0.0)
        if score >= 0.5:
            label = UNSAFE
        elif score <= 0.1:
            label = SAFE
        else:
            continue
        out.append(Example(text=row["text"] or "", label=label, source="civil_comments"))
    rng.shuffle(out)
    return out[:max_per_source]


def load_beavertails(max_per_source: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    out: list[Example] = []
    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", "default", split="30k_train")
    except Exception as exc:
        print(f"  beavertails skipped: {exc}")
        return out
    for row in ds:
        out.append(
            Example(
                text=row["prompt"],
                label=SAFE if row["is_safe"] else UNSAFE,
                source="beavertails",
            )
        )
    rng.shuffle(out)
    return out[:max_per_source]


def load_toxic_chat(max_per_source: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    out: list[Example] = []
    try:
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124", split="train")
    except Exception as exc:
        print(f"  toxic_chat skipped: {exc}")
        return out
    for row in ds:
        if not row.get("human_annotation"):
            continue
        out.append(
            Example(
                text=row["user_input"] or "",
                label=UNSAFE if row.get("toxicity") else SAFE,
                source="toxic_chat",
            )
        )
    rng.shuffle(out)
    return out[:max_per_source]


def load_safe_instructions(max_per_source: int, rng: random.Random) -> list[Example]:
    from datasets import load_dataset

    out: list[Example] = []
    try:
        ds = load_dataset("databricks/databricks-dolly-15k", split="train")
        for row in ds:
            out.append(Example(text=row["instruction"], label=SAFE, source="dolly"))
    except Exception as exc:
        print(f"  dolly skipped: {exc}")
    rng.shuffle(out)
    return out[:max_per_source]


def load_handwritten() -> list[Example]:
    negatives = [Example(text=t, label=SAFE, source="hard_negative") for t in HARD_NEGATIVES]
    unsafe = [Example(text=t, label=UNSAFE, source="curated_unsafe") for t in HANDWRITTEN_UNSAFE]
    borderline = [Example(text=t, label=SAFE, source="borderline") for t in BORDERLINE]
    return negatives + unsafe + borderline


def load_synthetic_safe(count: int, seed: int) -> list[Example]:
    from .synthetic import generate

    return [Example(text=t, label=SAFE, source="synthetic_safe") for t in generate(count, seed)]


CURATED_SOURCES = {"hard_negative", "borderline", "curated_unsafe", "synthetic_safe", "synthetic_unsafe", "aug_scraped"}


def balance(examples: list[Example], rng: random.Random, safe_ratio: float, curated_repeat: int) -> list[Example]:
    """Keep every unsafe and every curated example; fill the rest with safe data.

    Curated examples are repeated so their signal is not drowned out by the
    hundreds of thousands of scraped rows.
    """
    unsafe = [e for e in examples if e.label == UNSAFE and e.source not in CURATED_SOURCES]
    curated = [e for e in examples if e.source in CURATED_SOURCES]
    pooled = [e for e in examples if e.label == SAFE and e.source not in CURATED_SOURCES]
    if not unsafe:
        return examples

    curated_repeated = curated * curated_repeat
    target = min(len(pooled), int((len(unsafe) + len(curated_repeated)) * safe_ratio))
    priority = [e for e in pooled if has_risk_keyword(e.text)]
    rest = [e for e in pooled if not has_risk_keyword(e.text)]
    rng.shuffle(priority)
    rng.shuffle(rest)
    chosen = (priority + rest)[:target]
    return unsafe + curated_repeated + chosen


def split_for(fp: str, val_fraction: float, test_fraction: float) -> str:
    bucket = int(hashlib.sha256(fp.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if bucket < test_fraction:
        return "test"
    if bucket < test_fraction + val_fraction:
        return "val"
    return "train"


def build(config: dict, out_dir: Path) -> dict:
    rng = random.Random(config["seed"])
    max_chars = config["max_chars"]
    dcfg = config["data"]
    cap = dcfg["max_per_source"]

    loaders = [
        ("aegis", load_aegis),
        ("jigsaw", load_jigsaw),
        ("civil_comments", load_civil_comments),
        ("beavertails", load_beavertails),
        ("toxic_chat", load_toxic_chat),
        ("dolly", load_safe_instructions),
    ]

    raw: list[Example] = []
    for name, fn in loaders:
        print(f"loading {name} ...")
        got = fn(cap, rng)
        print(f"  {name}: {len(got)} examples")
        raw.extend(got)
    handwritten = load_handwritten()
    print(f"  handwritten: {len(handwritten)} examples")
    raw.extend(handwritten)

    synthetic = load_synthetic_safe(dcfg.get("synthetic_safe", 0), config["seed"])
    print(f"  synthetic_safe: {len(synthetic)} examples")
    raw.extend(synthetic)

    seen: set[str] = set()
    cleaned: list[Example] = []
    for ex in raw:
        text = clean_text(ex.text, max_chars)
        if len(text) < 3:
            continue
        fp = fingerprint(text)
        if fp in seen:
            continue
        seen.add(fp)
        cleaned.append(Example(text=text, label=ex.label, source=ex.source))

    splits: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    raw_splits: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for ex in cleaned:
        if ex.source in CURATED_SOURCES:
            raw_splits["train"].append(ex)
            continue
        bucket = split_for(fingerprint(ex.text), dcfg["val_fraction"], dcfg["test_fraction"])
        raw_splits[bucket].append(ex)

    safe_ratio = dcfg["safe_ratio"]
    for name, rows in raw_splits.items():
        balanced = balance(rows, rng, safe_ratio, dcfg.get('curated_repeat', 1))
        rng.shuffle(balanced)
        splits[name] = [{"text": e.text, "label": e.label, "source": e.source} for e in balanced]
        print(f"{name}: {len(rows)} -> {len(balanced)} after balancing")

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"max_chars": max_chars, "config": config}
    counts = {}
    for name, rows in splits.items():
        rows.sort(key=lambda r: r["text"])
        path = out_dir / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        label_counts = Counter(r["label"] for r in rows)
        counts[name] = {
            "total": len(rows),
            "safe": label_counts.get(SAFE, 0),
            "unsafe": label_counts.get(UNSAFE, 0),
            "sources": dict(Counter(r["source"] for r in rows)),
        }
        print(f"{name}: {counts[name]['total']} rows (safe={counts[name]['safe']}, unsafe={counts[name]['unsafe']})")

    (out_dir / "meta.json").write_text(json.dumps({**meta, "counts": counts}, indent=2))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the Censer .1 dataset.")
    parser.add_argument("--config", default="configs/censer9a.json")
    parser.add_argument("--out", default="data/processed")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    build(config, Path(args.out))


if __name__ == "__main__":
    main()
