"""Extend the .6 train split with multilingual + category-labelled rows (train-only).

- enriches existing aegis/jigsaw/beavertails train rows with category vectors
- appends multilingual scraped rows (de hate speech, multilingual hate speech) as
  train-only sources
- appends synthetic cs/de/es/fr templates (safe + unsafe)
- appends adversarial (leet/homoglyph/zero-width) variants of curated unsafe rows

val/test files are not touched.

    .venv/bin/python scripts/extend_train_ml.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kuclab.categories import AEGIS_MAP, BEAVERTAILS_MAP, JIGSAW_MAP, categorize_text
from kuclab.data import EXTRA_HANDWRITTEN_UNSAFE, HANDWRITTEN_UNSAFE, fingerprint
from kuclab.dataset import read_jsonl
from kuclab.multilingual import adversarial_variants, generate_ml
from kuclab.prepare_data import split_for


def _cats_for_aegis(row) -> list[int]:
    labels = [row[f"labels_{i}"] for i in range(row.get("num_annotations") or 5)]
    cats = set()
    for l in labels:
        if l in AEGIS_MAP:
            cats.add(AEGIS_MAP[l])
    return sorted(cats)


def _cats_for_jigsaw(row) -> list[int]:
    cats = set()
    for k, v in JIGSAW_MAP.items():
        if row.get(k):
            cats.add(v)
    return sorted(cats)


def _cats_for_beavertails(row) -> list[int]:
    cats = set()
    for c in (row.get("categories") or []):
        key = str(c).lower().split(":")[0].split("/")[0].strip()
        if key in BEAVERTAILS_MAP:
            cats.add(BEAVERTAILS_MAP[key])
        else:
            for k, v in BEAVERTAILS_MAP.items():
                if k in key:
                    cats.add(v)
                    break
    return sorted(cats)


def build_cat_lookup() -> dict[str, list[int]]:
    """fingerprint -> cats for category-labelled sources (full load, bucket-independent)."""
    lookup: dict[str, list[int]] = {}
    from datasets import load_dataset

    ds = load_dataset("nvidia/Aegis-AI-Content-Safety-Dataset-1.0", split="train")
    for row in ds:
        text = (row["text"] or "").strip()
        if len(text) < 3:
            continue
        cats = _cats_for_aegis(row)
        if cats:
            lookup[fingerprint(text)] = cats
    print(f"aegis cat lookup: {len(lookup)}")

    ds = load_dataset("thesofakillers/jigsaw-toxic-comment-classification-challenge", split="train")
    n = 0
    for row in ds:
        text = (row["comment_text"] or "").strip()
        if len(text) < 3:
            continue
        cats = _cats_for_jigsaw(row)
        if cats:
            lookup.setdefault(fingerprint(text), cats)
            n += 1
    print(f"jigsaw cat rows: {n}")

    try:
        ds = load_dataset("PKU-Alignment/BeaverTails", "default", split="30k_train")
        n = 0
        for row in ds:
            text = (row["prompt"] or "").strip()
            if len(text) < 3 or not row["is_safe"]:
                continue
            cats = _cats_for_beavertails(row)
            if cats:
                lookup.setdefault(fingerprint(text), cats)
                n += 1
        print(f"beavertails cat rows: {n}")
    except Exception as exc:
        print("beavertails skipped:", exc)
    return lookup


def load_ml_scraped() -> list[dict]:
    """Multilingual scraped rows, train-only (bucket check keeps val/test identical)."""
    from datasets import load_dataset

    out: list[dict] = []
    specs = [
        ("manueltonneau/german-hate-speech-superset", "de_hate", 3),
        ("vidhur2k/multilingual-hate-speech", "ml_hate", 3),
        ("FrancophonIA/multilingual-hatespeech-dataset", "fr_hate", 3),
    ]
    for repo, src, cat in specs:
        try:
            ds = load_dataset(repo, split="train")
        except Exception:
            try:
                ds = load_dataset(repo)
                ds = ds[list(ds.keys())[0]]
            except Exception as exc:
                print(f"  {repo} skipped: {exc}")
                continue
        cols = ds.column_names
        text_col = next((c for c in ("text", "comment", "tweet", "post", "content", "sentence") if c in cols), cols[0])
        label_col = next((c for c in ("label", "labels", "hate", "is_hate", "toxic", "class") if c in cols), None)
        n = 0
        for row in ds:
            text = str(row.get(text_col) or "").strip()
            if len(text) < 3 or len(text) > 500:
                continue
            if label_col is not None:
                v = row.get(label_col)
                s = str(v).lower()
                is_hate = s in ("1", "hate", "hate_speech", "toxic", "offensive", "yes", "true") or (isinstance(v, (int, float)) and float(v) >= 1)
                if not is_hate:
                    continue
            if split_for(fingerprint(text), 0.08, 0.08) != "train":
                continue  # keep val/test byte-identical
            out.append({"text": text, "label": 1, "source": f"ml_scraped_{src}", "cats": [cat]})
            n += 1
            if n >= 20000:
                break
        print(f"  {repo}: {n} rows")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--ml-per-lang", type=int, default=150)
    parser.add_argument("--seed", type=int, default=4242)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_path = data_dir / "train.jsonl"
    rows = read_jsonl(train_path)
    seen = {fingerprint(r["text"]) for r in rows}
    print(f"train rows: {len(rows)}")

    # 1) enrich existing category-labelled train rows
    lookup = build_cat_lookup()
    enriched = 0
    for r in rows:
        if r.get("source") in ("aegis", "jigsaw", "beavertails") and "cats" not in r:
            cats = lookup.get(fingerprint(r["text"]))
            if cats:
                r["cats"] = cats
                enriched += 1
    print(f"enriched {enriched} existing rows with cats")

    # 2) multilingual scraped (train-only via bucket check)
    for r in load_ml_scraped():
        fp = fingerprint(r["text"])
        if fp in seen:
            continue
        seen.add(fp)
        rows.append(r)

    # 3) synthetic multilingual templates
    n_ml = 0
    for text, label, source in generate_ml(args.ml_per_lang, args.seed):
        fp = fingerprint(text)
        if fp in seen or len(text) < 3:
            continue
        seen.add(fp)
        cats = categorize_text(text) if label == 1 else []
        rows.append({"text": text, "label": label, "source": source, **({"cats": cats} if cats else {})})
        n_ml += 1
    print(f"added {n_ml} synthetic multilingual rows")

    # 4) adversarial variants of curated unsafe rows
    n_adv = 0
    base = HARDWRITTEN_UNSAFE_CHECK = HANDWRITTEN_UNSAFE + EXTRA_HANDWRITTEN_UNSAFE
    for i, t in enumerate(base):
        for v in adversarial_variants(t, i, n=2):
            fp = fingerprint(v)
            if fp in seen or len(v) < 3:
                continue
            seen.add(fp)
            rows.append({"text": v, "label": 1, "source": "adversarial", "cats": categorize_text(t)})
            n_adv += 1
    print(f"added {n_adv} adversarial rows")

    rows.sort(key=lambda r: r["text"])
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows")
    for f in ("train_soft.npy", "train_soft5.npy", "train_soft6.npy"):
        p = data_dir / f
        if p.exists():
            p.unlink()
            print("removed stale", f)


if __name__ == "__main__":
    main()
