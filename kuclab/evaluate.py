from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from .model import remap_legacy_state
import torch.nn.functional as F

from .dataset import Batcher, SafetyDataset, collate
from .model import KucLabCneser, ModelConfig
from .tokenizer import KucTokenizer


def load_checkpoint(path: str | Path, device: torch.device, tok_dir: str | Path | None = None):
    state = torch.load(path, map_location=device, weights_only=False)
    state = remap_legacy_state(state)
    mcfg = ModelConfig(**state["model_config"])
    model = KucLabCneser(mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if tok_dir is None:
        tok_dir = Path(path).parent / "tokenizer"
        if not (Path(tok_dir) / "tokenizer.json").exists():
            tok_dir = Path("checkpoints/tokenizer")
    tok = KucTokenizer.load(tok_dir)
    return model, tok, mcfg, state


@torch.inference_mode()
def predict_full(model: KucLabCneser, tok: KucTokenizer, texts: list[str], max_len: int, device: torch.device, batch_size: int = 128, temperature: float = 1.0) -> list[list[float]]:
    """Full multi-label probabilities over all answer tokens [n_texts, n_answers]."""
    out: list[list[float]] = [[] for _ in texts]
    for start in range(0, len(texts), batch_size):
        idx = list(range(start, min(start + batch_size, len(texts))))
        ids = [tok.encode_prompt(texts[i], max_len) for i in idx]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        sorted_ids = [ids[i] for i in order]
        max_batch_len = len(sorted_ids[-1])
        input_ids = torch.full((len(sorted_ids), max_batch_len), tok.pad_id, dtype=torch.long)
        for i, seq in enumerate(sorted_ids):
            input_ids[i, : len(seq)] = torch.tensor(seq)
        raw = model.answer_logits(input_ids.to(device)) / max(temperature, 1e-6)
        if len(model.cfg.answer_ids) == 2:
            probs = raw.softmax(-1)
        else:
            probs = raw.sigmoid()
        probs = probs.float().cpu().tolist()
        for pos, i in enumerate(order):
            out[idx[i]] = probs[pos]
    return out


@torch.inference_mode()
def predict_proba(model: KucLabCneser, tok: KucTokenizer, texts: list[str], max_len: int, device: torch.device, batch_size: int = 128, temperature: float = 1.0) -> list[float]:
    """Binary unsafe probability per text (answer token "1"); batches sorted by length."""
    return [row[1] for row in predict_full(model, tok, texts, max_len, device, batch_size, temperature)]


@torch.inference_mode()
def predict_proba_chunked(model: KucLabCneser, tok: KucTokenizer, texts: list[str], max_len: int, device: torch.device,
                          batch_size: int = 128, chunk_tokens: int = 256, stride: int = 192, temperature: float = 1.0) -> list[float]:
    """Long-input inference: texts longer than `chunk_tokens` are split into
    overlapping token chunks, each chunk is classified, and the max unsafe prob
    is returned (a single unsafe passage anywhere flags the whole text).
    Effective context is unbounded; short texts take the single-pass path."""
    out: list[float] = []
    for text in texts:
        ids = tok.encode_prompt(text, 10 * max_len)  # tokenize whole text
        body = ids[2:-1]  # strip [BOS][USER] ... [ASSISTANT]
        if len(body) <= chunk_tokens:
            out.append(predict_proba(model, tok, [text], max_len, device, batch_size, temperature)[0])
            continue
        chunks = []
        start = 0
        while start < len(body):
            piece = body[start:start + chunk_tokens]
            chunks.append(piece)
            if start + chunk_tokens >= len(body):
                break
            start += stride
        # rebuild prompts per chunk
        chunk_texts = []
        for piece in chunks:
            chunk_texts.append(tok.decode(piece))
        probs = predict_proba(model, tok, chunk_texts, max_len, device, batch_size=len(chunk_texts), temperature=temperature)
        out.append(max(probs))
    return out


def metrics_from(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict:
    tp = fp = tn = fn = 0
    for label, prob in zip(labels, probs):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def best_threshold(labels: list[int], probs: list[float]) -> tuple[float, dict]:
    best = (0.5, metrics_from(labels, probs, 0.5))
    for i in range(5, 96):
        threshold = i / 100
        m = metrics_from(labels, probs, threshold)
        if m["f1"] > best[1]["f1"]:
            best = (threshold, m)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Censer checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--config", default="configs/censer9a.json")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)

    ds = SafetyDataset(Path(args.data) / f"{args.split}.jsonl", tok, config["max_seq_len"])
    texts = [row["text"] for row in ds.rows]
    labels = [int(row["label"]) for row in ds.rows]

    probs = predict_proba(model, tok, texts, config["max_seq_len"], device)
    m = metrics_from(labels, probs)
    threshold, m_best = best_threshold(labels, probs)

    print(f"checkpoint: {args.checkpoint}  split: {args.split}  n={len(texts)}")
    print(f"0.50 threshold : acc={m['accuracy']:.4f} f1={m['f1']:.4f} p={m['precision']:.4f} r={m['recall']:.4f}")
    print(f"best threshold : {threshold:.2f} acc={m_best['accuracy']:.4f} f1={m_best['f1']:.4f} p={m_best['precision']:.4f} r={m_best['recall']:.4f}")
    print(f"confusion @0.5: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")

    by_source: dict[str, dict] = defaultdict(lambda: {"labels": [], "probs": []})
    for row, prob in zip(ds.rows, probs):
        by_source[row["source"]]["labels"].append(int(row["label"]))
        by_source[row["source"]]["probs"].append(prob)

    print("\nper source (threshold 0.50):")
    source_report = {}
    for source, data in sorted(by_source.items(), key=lambda kv: -len(kv[1]["labels"])):
        sm = metrics_from(data["labels"], data["probs"], 0.5)
        source_report[source] = sm
        print(f"  {source:18s} n={sm['total']:6d} acc={sm['accuracy']:.3f} f1={sm['f1']:.3f} p={sm['precision']:.3f} r={sm['recall']:.3f}")

    errors = [
        {"text": t, "label": l, "prob": p}
        for t, l, p in zip(texts, labels, probs)
        if (p >= 0.5) != (l == 1)
    ]
    errors.sort(key=lambda e: -abs(e["prob"] - 0.5))
    print(f"\nfalse positives: {sum(1 for e in errors if e['label'] == 0)}")
    for e in [e for e in errors if e["label"] == 0][:10]:
        print(f"  p={e['prob']:.3f} {e['text'][:100]!r}")
    print(f"false negatives: {sum(1 for e in errors if e['label'] == 1)}")
    for e in [e for e in errors if e["label"] == 1][:10]:
        print(f"  p={e['prob']:.3f} {e['text'][:100]!r}")

    from .probes import PROBES

    probe_texts = [t for t, _ in PROBES]
    probe_labels = [l for _, l in PROBES]
    probe_probs = predict_proba(model, tok, probe_texts, config["max_seq_len"], device)
    pm = metrics_from(probe_labels, probe_probs, 0.5)
    print(f"\nprobe set (hand-written, {len(PROBES)} cases): acc={pm['accuracy']:.3f} f1={pm['f1']:.3f}")
    misses = [(t, l, p) for t, l, p in zip(probe_texts, probe_labels, probe_probs) if int(p >= 0.5) != l]
    for t, l, p in misses:
        print(f"  MISS exp={l} p={p:.3f} {t!r}")
    if not misses:
        print("  all probes correct")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "at_0.5": m,
                    "best": {"threshold": threshold, **m_best},
                    "per_source": source_report,
                    "model_metrics": state.get("metrics", {}),
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
