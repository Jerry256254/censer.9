"""Benchmark Censer.9 against other public safety classifiers on the same test split.

Every model is evaluated on data/processed/test.jsonl (6,496 rows, label 1 = unsafe):
accuracy / precision / recall / F1 at threshold 0.5, single-text latency,
batched throughput, peak VRAM, parameter count and checkpoint size.

Writes reports/comparison.json. Run from the repo root:

    .venv/bin/python scripts/compare_models.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "data" / "processed" / "test.jsonl"
OUT = REPO_ROOT / "reports" / "comparison.json"

# name -> (repo id, unsafe-prob function over logits/probs)
HF_MODELS = [
    "martin-ha/toxic-comment-model",
    "unitary/toxic-bert",
    "unitary/unbiased-toxic-roberta",
    "Hate-speech-CNERG/dehatebert-mono-english",
    "Hate-speech-CNERG/bert-base-uncased-hatexplain",
]

# classes that count as "unsafe" for multi-class/multi-label models
TOXIC_CLASS_NAMES = {
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
    "toxicity", "severe_toxicity", "identity_attack", "sexual_explicit",
    "hate", "hate speech", "offensive",
}

KEYWORDS = [
    "bomb", "molotov", "fentanyl", "heroin", "cocaine", "meth ", "overdose",
    "rape", "rapist", "murder", "kill you", "kill yourself", "suck my dick",
    "terrorist", "terrorism", "shoot up", "school shooting", "build a bomb",
    "make a bomb", "poison", "anthrax", "child porn", "cp ", "hitman",
]


def load_test_rows():
    rows = [json.loads(line) for line in DATA.read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [r["text"][:500] for r in rows]
    labels = [int(r["label"]) for r in rows]
    return texts, labels


def metrics(labels, probs, threshold=0.5):
    tp = fp = tn = fn = 0
    for label, prob in zip(labels, probs):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1:
            fp += 1
        elif label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = (tp + tn) / (tp + fp + tn + fn)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn, "total": len(labels)}


def unsafe_indices(id2label):
    """Indices of the logit dimensions that mean 'unsafe'."""
    return [i for i, name in id2label.items() if str(name).lower() in TOXIC_CLASS_NAMES]


def bench_hf_model(repo, texts, device, batch_size=64):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    unsafe_idx = unsafe_indices(id2label)
    # single-class "toxic" style models: use the positive class directly
    if len(unsafe_idx) == 1 and len(id2label) == 2:
        unsafe_idx = [i for i, n in id2label.items() if str(n).lower() not in {"non-toxic", "non_hate", "normal", "safe", "nohate"}]

    def predict_batch(batch_texts):
        enc = tok(batch_texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        probs = torch.sigmoid(logits) if logits.shape[-1] > 2 else torch.softmax(logits, -1)
        if len(unsafe_idx) == logits.shape[-1]:
            return probs.max(-1).values  # all classes are toxic classes
        return probs[:, unsafe_idx].max(-1).values

    probs = []
    for start in range(0, len(texts), batch_size):
        probs.extend(predict_batch(texts[start : start + batch_size]).float().cpu().tolist())

    # latency / throughput / vram
    torch.cuda.reset_peak_memory_stats(device)
    for t in texts[:20]:
        predict_batch([t])
    lat_samples = texts[:200]
    t0 = time.perf_counter()
    for t in lat_samples:
        predict_batch([t])
    latency_ms = (time.perf_counter() - t0) / len(lat_samples) * 1000

    bench_texts = texts[:1024]
    t0 = time.perf_counter()
    for start in range(0, len(bench_texts), 32):
        predict_batch(bench_texts[start : start + 32])
    dt = time.perf_counter() - t0
    throughput = len(bench_texts) / dt
    vram_mb = torch.cuda.max_memory_allocated(device) / 1e6

    params = sum(p.numel() for p in model.parameters())
    from huggingface_hub import HfApi
    info = HfApi().model_info(repo)
    size_mb = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith((".safetensors", ".bin", ".pt"))) / 1e6
    del model
    torch.cuda.empty_cache()
    return {
        "repo": repo, "params": params, "checkpoint_mb": round(size_mb, 1),
        "latency_ms": latency_ms, "throughput_s": throughput, "vram_mb": vram_mb,
        "probs": probs,
    }


def bench_censer(texts, device, ckpt_path=None, tok_dir=None, label="KucLab/Censer.9 (this work)"):
    from kuclab.evaluate import predict_proba
    from kuclab.model import KucLabCneser, ModelConfig
    from kuclab.tokenizer import KucTokenizer

    ckpt_path = Path(ckpt_path) if ckpt_path else (REPO_ROOT / "checkpoints" / "best.pt")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    from kuclab.model import remap_legacy_state
    state = remap_legacy_state(state)
    mcfg = ModelConfig(**state["model_config"])
    model = KucLabCneser(mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    tok = KucTokenizer.load(tok_dir or (ckpt_path.parent / "tokenizer"))
    max_len = mcfg.max_seq_len

    def predict_batch(batch_texts):
        return predict_proba(model, tok, batch_texts, max_len, device, batch_size=len(batch_texts))

    probs = predict_proba(model, tok, texts, max_len, device)

    torch.cuda.reset_peak_memory_stats(device)
    for t in texts[:20]:
        predict_batch([t])
    lat_samples = texts[:200]
    t0 = time.perf_counter()
    for t in lat_samples:
        predict_batch([t])
    latency_ms = (time.perf_counter() - t0) / len(lat_samples) * 1000

    bench_texts = texts[:1024]
    t0 = time.perf_counter()
    for start in range(0, len(bench_texts), 8):
        predict_batch(bench_texts[start : start + 8])
    throughput = len(bench_texts) / (time.perf_counter() - t0)
    vram_mb = torch.cuda.max_memory_allocated(device) / 1e6

    return {
        "repo": label, "params": model.num_parameters(),
        "checkpoint_mb": round(Path(ckpt_path).stat().st_size / 1e6, 2),
        "latency_ms": latency_ms, "throughput_s": throughput, "vram_mb": vram_mb,
        "probs": probs,
    }


def bench_keywords(texts):
    kw = [re.escape(k) for k in KEYWORDS]
    pattern = re.compile("|".join(kw), re.IGNORECASE)
    t0 = time.perf_counter()
    probs = [1.0 if pattern.search(t) else 0.0 for t in texts]
    dt = time.perf_counter() - t0
    return {
        "repo": "keyword baseline", "params": 0, "checkpoint_mb": 0.0,
        "latency_ms": dt / len(texts) * 1000, "throughput_s": len(texts) / dt, "vram_mb": 0.0,
        "probs": probs,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})")
    texts, labels = load_test_rows()
    print(f"test rows: {len(texts)}  unsafe: {sum(labels)}")

    results = []
    censer = bench_censer(texts, device)
    censer["metrics"] = metrics(labels, censer.pop("probs"))
    results.append(censer)
    print(f"[ours] f1={censer['metrics']['f1']:.4f} acc={censer['metrics']['accuracy']:.4f} "
          f"lat={censer['latency_ms']:.2f}ms thr={censer['throughput_s']:.0f}/s vram={censer['vram_mb']:.0f}MB", flush=True)

    for label, ck, tk in [
        ("KucLab/Censer.5 (prev)", "checkpoints_ref/censer5.pt", "checkpoints_ref/tokenizer_censer5"),
        ("KucLab/Censer.4 (old)", "checkpoints_ref/censer4.pt", "checkpoints_ref/tokenizer_censer4"),
        ("KucLab/Censer.3 (old)", "checkpoints_ref/censer3.pt", "checkpoints_ref/tokenizer_censer3"),
        ("KucLab/Censer.2 (old)", "checkpoints_ref/censer2.pt", "checkpoints_ref/tokenizer_censer2"),
        ("KucLab/Censer.1 (baseline)", "checkpoints_ref/censer1.pt", "checkpoints_ref/tokenizer_censer1"),
    ]:
        ck_path = REPO_ROOT / ck
        if not ck_path.exists():
            continue
        base = bench_censer(texts, device, ck_path, REPO_ROOT / tk, label=label)
        base["metrics"] = metrics(labels, base.pop("probs"))
        results.append(base)
        print(f"[{label}] f1={base['metrics']['f1']:.4f} acc={base['metrics']['accuracy']:.4f} "
              f"lat={base['latency_ms']:.2f}ms thr={base['throughput_s']:.0f}/s vram={base['vram_mb']:.0f}MB", flush=True)

    kw = bench_keywords(texts)
    kw["metrics"] = metrics(labels, kw.pop("probs"))
    results.append(kw)
    print(f"[kw]   f1={kw['metrics']['f1']:.4f} acc={kw['metrics']['accuracy']:.4f}")

    for repo in HF_MODELS:
        print(f"evaluating {repo} ...", flush=True)
        r = bench_hf_model(repo, texts, device)
        r["metrics"] = metrics(labels, r.pop("probs"))
        results.append(r)
        print(f"[{repo.split('/')[-1][:12]}] f1={r['metrics']['f1']:.4f} acc={r['metrics']['accuracy']:.4f} "
              f"lat={r['latency_ms']:.2f}ms thr={r['throughput_s']:.0f}/s vram={r['vram_mb']:.0f}MB "
              f"params={r['params']/1e6:.1f}M size={r['checkpoint_mb']:.0f}MB", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "n_test": len(texts),
        "models": results,
    }, indent=2))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
