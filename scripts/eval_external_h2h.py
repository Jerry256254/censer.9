"""Head-to-head on external benchmarks: Censer.7 vs public safety classifiers.

Evaluates on ToxiGen (annotated) and HateXplain (test split) — never used for
training — and reports F1/accuracy/latency for Censer.7 AND
unitary/unbiased-toxic-roberta + unitary/toxic-bert, so the comparison table in
the model card is apples-to-apples on the external sets too.

    .venv/bin/python scripts/eval_external_h2h.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from kuclab.evaluate import load_checkpoint, metrics_from, predict_proba
from eval_external import load_hatexplain, load_toxigen  # reuse loaders


def eval_hf_model(repo: str, texts, labels, device, max_chars=4000):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForSequenceClassification.from_pretrained(repo).to(device).eval()
    n_toxic = sum(1 for v in model.config.id2label.values() if v.lower() in {
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
        "hate", "offensive", "sexual_explicit", "toxicity"})
    toxic_idx = [i for i, v in model.config.id2label.items() if v.lower() in {
        "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate",
        "hate", "offensive", "sexual_explicit", "toxicity"}]
    probs = []
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], return_tensors="pt", truncation=True, max_length=512, padding=True).to(device)
            lg = model(**enc).logits.float()
            if len(toxic_idx) > 1:
                p = lg.softmax(-1)[:, toxic_idx].sum(-1)
            else:
                p = lg.softmax(-1)[:, -1]
            probs.extend(p.tolist())
    lat = (time.time() - t0) / len(texts) * 1000
    m05 = metrics_from(labels, probs, 0.5)
    best = (0.5, m05)
    for i in range(5, 96):
        th = i / 100
        m = metrics_from(labels, probs, th)
        if m["f1"] > best[1]["f1"]:
            best = (th, m)
    return {"at_0.5": m05, "best": {"threshold": best[0], **best[1]}, "latency_ms": lat}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="reports/external_h2h.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    sets = {"toxigen": load_toxigen(), "hatexplain_test": load_hatexplain()}
    results = {}
    for name, (texts, labels) in sets.items():
        probs = predict_proba(model, tok, [t[:4000] for t in texts], max_len, device, batch_size=64)
        m05 = metrics_from(labels, probs, 0.5)
        best = (0.5, m05)
        for i in range(5, 96):
            th = i / 100
            m = metrics_from(labels, probs, th)
            if m["f1"] > best[1]["f1"]:
                best = (th, m)
        results[name] = {"n": len(labels), "unsafe": sum(labels),
                         "censer7": {"at_0.5": m05, "best": {"threshold": best[0], **best[1]}}}
        print(f"[censer.7] {name}: n={len(labels)} @0.5 f1={m05['f1']:.4f} acc={m05['accuracy']:.4f} best@{best[0]:.2f} f1={best[1]['f1']:.4f}", flush=True)
        for repo in ["unitary/unbiased-toxic-roberta", "unitary/toxic-bert"]:
            try:
                r = eval_hf_model(repo, texts, labels, device)
                results[name][repo] = r
                print(f"  {repo}: @0.5 f1={r['at_0.5']['f1']:.4f} acc={r['at_0.5']['accuracy']:.4f} "
                      f"best@{r['best']['threshold']:.2f} f1={r['best']['f1']:.4f} lat={r['latency_ms']:.1f}ms", flush=True)
            except Exception as exc:
                print(f"  {repo}: skipped ({str(exc)[:100]})", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "device": str(device), "results": results}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
