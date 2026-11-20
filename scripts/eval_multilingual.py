"""Multilingual evaluation: per-language binary F1.

Two sources:
1. textdetox/multilingual_toxicity_dataset — one split per language (de, es, fr, it,
   pt, ru, zh, ar, hi, ...); we sample up to --per-lang rows per language. This set is
   NEVER used for training.
2. Held-out synthetic cs/de/es/fr probes generated with a different seed than training.

Writes reports/multilingual.json + .md.

    .venv/bin/python scripts/eval_multilingual.py --checkpoint checkpoints/best.pt
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from kuclab.evaluate import load_checkpoint, metrics_from, predict_proba
from kuclab.multilingual import generate_ml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--per-lang", type=int, default=800)
    parser.add_argument("--out", default="reports/multilingual.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)
    max_len = mcfg.max_seq_len

    results = {}

    # 1. textdetox per-language splits, with a seeded 50/50 tune/test split per
    #    language (Censer.9: thresholds are tuned on the tune half targeting
    #    FPR<=5% on benign rows, floor 0.3; reported on the untouched test half)
    lang_thresholds_out = {"default": 0.5, "tuned_on": "tune half", "target": "FPR<=5% benign, floor 0.3", "langs": {}}
    try:
        from datasets import load_dataset

        ds = load_dataset("textdetox/multilingual_toxicity_dataset")
        for lang in sorted(ds.keys()):
            split = ds[lang]
            if len(split) > args.per_lang:
                idx = random.Random(13).sample(range(len(split)), args.per_lang)
                split = split.select(idx)
            texts = [str(r["text"])[:4000] for r in split]
            labels = [int(r["toxic"]) for r in split]
            if sum(labels) == 0:
                continue
            probs = predict_proba(model, tok, texts, max_len, device, batch_size=64)
            # seeded 50/50 tune/test split (per language)
            order = list(range(len(texts)))
            random.Random(777 + hash(lang) % 1000).shuffle(order)
            half = len(order) // 2
            tune_idx, test_idx = order[:half], order[half:]
            t_texts = [texts[i] for i in tune_idx]; t_labels = [labels[i] for i in tune_idx]
            s_texts = [texts[i] for i in test_idx]; s_labels = [labels[i] for i in test_idx]
            t_probs = [probs[i] for i in tune_idx]; s_probs = [probs[i] for i in test_idx]

            # tune: maximize F1 on tune half subject to FPR<=5% on benign, floor 0.3
            FLOOR, FPR_TARGET = 0.3, 0.05
            best_thr, best_f1 = None, -1.0
            for i in range(30, 96):
                th = i / 100
                tp = sum(1 for l, p in zip(t_labels, t_probs) if l == 1 and p >= th)
                fp = sum(1 for l, p in zip(t_labels, t_probs) if l == 0 and p >= th)
                fn = sum(1 for l, p in zip(t_labels, t_probs) if l == 1 and p < th)
                n_neg = sum(1 for l in t_labels if l == 0)
                fpr = fp / max(1, n_neg)
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                if fpr <= FPR_TARGET and f1 > best_f1:
                    best_f1, best_thr = f1, th
            if best_thr is None:
                best_thr = 0.5  # could not hit FPR target -> default
            # report on the untouched test half
            m05 = metrics_from(s_labels, s_probs, 0.5)
            mthr = metrics_from(s_labels, s_probs, best_thr)
            fpr_test = sum(1 for l, p in zip(s_labels, s_probs) if l == 0 and p >= best_thr) / max(1, sum(1 for l in s_labels if l == 0))
            results[f"textdetox_{lang}"] = {"n": len(labels), "unsafe": sum(labels),
                                             "f1@0.5": metrics_from(labels, probs, 0.5)["f1"],
                                             "acc@0.5": metrics_from(labels, probs, 0.5)["accuracy"],
                                             "test_n": len(s_labels),
                                             "test_f1@tuned": mthr["f1"], "test_fpr@tuned": fpr_test,
                                             "tuned_thr": best_thr,
                                             "note": "thr tuned on tune half (FPR<=5% benign, floor 0.3); F1/FPR on test half"}
            if results[f"textdetox_{lang}"]["f1@0.5"] >= 0.45:  # keep langs where the model works at 0.5 at all
                lang_thresholds_out["langs"][lang] = round(best_thr, 3)
            print(f"textdetox_{lang}: n={len(labels)} f1@0.5={results[f'textdetox_{lang}']['f1@0.5']:.4f} "
                  f"test_f1@{best_thr:.2f}={mthr['f1']:.4f} test_fpr={fpr_test:.3f}", flush=True)
    except Exception as exc:
        print("textdetox skipped:", str(exc)[:150])
    Path("reports/lang_thresholds.json").write_text(json.dumps(lang_thresholds_out, indent=2, ensure_ascii=False))

    # 2. held-out synthetic cs/de/es/fr (different seed than training data)
    rows = generate_ml(count_per_lang=150, seed=999)  # seed 999 != training seed; rows are (text, label, source)
    for lang in ["cs", "de", "es", "fr"]:
        sel = [(t, l) for t, l, src in rows if src.endswith("_" + lang)]
        texts = [t for t, _ in sel]
        labels = [l for _, l in sel]
        if not texts:
            continue
        probs = predict_proba(model, tok, texts, max_len, device, batch_size=64)
        m = metrics_from(labels, probs, 0.5)
        results[f"synth_{lang}"] = {"n": len(labels), "unsafe": sum(labels), "f1@0.5": m["f1"], "acc@0.5": m["accuracy"]}
        print(f"synth_{lang}: n={len(labels)} f1@0.5={m['f1']:.4f} acc={m['accuracy']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"checkpoint": args.checkpoint, "results": results}, indent=2))
    md = ["# Multilingual eval (textdetox: thr tuned on tune half, F1/FPR on untouched test half)", "",
          "| set | n | F1@0.5 | tuned thr | test F1@thr | test FPR@thr |", "| --- | --- | --- | --- | --- | --- |"]
    for k, v in results.items():
        if k.startswith("textdetox_"):
            md.append(f"| {k} | {v['n']} | {v.get('f1@0.5',0):.4f} | {v.get('tuned_thr','—')} | {v.get('test_f1@tuned','—')} | {v.get('test_fpr@tuned','—')} |")
        else:
            md.append(f"| {k} | {v['n']} | {v.get('f1@0.5',0):.4f} | — | — | — |")
    (Path("reports/multilingual.md")).write_text("\n".join(md) + "\n")
    print("wrote", out, "and reports/multilingual.md")


if __name__ == "__main__":
    main()
