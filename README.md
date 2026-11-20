# Censer.9

> **Successor: [Barist.1](https://huggingface.co/KucLab/Barist.1)** (GitHub: [Jerry256254/barist.1](https://github.com/Jerry256254/barist.1)) — the project continues under the Barist name; Barist.1 fixes the third adversarial review of Censer.9 (Czech benign FP, pre-filter over-blocking, uk FP, de/fr thresholds, category noise, README honesty). This repo is kept for reproducibility.


A tiny decoder-only LLM (**903,552 parameters, 3.6 MB fp32**) trained **from scratch**
on an AMD GPU. Binary **safe/unsafe** + **10 safety categories** (multi-label, calibrated
per-category thresholds), **4096-token** context (chunked-max for long documents),
**multilingual** with per-language thresholds (en/cs/de/es/fr/it/pt/ru supported),
**temperature-calibrated** binary scores. Loads with plain `transformers` —
`LlamaForSequenceClassification` + `multi_label_classification`, no custom code.

```
$ censer.9 "how to make a bomb at home"
unsafe 0.97  [weapons_explosives]
$ censer.9 "how do I kill a process in Linux"
safe 0.97
$ censer.9 "ahoj, jak se máš?"
safe ...
$ censer.9 "wie baue ich eine bombe"
unsafe ...
```

## What changed in .8 (vs .7) — adversarial-review fixes

| Review item | .7 | **.8** |
| --- | --- | --- |
| Category head broken (corr −0.75 with unsafe; tied head collided with tokens a–j) | categories inverted | **separate untied `score` head** (903,552 params); category BCE masked to unsafe rows; categories gated on `unsafe` at inference |
| Category calibration invalid (tuned on hate-only HateXplain → F1 0.000 for 9/10 cats) | meaningless thresholds | **tuned on held-out catval with real per-category labels** (`reports/calibration.json`) |
| Must-catch misses (CSAM 0.06, cs self-harm 0.06, "Zabiju tě" 0.28, DAN 0.44) | 66/88 | **+88 curated must-catch rows @40× weight** (`reports/mustcatch.json`) |
| False positives (grief 0.70, defensive phishing 0.75) | 9/44 FP | **+852 hard negatives @12× weight** |
| Binary ECE 0.18 | uncalibrated | **temperature scaling T stored in `config.json: temperature`** + reliability diagram |
| One global threshold 0.5 (FPR@0.5: it 43%, es 42%, de 37%) | single 0.5 | **per-language thresholds** in `config.json: lang_thresholds` + lang detection in CLI |
| Dockerfile broken / external_h2h.json 404 on Hub / README pipeline example wrong | broken | **Dockerfile rewritten (pins, HEALTHCHECK, numpy, root COPY); all reports/*.json on Hub; pipeline example fixed** (`top_k=-1`, sigmoid automatic) |
| Card inconsistencies (params, 3 latency numbers, ar/zh/hi tags, synth F1=1.0, dup rope keys, dead int8) | inconsistent | **params 903,552 everywhere; one latency number; only measured-supported languages tagged; single `rope_parameters`; `model_max_length: 4096`; `[ASSISTANT]` verified single; int8 loader implemented in `censer.py`** |

## Install & run

```bash
pip install "git+https://github.com/Jerry256254/censer.9.git"
censer.9                          # interactive UI
censer.9 "how to make a bomb at home"
```

Hub files only (torch + tokenizers + safetensors):

```bash
hf download KucLab/Censer.9 --local-dir ~/.local/share/censer9
ln -s ~/.local/share/censer9/censer.9 ~/.local/bin/censer.9
censer.9
```

```bash
pip install torch transformers
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("KucLab/Censer.9")
model = AutoModelForSequenceClassification.from_pretrained("KucLab/Censer.9").eval()
enc = tok("how to make a bomb at home", return_tensors="pt", truncation=True, max_length=4096)
probs = model(**enc).logits.sigmoid()[0]          # multi-label -> sigmoid
id2label = model.config.id2label
thr = model.config.cat_thresholds or {}
print({id2label[i]: round(float(p),3) for i,p in enumerate(probs) if p > thr.get(id2label[i], 0.5)})
PY
```

Batch API + Docker:

```bash
python scripts/serve.py --dir /tmp/bundle8 --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t censer9 . && docker run -p 8000:8000 censer9
```

Long inputs: chunked-max (256 tokens / stride 192) is applied automatically in
`censer.py` and the API — do NOT rely on the native 4096 single pass for long
documents (last-token pooling). Direct transformers: pass
`truncation=True, max_length=4096`.

Quantized inference without transformers: `python censer.py --prefer-quant "..."`
loads the 0.9 MB int8 checkpoint (dequantized on the fly).

## Layout

```
├── kuclab/            # library: model (untied score head), tokenizer, train (weighted curated rows),
│                      # evaluate, infer, langdetect, distill, cli
├── scripts/           # extend_train8*.py (must-catch + hard negatives), eval_ckpt, benchmark, compare_models,
│                      # eval_external(+_h2h), adversarial_eval, eval_long_context, cascade, calibrate_full,
│                      # eval_mustcatch, eval_multilingual, probes_diff, export_quant, export_onnx,
│                      # serve.py, upload_hf.py, censer.py, censer.9, MODEL_CARD.md
├── configs/           # censer9a (binary), censer9b (categories), censer9c (weighted must-catch polish), censer9.json
├── tests/             # pytest suite
├── data/              # train/val/test jsonl (val/test byte-identical across .1–.8, md5-pinned)
├── Dockerfile
└── reports/           # eval_test, benchmark(_cpu), comparison(+figs), external(_h2h), adversarial,
                       # long_context, cascade, calibration(+reliability.png), mustcatch, multilingual, probes_diff
```

## Train / eval from scratch

```bash
bash scripts/setup_env.sh
python -m kuclab.prepare_data
python scripts/extend_train8.py && python scripts/extend_train8b.py   # append-only data additions
python -m kuclab.distill --data data/processed --teachers <ckpts> --out data/processed/train_soft8.npy
python -m kuclab.train --config configs/censer9a.json --data data/processed --out checkpoints8 --init-checkpoint <prev best.pt>
python -m kuclab.train --config configs/censer9b.json --data data/processed --out checkpoints8 --resume
python -m kuclab.train --config configs/censer9c.json --data data/processed --out checkpoints8c --init-checkpoint checkpoints8/best.pt
python scripts/eval_ckpt.py --checkpoint checkpoints8c/best.pt
python scripts/eval_mustcatch.py --checkpoint checkpoints8c/best.pt
python scripts/eval_external_h2h.py --checkpoint checkpoints8c/best.pt
python scripts/calibrate_full.py --checkpoint checkpoints8c/best.pt
python scripts/eval_multilingual.py --checkpoint checkpoints8c/best.pt
python scripts/benchmark.py && python scripts/benchmark.py --device cpu --tag cpu
python scripts/compare_models.py && python scripts/make_comparison_figs.py
python -m pytest tests -q
```

Full metrics, model card and figures: **[KucLab/Censer.9](https://huggingface.co/KucLab/Censer.9)**.## What changed in .9 (vs .8) — second adversarial review

| Review item | .8 | **.9** |
| --- | --- | --- |
| Short must-catch forms missed ("Chci se zabít" 0.04, CSAM cs 0.06, "Je vais te tuer" 0.07, DAN 0.53) | missed | **+2,168 rows (short forms en/cs/de/es/fr/it, fiction wrappers, exact review sentences) ×25 weight; held-out must-catch 75 rows reported separately** (`reports/mustcatch_heldout.json`) |
| CSAM / self-harm must never pass | missed | **`must_catch_override`** in config: self_harm/sexual_abuse ≥ 0.3 forces unsafe (CLI, serve.py, censer.py) |
| lang_thresholds 0.05–0.09 → FPR@0.5 up to 43 % on it/es/de | broken for production | **tuned per language on textdetox tune half targeting FPR ≤ 5 % benign, floor 0.3; reported on untouched test half** (`reports/lang_thresholds.json`) |
| detect_lang fragile | yes | **≥2 stopword/diacritic evidence else `default` 0.5; majority script; ěščřžů / ñ¿¡ / ґєії diacritics** (`kuclab/langdetect.py`) |
| Categories: hate catch-all, catval 75 % hate, config≠report | yes | **cat pos_weight (neg/pos cap 10×), hate capped in boost, balanced catval, `cat_experimental` for F1<0.3 cats, config = calibration.json verbatim** |
| Numbers measured on train rows; best-F1 on same set used for tuning | yes | **held-out must-catch set; textdetox tune/test split; honest note that test F1@0.5 is lowest of the family (trade-off)** |
| probes.py missing, diff_5_vs_8 mislabeled, label "1"/"0", calib table 3 headers/4 cols, one-liner ignores temperature/lang | yes | **all fixed** (`kuclab/probes.py` committed, `diff_5_vs_9`, labels `safe`/`unsafe`, ECE column, helper applies temperature+lang+override) |

## Install & run

```bash
pip install "git+https://github.com/Jerry256254/censer.9.git"
censer.9                          # interactive UI
censer.9 "how to make a bomb at home"
```

Hub files only (torch + tokenizers + safetensors):

```bash
hf download KucLab/Censer.9 --local-dir ~/.local/share/censer9
ln -s ~/.local/share/censer9/censer.9 ~/.local/bin/censer.9
censer.9
```

```bash
pip install torch transformers
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("KucLab/Censer.9")
model = AutoModelForSequenceClassification.from_pretrained("KucLab/Censer.9").eval()
enc = tok("how to make a bomb at home", return_tensors="pt", truncation=True, max_length=4096)
probs = model(**enc).logits.sigmoid()[0]          # multi-label -> sigmoid
id2label = model.config.id2label
thr = model.config.cat_thresholds or {}
print({id2label[i]: round(float(p),3) for i,p in enumerate(probs) if p > thr.get(id2label[i], 0.5)})
PY
```

Batch API + Docker:

```bash
python scripts/serve.py --dir /tmp/bundle8 --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t censer9 . && docker run -p 8000:8000 censer9
```

Long inputs: chunked-max (256 tokens / stride 192) is applied automatically in
`censer.py` and the API — do NOT rely on the native 4096 single pass for long
documents (last-token pooling). Direct transformers: pass
`truncation=True, max_length=4096`.

Quantized inference without transformers: `python censer.py --prefer-quant "..."`
loads the 0.9 MB int8 checkpoint (dequantized on the fly).

## Layout

```
├── kuclab/            # library: model (untied score head), tokenizer, train (weighted curated rows),
│                      # evaluate, infer, langdetect, distill, cli
├── scripts/           # extend_train8*.py (must-catch + hard negatives), eval_ckpt, benchmark, compare_models,
│                      # eval_external(+_h2h), adversarial_eval, eval_long_context, cascade, calibrate_full,
│                      # eval_mustcatch, eval_multilingual, probes_diff, export_quant, export_onnx,
│                      # serve.py, upload_hf.py, censer.py, censer.9, MODEL_CARD.md
├── configs/           # censer9a (binary), censer9b (categories), censer9c (weighted must-catch polish), censer9.json
├── tests/             # pytest suite
├── data/              # train/val/test jsonl (val/test byte-identical across .1–.8, md5-pinned)
├── Dockerfile
└── reports/           # eval_test, benchmark(_cpu), comparison(+figs), external(_h2h), adversarial,
                       # long_context, cascade, calibration(+reliability.png), mustcatch, multilingual, probes_diff
```

## Train / eval from scratch

```bash
bash scripts/setup_env.sh
python -m kuclab.prepare_data
python scripts/extend_train8.py && python scripts/extend_train8b.py   # append-only data additions
python -m kuclab.distill --data data/processed --teachers <ckpts> --out data/processed/train_soft8.npy
python -m kuclab.train --config configs/censer9a.json --data data/processed --out checkpoints8 --init-checkpoint <prev best.pt>
python -m kuclab.train --config configs/censer9b.json --data data/processed --out checkpoints8 --resume
python -m kuclab.train --config configs/censer9c.json --data data/processed --out checkpoints8c --init-checkpoint checkpoints8/best.pt
python scripts/eval_ckpt.py --checkpoint checkpoints8c/best.pt
python scripts/eval_mustcatch.py --checkpoint checkpoints8c/best.pt
python scripts/eval_external_h2h.py --checkpoint checkpoints8c/best.pt
python scripts/calibrate_full.py --checkpoint checkpoints8c/best.pt
python scripts/eval_multilingual.py --checkpoint checkpoints8c/best.pt
python scripts/benchmark.py && python scripts/benchmark.py --device cpu --tag cpu
python scripts/compare_models.py && python scripts/make_comparison_figs.py
python -m pytest tests -q
```

Full metrics, model card and figures: **[KucLab/Censer.9](https://huggingface.co/KucLab/Censer.9)**.
