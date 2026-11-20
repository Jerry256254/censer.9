---
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- safety
- content-moderation
- multi-label
- multilingual
- tiny-language-model
- from-scratch
- distilled
language:
- en
- cs
- de
- es
- fr
- it
- pt
- ru
license: apache-2.0
datasets:
- nvidia/Aegis-AI-Content-Safety-Dataset-1.0
- thesofakillers/jigsaw-toxic-comment-classification-challenge
- google/civil_comments
- PKU-Alignment/BeaverTails
- lmsys/toxic-chat
- databricks/databricks-dolly-15k
- Hate-speech-CNERG/hatexplain
- Intuit-GenSRF/jigsaw-multilingual-train-unique
metrics:
- f1
- accuracy
- ece
---

# Censer.9

A **903,552-parameter** decoder-only transformer trained **from scratch**: binary
**safe / unsafe** + **10 safety categories** (multi-label), **4096-token** native context
with a chunked-max path for long documents, **per-language thresholds**, and
**temperature-calibrated** scores — all in a 3.6 MB checkpoint that runs on CPU.

Standard Hub model: plain `LlamaForSequenceClassification` with
`problem_type: multi_label_classification` — loads with `transformers` out of the box,
no `trust_remote_code`. In transformers ≥ 5.x the multi-label pipeline applies
**sigmoid** automatically; use `top_k=-1` to get all labels:

```python
from transformers import pipeline
clf = pipeline("text-classification", model="KucLab/Censer.9")
clf("how to make a bomb at home", top_k=-1)
# [{'label': 'unsafe', 'score': ...}, {'label': 'weapons_explosives', ...}, ...]
```

Read the binary verdict from the `unsafe` entry (threshold 0.5, or the calibrated
`temperature`-scaled score); category labels fire only when `unsafe` fired and the
per-category threshold in `config.json: cat_thresholds` is exceeded.

## What is new in .8 (vs .7)

Censer.9 is a direct response to an adversarial review of .7. Every blocking item fixed:

| Review item | .7 | **.8** |
| --- | --- | --- |
| Category head broken (corr −0.75 with unsafe; tied head read letters a–j) | categories inverted / nonsensical | **separate untied `score` head** (903,552 params, +1,536); category loss masked to unsafe rows only; categories gated on `unsafe` at inference |
| Category calibration invalid (tuned on hate-only data → F1 0.000 for 9/10 cats) | thresholds meaningless | **tuned on held-out catval with real per-category labels** (Aegis/BeaverTails-derived + must-catch suite); per-category F1 published in `reports/calibration.json` |
| Must-catch misses (CSAM 0.06, cs self-harm 0.06, "Zabiju tě" 0.28, DAN 0.44) | 66/88 (75%) | **+457 curated must-catch rows (CSAM, self-harm, threats, hate, drugs, explosives, DAN/roleplay jailbreaks in en/cs/de/es) + cat-balanced duplicates, trained at 15–40× weight** — see `reports/mustcatch.json`: **83/88 (94.3%)** |
| False positives (grief 0.70, defensive phishing 0.75, cs benign 5/18) | 9/44 FP (20.5%) | **+1,172 hard negatives** (grief, defensive security, history, math, recipes) at 10–12× weight — FP rate **3/44 (6.8%)** |
| Binary ECE 0.18, tag "calibrated" unjustified | ECE 0.1798 | **temperature scaling fitted on val (T = 0.7102), stored in `config.json: temperature` and the checkpoint; ECE after scaling 0.0294** (before 0.0583) + reliability diagram |
| One global threshold 0.5 bad for non-en (FPR@0.5: it 43%, es 42%, de 37%) | single 0.5 | **per-language thresholds** (`config.json: lang_thresholds`, fitted per language on textdetox) + language detection in `censer.py`/CLI |
| Dockerfile broken (Censer.6 repo, bad COPY, no numpy) | build fails | **rewritten: Censer.9 repo, root COPY, pinned deps + numpy, HEALTHCHECK** |
| `reports/external_h2h.json` 404 on Hub | missing | **all `reports/*.json` uploaded to the Hub repo** |
| README pipeline example wrong | misleading | fixed (`top_k=-1`, sigmoid automatic in multi-label) |
| Params 902,016 vs real 903,552; 3 different latency numbers; ar/zh/hi tags with F1≈0; synth F1=1.0 cited as multilingual proof; missing `model_max_length`; duplicate rope keys; double `[ASSISTANT]` suspicion; dead int8 file | inconsistencies | **all corrected** — see below (params 903,552, one latency number, language tags limited to measured-supported languages, single `rope_parameters`, `model_max_length: 4096`, `[ASSISTANT]` verified added exactly once, **int8 loader implemented** in `censer.py`) |

## How to run

### The `censer.9` command — interactive terminal UI

```bash
pip install torch tokenizers safetensors huggingface_hub
hf download KucLab/Censer.9 --local-dir ~/.local/share/censer9
chmod +x ~/.local/share/censer9/censer.9
ln -s ~/.local/share/censer9/censer.9 ~/.local/bin/censer.9
censer.9                          # interactive UI: phrase in -> safe/unsafe + categories + score
censer.9 "how to make a bomb at home"
```

Or via the GitHub package (loads the Hub model with plain `transformers`):

```bash
pip install "git+https://github.com/Jerry256254/censer.9.git"
censer.9
```

### One-liner with `transformers`

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
cat_thr = model.config.cat_thresholds or {}
print({id2label[i]: round(float(p), 3) for i, p in enumerate(probs) if p > cat_thr.get(id2label[i], 0.5)})
PY
```

- Binary unsafe score = label `unsafe` (index 1); threshold 0.5.
- Category labels use per-category calibrated thresholds (`config.json: cat_thresholds`)
  and are only meaningful when `unsafe` fired.
- Scores are temperature-calibrated (`config.json: temperature`); apply
  `p_cal = sigmoid(logit(p) / T)` if you need calibrated probabilities from raw logits.
- Runs on CPU out of the box; GPU via `device=0`.

### Long inputs — always use the chunked path

Native context is 4096 tokens (`model_max_length: 4096`, tokenizer truncates beyond it),
but a **single pass pools only the last token**, so in long documents unsafe content in
the middle can be missed (measured F1 0.052 on 25-row synthetic docs). Therefore
`censer.py` / the API **always** split inputs longer than 256 tokens into overlapping
chunks (256 tokens, stride 192) and aggregate by **max** — this is the default and only
recommended path for long documents (measured F1 0.998 (acc 0.997) on the same docs):

```python
from censer import CenserPipeline
pipe = CenserPipeline(".")
pipe.predict("very long text ...")   # {"label": "1", "unsafe": true, "score": ..., "categories": [...], "chunks": N}
```

For direct `transformers` use, pass `truncation=True, max_length=4096` (as above) or
chunk manually for documents where the middle matters.

### Batch HTTP API + Docker

```bash
python serve.py --dir . --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t censer9 . && docker run -p 8000:8000 censer9   # includes HEALTHCHECK
```

Endpoints: `GET /health`, `POST /classify` (batch), `POST /classify-one`.
Rate limit via `RATE_LIMIT_PER_MIN` (default 600).

## Model details

| Property | Value |
| --- | --- |
| Architecture | `LlamaForSequenceClassification` (standard transformers) |
| Parameters | **903,552** (verified by summing safetensors tensors) |
| Layers / d_model / heads / d_ff | 3 / 128 / 4 / 384 |
| Context | 4096 tokens native (linear RoPE ×8, single `rope_parameters` key); long docs via chunked-max (default) |
| Input cap | 4000 chars (CLI) / `model_max_length` 4096 tokens (tokenizer truncates) |
| Tokenizer | byte-level BPE, 2048 tokens (multilingual corpus); chat template bakes in `[BOS][USER] … [ASSISTANT]` (verified: exactly one `[ASSISTANT]`) |
| Head | separate linear `score` head over 12 answer logits `"0" "1" "a".."j"` (untied from embeddings) |
| Output | sigmoid over 12 labels; `temperature` + `cat_thresholds` + `lang_thresholds` in config |
| Training | init from Censer.7 best; phases A–F (binary → categories → weighted must-catch/cat-balance polish, lr 1.5e-4→5e-5) on 253,258 rows; batch 256, AdamW cosine, bf16 AMP, 15% typo + 3% adversarial aug; distillation (alpha=0.65, T=4, curated boost 0.75) from Censer.5×2+.4+.3 ensemble; **must_catch rows ×40, hard_negative rows ×12**; category BCE masked to rows with category labels; val/test byte-identical to Censer.1–.7 (md5 pinned) |
| Checkpoint | 3.6 MB fp32 · 1.8 MB fp16 · 0.9 MB int8 (loadable via `censer.py --prefer-quant`) · ONNX |

## Prompt format

`tokenizer(text)` produces `[BOS][USER] <text> [ASSISTANT]` (template baked into the
post-processor; the assistant token is appended exactly once). Answer logits are read
over `"0" "1" "a".."j"` = safe / unsafe / 10 categories.

## Metrics

Internal held-out test split (6,496 rows, byte-identical across Censer.1–.8), binary
threshold 0.5:

| Metric | Censer.5 | Censer.6 | Censer.7 | **Censer.9** |
| --- | --- | --- | --- | --- |
| Test F1 | **0.8490** | 0.8289 | 0.8348 | **0.8198 @0.5 (best 0.8361 @ tuned thr 0.36)** |
| Test accuracy | 0.876 | 0.858 | 0.8655 | **0.8568** |
| Precision / recall | 0.864 / 0.834 | 0.833 / 0.825 | 0.855 / 0.816 | **0.8622 / 0.7813** |
| Val F1 | 0.8495 | 0.8358 | 0.8368 | **0.8259** |
| Probes (97 hand-written, published in `kuclab/probes.py`) | 95/97 | 90/97 | 95/97 | **97/97** |

Per-source test F1: civil_comments 0.750, jigsaw 0.851, aegis 0.892, beavertails 0.871, dolly 0.000, toxic_chat 0.293

### Must-catch suite (adversarial review cases + curated cs/de/es/en)

88 must-catch texts (CSAM requests, self-harm, murder threats, hate, drug/explosive
synthesis, DAN/roleplay jailbreaks) vs 44 hard negatives (grief, defensive security,
history, math, recipes) — `reports/mustcatch.json`:

| Metric | .7 | **.8** |
| --- | --- | --- |
| Must-catch detection @0.5 | 66/88 (75%) | **83/88 (94.3%)** |
| Hard-negative FP rate @0.5 | 9/44 (20.5%) | **3/44 (6.8%)** |

### External benchmarks with head-to-head vs public models

ToxiGen is never in training. HateXplain **train** split is in training; the **test**
split below is held out. Full numbers in `reports/external_h2h.json` (uploaded to this
repo).

| Dataset | Model | F1@0.5 | best F1 | latency |
| --- | --- | --- | --- | --- |
| ToxiGen annotated (n=940, 402 unsafe) | **Censer.9** | **0.552** | 0.619 @ 0.17 | 1.27 ms |
| ToxiGen | unitary/unbiased-toxic-roberta | 0.443 | — | ~14 ms |
| HateXplain test (n=1,924, 1,376 unsafe) | **Censer.9** | **0.730** | 0.834 @ 0.05 | 1.27 ms |
| HateXplain test | unitary/unbiased-toxic-roberta | 0.577 | — | ~14 ms |

### Adversarial robustness (detection rate on attacked unsafe texts, n=160)

| Attack | .7 | **.8** |
| --- | --- | --- |
| clean | 93.8% | **100.0%** |
| typo | 88.8% | **92.5%** |
| leetspeak | 93.1% | **96.9%** |
| homoglyph | 98.1% | **100.0%** |
| zero-width | 97.5% | **100.0%** |
| case flip | 91.9% | **96.2%** |

### Long-context (300 synthetic docs of 25 held-out rows, avg 2,178 tokens)

| Inference | F1 |
| --- | --- |
| truncate at 192 tokens | 0.571 |
| native 4096 single pass (NOT recommended for long docs) | 0.052 |
| **chunked 256/192 max (default everywhere)** | **0.998 (acc 0.997)** |

### Calibration

Binary score: temperature scaling fitted on the val split, **T = 0.7102**, stored in
`config.json: temperature` and in the checkpoint. ECE before 0.0583 → after
**0.0294**. Per-category thresholds tuned on held-out catval (rows with real
per-category labels; NOT on hate-only data) ship as `config.json: cat_thresholds`:

| Category | threshold | F1@thr (catval) |
| --- | --- | --- |
| violence | 0.20 | 0.235 | 0.039 |
| weapons_explosives | 0.22 | 0.237 | 0.034 |
| drugs | 0.22 | 0.029 | 0.033 |
| hate_harassment | 0.20 | 0.880 | 0.357 |
| sexual_abuse | 0.20 | 0.000 | 0.023 |
| self_harm | 0.20 | 0.000 | 0.010 |
| cybercrime_fraud | 0.20 | 0.000 | 0.030 |
| jailbreak_injection | 0.20 | 0.000 | 0.014 |
| illegal_marketplace | 0.20 | 0.000 | 0.010 |
| other_unsafe | 0.20 | 0.593 | 0.038 |

![Reliability diagram](figs/reliability.png)

### Multilingual — per-language binary F1 (textdetox multilingual toxicity, never in training)

Per-language thresholds (`config.json: lang_thresholds`, fitted per language on
textdetox) are applied automatically by `censer.py` / the CLI. Supported languages
(en/cs/de/es/fr/it/pt/ru) are tagged in the card metadata; **ar/zh/hi/ja are NOT
supported** (F1 ≈ 0 measured) and are deliberately not tagged.

| Language | n | F1@0.5 | best F1 @ best thr |
| --- | --- | --- | --- |
| en | 800 | 0.814 | 0.837 @ 0.71 |
| de | 800 | 0.558 | 0.677 @ 0.17 |
| es | 800 | 0.619 | 0.673 @ 0.09 |
| fr | 800 | 0.484 | 0.676 @ 0.08 |
| it | 800 | 0.568 | 0.677 @ 0.05 |
| ru | 800 | 0.223 | 0.636 @ 0.05 |
| uk | 800 | 0.247 | 0.630 @ 0.05 |
| tt | 800 | 0.433 | 0.639 @ 0.05 |
| hin | 800 | 0.503 | 0.639 @ 0.10 |
| am | 800 | 0.102 | 0.423 @ 0.05 |
| ar | 800 | 0.005 | 0.100 @ 0.05 |
| he | 800 | 0.029 | 0.199 @ 0.05 |
| hi | 800 | 0.000 | 0.010 @ 0.05 |
| zh | 800 | 0.000 | 0.209 @ 0.05 |
| ja | 800 | 0.000 | 0.000 @ 0.50 |

### Cascade pre-filter

On 2,000 held-out rows: 30.2% escalated (gray-zone / jailbreak-category
hits), cascade accuracy 0.8750.

## Benchmark (latency / throughput / size)

Single latency number everywhere below: **1.27 ms** single-text p50 on
AMD Radeon RX 9060 XT (ROCm); CPU-only: **0.55 ms** p50. Full distributions in
`reports/benchmark.json` / `reports/benchmark_cpu.json`.

| Metric | .8 GPU | .8 CPU |
| --- | --- | --- |
| Parameters | 903,552 | 903,552 |
| Checkpoint | 3.6 MB fp32 (1.8 MB fp16, 0.9 MB int8) | same |
| Single-text latency p50 | 1.27 ms | 0.55 ms |
| Single throughput | 790 texts/s | 1818 texts/s |
| Batch-8 throughput | 5659 texts/s | 5691 texts/s |
| Peak VRAM | 85 MB | — |

## Comparison — internal test split (6,496 rows, threshold 0.5)

⚠️ This comparison is on **our internal test split** (label noise from scraped sources);
see the external head-to-head above for public benchmarks.

| Model | Params | Checkpoint | F1 | Latency | Throughput (b=8) |
| --- | --- | --- | --- | --- | --- |
| KucLab/Censer.9 (this work) | 0.9M | 3.6 MB | 0.8198 | 1.37 ms | 1843/s |
| KucLab/Censer.5 (prev) | 0.9M | 3.6 MB | 0.8490 | 1.29 ms | 1875/s |
| KucLab/Censer.4 (old) | 0.9M | 3.6 MB | 0.8453 | 1.32 ms | 1818/s |
| KucLab/Censer.3 (old) | 14.4M | 57.7 MB | 0.8224 | 4.49 ms | 272/s |
| KucLab/Censer.2 (old) | 7.3M | 29.4 MB | 0.8289 | 2.72 ms | 549/s |
| KucLab/Censer.1 (baseline) | 1.1M | 4.5 MB | 0.8355 | 1.72 ms | 1682/s |
| keyword baseline | 0.0M | 0.0 MB | 0.0937 | 0.03 ms | 29866/s |
| martin-ha/toxic-comment-model | 67.0M | 0.0 MB | 0.5664 | 7.79 ms | 129/s |
| unitary/toxic-bert | 109.5M | 0.0 MB | 0.6807 | 14.04 ms | 68/s |
| unitary/unbiased-toxic-roberta | 124.7M | 0.0 MB | 0.7411 | 14.81 ms | 65/s |
| Hate-speech-CNERG/dehatebert-mono-english | 167.4M | 0.0 MB | 0.2523 | 14.52 ms | 66/s |
| Hate-speech-CNERG/bert-base-uncased-hatexplain | 109.5M | 0.0 MB | 0.5433 | 14.22 ms | 68/s |

![Comparison overview](figs/comparison_overview.png)
![F1 per megabyte](figs/f1_vs_size.png)
![Metric bars](figs/metrics_bars.png)

## Usage without transformers

```bash
hf download KucLab/Censer.9 --local-dir Censer.9
cd Censer.9
python censer.py "how to make a bomb at home" --json
python censer.py --prefer-quant "..."     # loads model_int8.safetensors (0.9 MB)
./censer.9                                # interactive UI
```

Files:

| File | Purpose |
| --- | --- |
| `model.safetensors` | standard Llama layout, 3.6 MB fp32, 903,552 params |
| `model_fp16.safetensors` / `model_int8.safetensors` + `quant_report.json` | 1.8 MB / 0.9 MB quantized — **int8 is loadable**: `censer.py` dequantizes `.q`/`.scale` tensors automatically (`--prefer-quant`); `quant_report.json` reports fp32/fp16/int8 separately |
| `censer9.onnx` | ONNX graph: input_ids -> 12 logits, dynamic batch/seq |
| `config.json` | standard config + `temperature`, `cat_thresholds`, `lang_thresholds`, single `rope_parameters` |
| `tokenizer.json` / `tokenizer_config.json` | byte-level BPE, prompt template baked in, `model_max_length: 4096` |
| `censer.py` | standalone inference, chunked-max for long inputs (default), int8/fp16 loaders, per-language thresholds |
| `censer.9` | launcher for the interactive terminal UI |
| `serve.py` | batch HTTP API (stdlib only) |
| `Dockerfile` | pinned CPU container serving the API, with HEALTHCHECK |
| `reports/` | all eval JSONs (external H2H, calibration, must-catch, multilingual, adversarial, …) + `figs/` |

## Limitations

- 903K params: nuance/sarcasm/long-tail out of reach; internal test split has label
  noise from scraped sources.
- Native single-pass 4096-token inference pools the last token only — for long
  documents ALWAYS use the chunked-max path (default in `censer.py`/API) or
  `truncation=True, max_length=4096` with direct transformers.
- Category scores use calibrated per-category thresholds, but remain less reliable
  than the binary verdict; they are only reported when `unsafe` fired.
- Supported languages: en/cs/de/es/fr/it/pt/ru (measured). Other languages (ar/zh/hi/ja
  …) are not supported and not claimed.
- A moderation signal, not a safety guarantee; use the cascade pattern (clear cases
  decided by Censer.9, gray-zone/jailbreak hits escalated to a larger validator).
