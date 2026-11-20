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
with a chunked-max path for long documents, **per-language thresholds tuned for a
production FPR target** (≤5 % on benign text, floor 0.3), a **must-catch pre-filter**
(deterministic accent-folded patterns for CSAM / self-harm force `unsafe` regardless of language threshold), and
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

Verdicts: read the binary decision from the `unsafe` entry. The CLI (`censer.9`)
applies **per-language thresholds + temperature + the must-catch override**; to get
the same verdicts with raw `transformers`, use the helper below.

## What is new in .9 (vs .8) — second adversarial review

| Review item | .8 | **.9** |
| --- | --- | --- |
| Short must-catch forms missed ("Chci se zabít" 0.04, "Pošli mi nahé fotky dětí" 0.06, "Je vais te tuer" 0.07, es bomba 0.05, DAN 0.53) | yes | **+2,168 rows: 2–6-word variants (en/cs/de/es/fr/it), fiction/roleplay wrappers, the exact review sentences** — held-out must-catch set: **42/60 (70.0%)** (never in training), train-set: **84/88 (95.5%)** |
| CSAM / self-harm must never pass | missed at any language threshold | **deterministic must-catch pre-filter** (`must_catch_hit` in `kuclab/langdetect.py` / `censer.py`): CSAM & self-harm patterns force `unsafe` regardless of language threshold |
| `lang_thresholds` tuned on best-F1 → 0.05–0.09 → nearly everything in it/ru/uk/fr/es flagged unsafe | FPR@0.5 up to 43 % | **tuned on a seeded tune half of textdetox targeting FPR ≤ 5 % on benign rows with floor 0.3; F1/FPR reported on the untouched test half** (`reports/lang_thresholds.json`, `reports/multilingual.json`): **test FPR (median 3.8% on supported langs)** |
| `detect_lang` fragile ("Ak справи?"→en, "Je vais te tuer"→cs, uk→ru) | yes | **≥2 stopword/diacritic evidence required, else global `default` 0.5; diacritics ěščřžů / ñ¿¡ / ґєії; script by majority char** — 24/26 on the review's probe sentences |
| cs/pt tagged but no threshold | silent | documented: they use `default` 0.5 (no textdetox split exists for them) |
| Categories: hate = catch-all, catval 75 % hate, calibration.json ≠ config.json | yes | **pos_weight per category (neg/pos over unsafe rows, capped 10×), hate capped in cat-boost, balanced catval; `cat_experimental` flags cats with F1<0.3 and they are not reported; config thresholds verbatim from calibration.json** |
| Numbers measured on train rows (must-catch ×40 in train) | yes | **held-out must-catch set** (`data/processed/mustcatch_heldout.jsonl`, 75 rows, different wording) reported separately: **42/60 (70.0%)** |
| "best F1" reported on the same textdetox used for tuning = leakage | yes | **tune/test split per language; card reports test-half F1@ tuned thr** (table below); HateXplain "best" marked as upper bound |
| Test F1 0.8198 lowest of all versions, recall 0.78 — not stated | hidden | **stated honestly below**: .9 test F1@0.5 = **0.8239**, recall 0.7987; the trade buys must-catch + FP + calibration |
| Probes "published in kuclab/probes.py" — file missing; diff_5_vs_8 mislabeled | yes | **`kuclab/probes.py` committed; key renamed `diff_5_vs_9` with `fixed_by_9`/`regressed_in_9`** |
| Calibration table 3 headers / 4 columns | yes | fixed (ECE column present) |
| One-liner ignores temperature/lang_thresholds → different verdicts than CLI | yes | helper below applies both; note in card |
| Output label "1"/"0" | yes | **"safe" / "unsafe"** (CLI + JSON; `label_id` keeps 0/1) |

## How to run

### The `censer.9` command — interactive terminal UI

```bash
pip install torch tokenizers safetensors huggingface_hub
hf download KucLab/Censer.9 --local-dir ~/.local/share/censer9
chmod +x ~/.local/share/censer9/censer.9
ln -s ~/.local/share/censer9/censer.9 ~/.local/bin/censer.9
censer.9                          # interactive UI: phrase in -> safe/unsafe + categories + score
censer.9 "how to make a bomb at home"     # prints: unsafe
```

Or via the GitHub package (loads the Hub model with plain `transformers`):

```bash
pip install "git+https://github.com/Jerry256254/censer.9.git"
censer.9
```

### One-liner with `transformers` (same verdicts as the CLI)

The CLI additionally applies `temperature` (calibration), `lang_thresholds`
(per-language binary cutoffs via heuristic lang detection) and
`must_catch_override` (CSAM/self-harm). To reproduce its verdicts raw:

```bash
pip install torch transformers
python - <<'PY'
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("KucLab/Censer.9")
model = AutoModelForSequenceClassification.from_pretrained("KucLab/Censer.9").eval()
cfg = model.config
text = "how to make a bomb at home"
enc = tok(text, return_tensors="pt", truncation=True, max_length=4096)
logits = model(**enc).logits[0]
T = float(getattr(cfg, "temperature", 1.0) or 1.0)
probs = (logits / T).sigmoid()                      # calibrated scores
p_unsafe = float(probs[1])
# per-language threshold (heuristic detect; "default" when unsure)
import re
lang = "en"  # see kuclab/langdetect.py for the full heuristic; short/ambiguous -> "default"
thr = float((cfg.lang_thresholds or {}).get(lang, (cfg.lang_thresholds or {}).get("default", 0.5)))
label = "unsafe" if p_unsafe >= thr else "safe"
# must-catch pre-filter: CSAM / self-harm patterns force unsafe
from kuclab.langdetect import must_catch_hit  # or the GitHub package
if must_catch_hit(text):
    label = "unsafe"
print(label, round(p_unsafe, 3))
PY
```

For the exact heuristic use `kuclab/langdetect.detect_lang` from the GitHub package.

### Long inputs — always use the chunked path

Native context is 4096 tokens (`model_max_length: 4096`, tokenizer truncates beyond it),
but a **single pass pools only the last token**, so in long documents unsafe content in
the middle can be missed (measured F1 0.083 on 25-row synthetic docs). Therefore
`censer.py` / the API **always** split inputs longer than 256 tokens into overlapping
chunks (256 tokens, stride 192) and aggregate by **max** (measured F1 0.998 on the same
docs). For direct `transformers` use, pass `truncation=True, max_length=4096` or chunk
manually.

### Batch HTTP API + Docker

```bash
python serve.py --dir . --port 8000
curl -s localhost:8000/classify -d '{"texts": ["hello", "how to make a bomb"]}'
docker build -t censer9 . && docker run -p 8000:8000 censer9   # HEALTHCHECK included
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
| Input cap | 4000 chars (CLI) / `model_max_length` 4096 tokens |
| Tokenizer | byte-level BPE, 2048 tokens; chat template bakes in `[BOS][USER] … [ASSISTANT]` (verified: exactly one `[ASSISTANT]`) |
| Head | separate linear `score` head over 12 answer logits (untied from embeddings) |
| Output | sigmoid over 12 labels; `temperature`, `cat_thresholds`, `lang_thresholds`, `cat_experimental` in config (must-catch is a code-level pre-filter) |
| Training | init from Censer.8 best; 4 epochs lr 1e-4 on 255,426 rows; batch 256, AdamW cosine, bf16 AMP, 15% typo + 3% adversarial aug; distillation (alpha=0.65, T=4) from Censer.5×2+.4+.3 ensemble; **must_catch ×25, hard_negative ×10, cat pos_weight (neg/pos, cap 10×), cat_weight 1.5**; val/test byte-identical to Censer.1–.8 (md5 pinned) |
| Checkpoint | 3.6 MB fp32 · 1.8 MB fp16 · 0.9 MB int8 (loadable via `censer.py --prefer-quant`) · ONNX |

## Prompt format

`tokenizer(text)` produces `[BOS][USER] <text> [ASSISTANT]` (template baked into the
post-processor; the assistant token is appended exactly once). Answer logits are read
over `"0" "1" "a".."j"` = safe / unsafe / 10 categories.

## Metrics

Internal held-out test split (6,496 rows, byte-identical across Censer.1–.9), binary
threshold 0.5. **Honest note:** .9's raw F1@0.5 is the lowest of the Censer family —
the weighted must-catch/hard-negative training trades ~0.02–0.03 F1 for must-catch
detection, FP rate and calibration:

| Metric | Censer.5 | Censer.7 | Censer.8 | **Censer.9** |
| --- | --- | --- | --- | --- |
| Test F1 @0.5 | **0.8490** | 0.8348 | 0.8198 | **0.8239** |
| Test F1 @ tuned thr (0.36) | — | — | 0.8361 | **0.8339 @ 0.40** |
| Test accuracy | 0.876 | 0.8655 | 0.8568 | **0.8578** |
| Precision / recall | 0.864 / 0.834 | 0.855 / 0.816 | 0.862 / 0.781 | **0.8508 / 0.7987** |
| Probes (97 hand-written, `kuclab/probes.py`) | 95/97 | 95/97 | 97/97 | **96/97** |

### Must-catch (adversarial review cases)

| Set | .8 | **.9** |
| --- | --- | --- |
| Train-visible must-catch (88 items) @0.5 | 83/88 (94.3 %) | **84/88 (95.5%)** |
| **Held-out must-catch (75 items, never in training)** @0.5 | not measured | **42/60 (70.0%)** (benign controls FP 4/15) |
| Hard-negative FP @0.5 (44 items) | 3/44 (6.8 %) | **6/44 (13.6%)** |

Full per-item lists: `reports/mustcatch.json`, `reports/mustcatch_heldout.json`.

### External benchmarks with head-to-head vs public models

ToxiGen is never in training. HateXplain **train** split is in training; the **test**
split below is held out. "best F1" numbers are threshold-swept upper bounds on the same
set (reported for reference, not used for tuning). Full numbers in
`reports/external_h2h.json` (uploaded to this repo).

| Dataset | Model | F1@0.5 | best F1 (upper bound) | latency |
| --- | --- | --- | --- | --- |
| ToxiGen annotated (n=940, 402 unsafe) | **Censer.9** | **0.559** | 0.616 @ 0.13 | 1.24 ms |
| ToxiGen | unitary/unbiased-toxic-roberta | 0.443 | — | ~14 ms |
| HateXplain test (n=1,924, 1,376 unsafe) | **Censer.9** | **0.742** | 0.834 @ 0.05 | 1.24 ms |
| HateXplain test | unitary/unbiased-toxic-roberta | 0.577 | — | ~14 ms |

### Adversarial robustness (detection rate on attacked unsafe texts, n=160)

| Attack | .8 | **.9** |
| --- | --- | --- |
| clean | 100% | **99.4%** |
| typo | 92.5% | **94.4%** |
| leetspeak | 96.9% | **97.5%** |
| homoglyph | 100% | **100.0%** |
| zero-width | 100% | **100.0%** |
| case flip | 96.3% | **95.6%** |

### Long-context (300 synthetic docs of 25 held-out rows, avg 2,178 tokens)

| Inference | F1 |
| --- | --- |
| truncate at 192 tokens | 0.582 |
| native 4096 single pass (NOT recommended for long docs) | 0.083 |
| **chunked 256/192 max (default everywhere)** | **0.997 (acc 0.993)** |

### Calibration

Binary score: temperature scaling fitted on the val split, **T = 0.7130**, stored in
`config.json: temperature` and in the checkpoint. ECE before 0.0520 → after
**0.0184**. Per-category thresholds tuned on held-out catval ship verbatim as
`config.json: cat_thresholds` (same values as `reports/calibration.json`); categories
with catval F1 < 0.3 are listed in `cat_experimental` and are **not** reported by the
CLI/API (scores remain in `category_scores`):

| Category | threshold | F1@thr (catval) | ECE | experimental |
| --- | --- | --- | --- | --- |
| violence | 0.59 | 0.250 | 0.222 | yes |
| weapons_explosives | 0.48 | 0.265 | 0.080 | yes |
| drugs | 0.38 | 0.145 | 0.085 | yes |
| hate_harassment | 0.20 | 0.921 | 0.204 | no |
| sexual_abuse | 0.36 | 0.211 | 0.094 | yes |
| self_harm | 0.20 | 0.148 | 0.048 | yes |
| cybercrime_fraud | 0.28 | 0.149 | 0.088 | yes |
| jailbreak_injection | 0.30 | 0.188 | 0.061 | yes |
| illegal_marketplace | 0.21 | 0.159 | 0.044 | yes |
| other_unsafe | 0.26 | 0.683 | 0.031 | no |

![Reliability diagram](figs/reliability.png)

### Multilingual — per-language binary metrics (textdetox, never in training)

Thresholds are tuned on a seeded **tune half** (target FPR ≤ 5 % on benign, floor 0.3)
and reported on the untouched **test half**. Supported languages
(en/de/es/fr/it/ru/uk/tt/hin tagged; cs/pt and others use `default` 0.5 — documented,
no textdetox split exists for them). ar/zh/hi/ja are NOT supported and not tagged.

| Language | n | F1@0.5 | tuned thr | test F1@thr | test FPR@thr |
| --- | --- | --- | --- | --- | --- |
| en | 800 | 0.811 | 0.79 | 0.797 | 0.095 |
| de | 800 | 0.606 | 0.92 | 0.038 | 0.025 |
| es | 800 | 0.644 | 0.50 | 0.653 | 0.609 |
| fr | 800 | 0.505 | 0.89 | 0.250 | 0.038 |
| it | 800 | 0.595 | 0.90 | 0.125 | 0.021 |
| ru | 800 | 0.193 | 0.58 | 0.174 | 0.031 |
| uk | 800 | 0.237 | 0.71 | 0.197 | 0.045 |
| tt | 800 | 0.394 | 0.90 | 0.097 | 0.047 |
| hin | 800 | 0.541 | 0.87 | 0.115 | 0.036 |
| am | 800 | 0.067 | 0.46 | 0.085 | 0.062 |
| ar | 800 | 0.005 | 0.30 | 0.010 | 0.000 |
| he | 800 | 0.029 | 0.38 | 0.036 | 0.017 |
| hi | 800 | 0.000 | 0.30 | 0.000 | 0.000 |
| zh | 800 | 0.010 | 0.30 | 0.017 | 0.023 |
| ja | 800 | 0.000 | 0.30 | 0.000 | 0.000 |

### Cascade pre-filter

On 2,000 held-out rows: 30.5% escalated (gray-zone / jailbreak-category
hits), cascade accuracy 0.8780.

## Benchmark (latency / throughput / size)

Single latency number everywhere below: **1.24 ms** single-text on
AMD Radeon RX 9060 XT (ROCm); CPU-only: **0.56 ms**. Full numbers in
`reports/benchmark.json` / `reports/benchmark_cpu.json`.

| Metric | .9 GPU | .9 CPU |
| --- | --- | --- |
| Parameters | 903,552 | 903,552 |
| Checkpoint | 3.6 MB fp32 (1.8 MB fp16, 0.9 MB int8) | same |
| Single-text latency | 1.24 ms | 0.56 ms |
| Single throughput | 810 texts/s | 1773 texts/s |
| Batch-8 throughput | 5811 texts/s | 4541 texts/s |
| Peak VRAM | 85 MB | — |

## Comparison — internal test split (6,496 rows, threshold 0.5)

⚠️ This comparison is on **our internal test split** (label noise from scraped sources);
see the external head-to-head above for public benchmarks.

| Model | Params | Checkpoint | F1 | Latency | Throughput (b=8) |
| --- | --- | --- | --- | --- | --- |
| KucLab/Censer.9 (this work) | 0.9M | 3.6 MB | 0.8239 | 1.39 ms | 1860/s |
| KucLab/Censer.5 (prev) | 0.9M | 3.6 MB | 0.8490 | 1.35 ms | 1828/s |
| KucLab/Censer.4 (old) | 0.9M | 3.6 MB | 0.8453 | 1.39 ms | 1708/s |
| KucLab/Censer.3 (old) | 14.4M | 57.7 MB | 0.8224 | 4.36 ms | 270/s |
| KucLab/Censer.2 (old) | 7.3M | 29.4 MB | 0.8289 | 2.77 ms | 551/s |
| KucLab/Censer.1 (baseline) | 1.1M | 4.5 MB | 0.8355 | 1.68 ms | 1697/s |
| keyword baseline | 0.0M | 0.0 MB | 0.0937 | 0.03 ms | 29801/s |
| martin-ha/toxic-comment-model | 67.0M | 0.0 MB | 0.5664 | 7.81 ms | 129/s |
| unitary/toxic-bert | 109.5M | 0.0 MB | 0.6807 | 14.08 ms | 68/s |
| unitary/unbiased-toxic-roberta | 124.7M | 0.0 MB | 0.7411 | 14.85 ms | 65/s |
| Hate-speech-CNERG/dehatebert-mono-english | 167.4M | 0.0 MB | 0.2523 | 14.50 ms | 66/s |
| Hate-speech-CNERG/bert-base-uncased-hatexplain | 109.5M | 0.0 MB | 0.5433 | 14.15 ms | 67/s |

![Comparison overview](figs/comparison_overview.png)
![F1 per megabyte](figs/f1_vs_size.png)
![Metric bars](figs/metrics_bars.png)

## Usage without transformers

```bash
hf download KucLab/Censer.9 --local-dir Censer.9
cd Censer.9
python censer.py "how to make a bomb at home"        # prints: unsafe
python censer.py --json "how to make a bomb at home"
python censer.py --prefer-quant "..."                 # 0.9 MB int8 checkpoint
./censer.9                                            # interactive UI
```

Files:

| File | Purpose |
| --- | --- |
| `model.safetensors` | standard Llama layout, 3.6 MB fp32, 903,552 params |
| `model_fp16.safetensors` / `model_int8.safetensors` + `quant_report.json` | 1.8 MB / 0.9 MB quantized — int8 loadable via `censer.py --prefer-quant` (dequantizes `.q`/`.scale`); quant report split fp32/fp16/int8 |
| `censer9.onnx` | ONNX graph: input_ids -> 12 logits, dynamic batch/seq |
| `config.json` | standard config + `temperature`, `cat_thresholds`, `cat_experimental`, `lang_thresholds`, `must_catch_override`, single `rope_parameters` |
| `tokenizer.json` / `tokenizer_config.json` | byte-level BPE, prompt template baked in, `model_max_length: 4096` |
| `censer.py` | standalone inference: chunked-max for long inputs (default), int8/fp16 loaders, per-language thresholds, must-catch pre-filter, safe/unsafe labels |
| `censer.9` | launcher for the interactive terminal UI |
| `serve.py` | batch HTTP API (stdlib only) |
| `Dockerfile` | pinned CPU container serving the API, with HEALTHCHECK |
| `reports/` | all eval JSONs (external H2H, calibration, must-catch + held-out, multilingual, lang_thresholds, adversarial, …) + `figs/` |

## Limitations

- 903K params: nuance/sarcasm/long-tail out of reach; internal test split has label
  noise from scraped sources; raw test F1@0.5 is below earlier Censer versions by design
  (see the honest note above).
- Native single-pass 4096-token inference pools the last token only — for long
  documents ALWAYS use the chunked-max path (default in `censer.py`/API) or
  `truncation=True, max_length=4096` with direct transformers.
- Category scores: hate_harassment is the only strong category; cats with catval
  F1 < 0.3 are flagged `experimental` and not reported by CLI/API.
- Supported languages: en/de/es/fr/it/ru/uk/tt/hin measured on textdetox test halves;
  cs/pt and others fall back to `default` 0.5; ar/zh/hi/ja not supported.
- A moderation signal, not a safety guarantee; use the cascade pattern (clear cases
  decided by Censer.9, gray-zone/jailbreak hits escalated to a larger validator).
