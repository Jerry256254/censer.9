# Comparison with other public safety classifiers

Same held-out test split for every model: `data/processed/test.jsonl` (n=6496), threshold 0.5. Hardware: AMD Radeon Graphics.

| Model | Params | Checkpoint | Accuracy | Precision | Recall | F1 | Latency | Throughput (b=32) | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Censer.5 (prev) | 0.9M | 3.6 MB | 0.876 | 0.864 | 0.834 | **0.849** | 1.29 ms | 1875 texts/s | 97 MB |
| Censer.4 (old) | 0.9M | 3.6 MB | 0.874 | 0.868 | 0.824 | **0.845** | 1.32 ms | 1818 texts/s | 95 MB |
| Censer.1 (old) | 1.1M | 4.5 MB | 0.865 | 0.848 | 0.823 | **0.835** | 1.72 ms | 1682 texts/s | 97 MB |
| Censer.2 (prev) | 7.3M | 29.4 MB | 0.863 | 0.861 | 0.799 | **0.829** | 2.72 ms | 549 texts/s | 171 MB |
| Censer.3 (old) | 14.4M | 57.7 MB | 0.857 | 0.855 | 0.792 | **0.822** | 4.49 ms | 272 texts/s | 240 MB |
| Censer.8 (ours) | 0.9M | 3.6 MB | 0.857 | 0.862 | 0.781 | **0.820** | 1.37 ms | 1843 texts/s | 105 MB |
| unbiased-toxic-roberta | 124.7M | 0.0 MB | 0.822 | 0.940 | 0.612 | **0.741** | 14.81 ms | 65 texts/s | 771 MB |
| toxic-bert | 109.5M | 0.0 MB | 0.794 | 0.965 | 0.526 | **0.681** | 14.04 ms | 68 texts/s | 708 MB |
| toxic-comment-model | 67.0M | 0.0 MB | 0.744 | 0.961 | 0.402 | **0.566** | 7.79 ms | 129 texts/s | 540 MB |
| bert-base-uncased-hatexplain | 109.5M | 0.0 MB | 0.704 | 0.761 | 0.423 | **0.543** | 14.22 ms | 68 texts/s | 692 MB |
| dehatebert-mono-english | 167.4M | 0.0 MB | 0.636 | 0.875 | 0.147 | **0.252** | 14.52 ms | 66 texts/s | 923 MB |
| keyword baseline | 0.0M | 0.0 MB | 0.568 | 0.374 | 0.054 | **0.094** | 0.03 ms | 29866 texts/s | 0 MB |

![overview](figs/comparison_overview.png)
![f1 vs size](figs/f1_vs_size.png)
![metrics](figs/metrics_bars.png)

Reproduce: `.venv/bin/python scripts/compare_models.py && .venv/bin/python scripts/make_comparison_figs.py`
