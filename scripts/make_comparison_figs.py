"""Render comparison charts from reports/comparison.json into reports/figs/.

    .venv/bin/python scripts/make_comparison_figs.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA = REPO_ROOT / "reports" / "comparison.json"
FIGS = REPO_ROOT / "reports" / "figs"

OURS = "Censer.9"


def short_name(repo: str) -> str:
    if "Censer.9" in repo:
        return "Censer.9 (ours)"
    if "Censer.2" in repo:
        return "Censer.2 (prev)"
    if "Censer.1" in repo:
        return "Censer.1 (old)"
    if repo == "keyword baseline":
        return "keyword baseline"
    return repo.split("/")[-1]


def load():
    data = json.loads(DATA.read_text())
    models = data["models"]
    for m in models:
        m["name"] = short_name(m["repo"])
        m["is_ours"] = "Censer.9" in m["repo"]
        m["is_base"] = ("Censer.1" in m["repo"]) or ("Censer.2" in m["repo"])
    models.sort(key=lambda m: m["params"])
    return data, models


def style_ax(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_xscale("log")


def plot_vs_params(ax, models, key, ylabel, log_y=False):
    for m in models:
        if m["params"] <= 0:
            continue
        color = "#d62728" if m["is_ours"] else ("#ff7f0e" if m.get("is_base") else "#1f77b4")
        marker = "*" if m["is_ours"] else ("D" if m.get("is_base") else "o")
        size = 160 if m["is_ours"] else 60
        ax.scatter(m["params"], m[key], s=size, c=color, marker=marker, zorder=3,
                   edgecolors="black", linewidths=0.5)
        dy = 1.12 if key in ("latency_ms",) else 1.06
        ax.annotate(m["name"], (m["params"], m[key] * dy), fontsize=8, ha="center")
    ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.set_xlabel("parameters")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linewidth=0.6)


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    data, models = load()
    gpu = data.get("gpu", "")

    # --- overview 2x2 -------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Censer.9 vs other public safety classifiers — same test split (n={data['n_test']}, {gpu})",
                 fontsize=12, fontweight="bold")
    # F1 vs params
    for m in models:
        if m["params"] <= 0:
            continue
        c = "#d62728" if m["is_ours"] else ("#ff7f0e" if m.get("is_base") else "#1f77b4")
        mk = "*" if m["is_ours"] else ("D" if m.get("is_base") else "o")
        s = 170 if m["is_ours"] else 60
        axes[0, 0].scatter(m["params"], m["metrics"]["f1"], s=s, c=c, marker=mk, zorder=3,
                          edgecolors="black", linewidths=0.5)
        axes[0, 0].annotate(m["name"], (m["params"], m["metrics"]["f1"] * 1.04), fontsize=8, ha="center")
    axes[0, 0].set_xscale("log")
    axes[0, 0].set_xlabel("parameters")
    axes[0, 0].set_ylabel("F1 (unsafe vs safe)")
    axes[0, 0].set_title("Quality vs model size")
    axes[0, 0].grid(True, alpha=0.3, linewidth=0.6)
    axes[0, 0].set_ylim(0, 1.05)

    plot_vs_params(axes[0, 1], models, "latency_ms", "single-text latency [ms]", log_y=True)
    axes[0, 1].set_title("Latency vs model size (log-log, lower is better)")

    plot_vs_params(axes[1, 0], models, "throughput_s", "batched throughput [texts/s]", log_y=True)
    axes[1, 0].set_title("Throughput vs model size (higher is better)")

    plot_vs_params(axes[1, 1], models, "vram_mb", "peak VRAM [MB]", log_y=True)
    axes[1, 1].set_title("Memory vs model size")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(FIGS / "comparison_overview.png", dpi=150)
    plt.close(fig)

    # --- F1 vs checkpoint size (efficiency frontier) -------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for m in models:
        x = max(m["checkpoint_mb"], 0.001)
        c = "#d62728" if m["is_ours"] else ("#ff7f0e" if m.get("is_base") else "#1f77b4")
        mk = "*" if m["is_ours"] else ("D" if m.get("is_base") else "o")
        s = 40 + 400 * (m["params"] / 130e6) ** 0.5 if m["params"] else 40
        ax.scatter(x, m["metrics"]["f1"], s=s, c=c, marker=mk, zorder=3,
                   edgecolors="black", linewidths=0.5, alpha=0.9)
        ax.annotate(f"{m['name']}\n{m['params']/1e6:.1f}M params" if m["params"] else m["name"],
                    (x, m["metrics"]["f1"]), textcoords="offset points", xytext=(0, 12),
                    fontsize=8, ha="center")
    ax.set_xscale("log")
    ax.set_xlabel("checkpoint size [MB] (log)")
    ax.set_ylabel("F1 (unsafe vs safe)")
    ax.set_title(f"Quality per megabyte — same test split (n={data['n_test']})")
    ax.grid(True, alpha=0.3, linewidth=0.6)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(FIGS / "f1_vs_size.png", dpi=150)
    plt.close(fig)

    # --- grouped metric bars ---------------------------------------------------
    bar_models = sorted(models, key=lambda m: -m["metrics"]["f1"])
    names = [m["name"] for m in bar_models]
    keys = ["accuracy", "precision", "recall", "f1"]
    fig, ax = plt.subplots(figsize=(max(9, 1.6 * len(names)), 5))
    width = 0.2
    for i, key in enumerate(keys):
        vals = [m["metrics"][key] for m in bar_models]
        bars = ax.bar([x + i * width for x in range(len(names))], vals, width=width, label=key)
        for b, m in zip(bars, bar_models):
            if m["is_ours"] or m.get("is_base"):
                b.set_edgecolor("#d62728" if m["is_ours"] else "#ff7f0e")
                b.set_linewidth(2.5)
    ax.set_xticks([x + 1.5 * width for x in range(len(names))])
    ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Metrics on the held-out test split (n={data['n_test']}) — red outline = Censer.9")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.6)
    fig.tight_layout()
    fig.savefig(FIGS / "metrics_bars.png", dpi=150)
    plt.close(fig)

    # --- markdown report ------------------------------------------------------
    lines = [
        "# Comparison with other public safety classifiers",
        "",
        f"Same held-out test split for every model: `data/processed/test.jsonl` "
        f"(n={data['n_test']}), threshold 0.5. Hardware: {data.get('gpu', 'GPU')}.",
        "",
        "| Model | Params | Checkpoint | Accuracy | Precision | Recall | F1 | Latency | Throughput (b=32) | Peak VRAM |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for m in sorted(models, key=lambda x: -x["metrics"]["f1"]):
        mt = m["metrics"]
        lines.append(
            f"| {m['name']} | {m['params']/1e6:.1f}M | {m['checkpoint_mb']:.1f} MB | "
            f"{mt['accuracy']:.3f} | {mt['precision']:.3f} | {mt['recall']:.3f} | **{mt['f1']:.3f}** | "
            f"{m['latency_ms']:.2f} ms | {m['throughput_s']:.0f} texts/s | {m['vram_mb']:.0f} MB |"
        )
    lines += [
        "",
        "![overview](figs/comparison_overview.png)",
        "![f1 vs size](figs/f1_vs_size.png)",
        "![metrics](figs/metrics_bars.png)",
        "",
        "Reproduce: `.venv/bin/python scripts/compare_models.py && .venv/bin/python scripts/make_comparison_figs.py`",
        "",
    ]
    (REPO_ROOT / "reports" / "comparison.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {FIGS}/comparison_overview.png, f1_vs_size.png, metrics_bars.png, reports/comparison.md")


if __name__ == "__main__":
    main()
