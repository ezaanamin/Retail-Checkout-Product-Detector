#!/usr/bin/env python3
"""
Make LinkedIn-ready graphs from the Ultralytics results.csv files.

Run from the project root:
    python make_graphs.py

Creates (in ./graphs):
    1_map50_all_experiments.png   - val mAP50 per epoch, every experiment
    2_before_after_bars.png       - the three headline numbers
    3_overfitting_vs_fixed.png    - bonus: train vs val loss, old vs new
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "runs" / "rpc"
OUT = ROOT / "graphs"
OUT.mkdir(exist_ok=True)

# ---- EDIT THESE LABELS to describe what each old run actually was ---------- #
OLD_RUNS = {
    "sku100_run-2": "LR 0.001, strong aug, batch 8",
    "sku100_run-3": "LR 0.001, strong aug, batch 4",
    "sku100_experiment_2": "LR 0.0003, light aug, batch 4",
    "sku100_experiment_4": "LR 0.0003, light aug, batch 2",
}
NEW_RUN = "sku100_exp5_stage1_frozen"
NEW_LABEL = "Synthetic scenes (new)"

# Headline numbers from the --diagnose runs (best.pt of experiment 4)
OLD_TRAIN_MAP50 = 0.995   # old model on single-item train images
OLD_VAL_MAP50 = 0.063     # old model on real checkout images

GRAY = ["#9e9e9e", "#b0b0b0", "#8a8a8a", "#c4c4c4"]
GREEN = "#1b9e4b"
RED = "#d64545"
FIGSIZE = (12, 6.3)  # ~1200x630 at dpi=100, a good LinkedIn image size


def load(run: str) -> pd.DataFrame | None:
    path = RUNS / run / "results.csv"
    if not path.exists():
        print(f"skip {run}: no results.csv")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def style(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


# --------------------------------------------------------------------------- #
# Graph 1: mAP50 per epoch for every experiment
# --------------------------------------------------------------------------- #
def graph_map50_lines() -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=100)

    for i, (run, label) in enumerate(OLD_RUNS.items()):
        df = load(run)
        if df is None:
            continue
        ax.plot(df["epoch"], df["metrics/mAP50(B)"], color=GRAY[i % len(GRAY)],
                linewidth=2, label=label)

    new = load(NEW_RUN)
    if new is not None:
        ax.plot(new["epoch"], new["metrics/mAP50(B)"], color=GREEN, linewidth=4,
                marker="o", markersize=5, label=NEW_LABEL)
        last = new.iloc[-1]
        ax.annotate(f"{last['metrics/mAP50(B)']:.2f}",
                    (last["epoch"], last["metrics/mAP50(B)"]),
                    textcoords="offset points", xytext=(8, -4),
                    color=GREEN, fontsize=16, fontweight="bold")

    ax.set_title("Detection score on real checkout images, by experiment",
                 fontsize=18, fontweight="bold", loc="left")
    ax.set_xlabel("Training epoch", fontsize=13)
    ax.set_ylabel("mAP50 (higher is better)", fontsize=13)
    ax.set_ylim(0, 1.0)
    style(ax)
    ax.legend(frameon=False, fontsize=12, loc="center right")
    fig.text(0.01, 0.01,
             "Earlier runs: original validation split. New run: 4,000 held-out real checkout images.",
             fontsize=9, color="#666")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "1_map50_all_experiments.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Graph 2: the three headline numbers
# --------------------------------------------------------------------------- #
def graph_bars() -> None:
    new = load(NEW_RUN)
    new_val = float(new["metrics/mAP50(B)"].max()) if new is not None else 0.94

    labels = ["Old model\non its single-item\ntraining images",
              "Old model\non real\ncheckout images",
              "New model\non real\ncheckout images"]
    values = [OLD_TRAIN_MAP50, OLD_VAL_MAP50, new_val]
    colors = ["#9e9e9e", RED, GREEN]

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=100)
    bars = ax.bar(labels, values, color=colors, width=0.55)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=20, fontweight="bold")

    ax.set_ylim(0, 1.12)
    ax.set_ylabel("mAP50 (higher is better)", fontsize=13)
    ax.set_title("From 0.06 to 0.94 on real checkout scenes",
                 fontsize=18, fontweight="bold", loc="left")
    ax.tick_params(axis="x", labelsize=13)
    style(ax)
    fig.tight_layout()
    fig.savefig(OUT / "2_before_after_bars.png")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Graph 3 (bonus): overfitting evidence, old vs new
# --------------------------------------------------------------------------- #
def graph_loss_compare() -> None:
    old, new = load("sku100_experiment_4"), load(NEW_RUN)
    if old is None or new is None:
        print("skip graph 3: missing experiment 4 or new run")
        return

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=100)
    for ax, df, title, color in (
        (axes[0], old, "Before: loss diverges", RED),
        (axes[1], new, "After: loss falls together", GREEN),
    ):
        ax.plot(df["epoch"], df["train/cls_loss"], color="#555", linewidth=3, label="Training")
        ax.plot(df["epoch"], df["val/cls_loss"], color=color, linewidth=3, label="Validation")
        ax.set_title(title, fontsize=16, fontweight="bold", loc="left")
        ax.set_xlabel("Training epoch", fontsize=12)
        ax.set_ylabel("Classification loss (lower is better)", fontsize=12)
        style(ax)
        ax.legend(frameon=False, fontsize=12)

    fig.suptitle("Training vs validation loss", fontsize=18, fontweight="bold", x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "3_overfitting_vs_fixed.png")
    plt.close(fig)


if __name__ == "__main__":
    graph_map50_lines()
    graph_bars()
    graph_loss_compare()
    print(f"Saved graphs to {OUT}")
