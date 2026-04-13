"""Analyze and visualize evaluation results from a completed eval run."""

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
from robocasa.utils.dataset_registry_utils import get_task_horizon

CATEGORY_COLORS = {
    "atomic_seen": "#2196F3",
    "composite_seen": "#4CAF50",
    "composite_unseen": "#FF9800",
}

CATEGORY_ORDER = ["atomic_seen", "composite_seen", "composite_unseen"]


def build_task_to_category() -> dict[str, str]:
    mapping = {}
    for cat in CATEGORY_ORDER:
        for task in TASK_SET_REGISTRY[cat]:
            mapping[task] = cat
    return mapping


def load_results(evals_dir: Path) -> pd.DataFrame:
    task_to_cat = build_task_to_category()
    rows = []
    for task_dir in sorted(evals_dir.iterdir()):
        stats_file = task_dir / "stats.json"
        if not stats_file.exists():
            continue
        with open(stats_file) as f:
            data = json.load(f)
        task = task_dir.name
        rows.append(
            {
                "task": task,
                "success_rate": data["success_rate"] * 100,
                "category": task_to_cat.get(task, "unknown"),
                "horizon": get_task_horizon(task),
            }
        )
    return pd.DataFrame(rows)


def fig_ranked_success_rates(df: pd.DataFrame, out_dir: Path):
    df_sorted = df.sort_values("success_rate", ascending=True)
    fig, ax = plt.subplots(figsize=(12, 14))
    colors = [CATEGORY_COLORS[c] for c in df_sorted["category"]]
    ax.barh(range(len(df_sorted)), df_sorted["success_rate"], color=colors)
    ax.set_yticks(range(len(df_sorted)))
    ax.set_yticklabels(df_sorted["task"], fontsize=8)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title("All Tasks Ranked by Success Rate")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORY_COLORS[c]) for c in CATEGORY_ORDER
    ]
    ax.legend(handles, CATEGORY_ORDER, loc="lower right")

    mean_sr = df["success_rate"].mean()
    median_sr = df["success_rate"].median()
    n_zero = (df["success_rate"] == 0).sum()
    annotation = (
        f"Mean: {mean_sr:.1f}%  |  Median: {median_sr:.1f}%  |  Tasks at 0%: {n_zero}"
    )
    ax.annotate(
        annotation,
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(out_dir / "ranked_success_rates.png", dpi=150)
    plt.close(fig)


def fig_category_breakdown(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))

    positions = []
    for i, cat in enumerate(CATEGORY_ORDER):
        cat_data = df[df["category"] == cat]["success_rate"]
        ax.boxplot(
            cat_data,
            positions=[i],
            widths=0.5,
            patch_artist=True,
            boxprops=dict(facecolor=CATEGORY_COLORS[cat], alpha=0.4),
            medianprops=dict(color="black"),
        )
        jitter = np.random.default_rng(42).uniform(-0.1, 0.1, len(cat_data))
        ax.scatter(
            [i] * len(cat_data) + jitter,
            cat_data,
            color=CATEGORY_COLORS[cat],
            s=40,
            zorder=3,
            edgecolors="black",
            linewidths=0.5,
        )
        mean_val = cat_data.mean()
        ax.annotate(
            f"μ={mean_val:.1f}%",
            xy=(i, mean_val),
            xytext=(i + 0.35, mean_val),
            fontsize=9,
            fontweight="bold",
        )
        positions.append(i)

    ax.set_xticks(positions)
    ax.set_xticklabels(CATEGORY_ORDER)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate Distribution by Category")
    plt.tight_layout()
    fig.savefig(out_dir / "category_breakdown.png", dpi=150)
    plt.close(fig)


def fig_sr_vs_horizon(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6))
    for cat in CATEGORY_ORDER:
        mask = df["category"] == cat
        ax.scatter(
            df.loc[mask, "horizon"],
            df.loc[mask, "success_rate"],
            color=CATEGORY_COLORS[cat],
            label=cat,
            s=60,
            edgecolors="black",
            linewidths=0.5,
        )

    slope, intercept, r_value, _, _ = stats.linregress(
        df["horizon"], df["success_rate"]
    )
    x_range = np.linspace(df["horizon"].min(), df["horizon"].max(), 100)
    ax.plot(x_range, slope * x_range + intercept, "r--", alpha=0.7, linewidth=2)
    ax.annotate(
        f"R² = {r_value**2:.3f}",
        xy=(0.05, 0.95),
        xycoords="axes fraction",
        fontsize=12,
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
    )

    ax.set_xlabel("Task Horizon")
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("Success Rate vs. Task Horizon")
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / "sr_vs_horizon.png", dpi=150)
    plt.close(fig)


def fig_performance_distribution(df: pd.DataFrame, out_dir: Path):
    bins = [0, 0.001, 10, 25, 50, 75, 100]
    labels = ["0%", "(0-10%]", "(10-25%]", "(25-50%]", "(50-75%]", "(75-100%]"]
    df_copy = df.copy()
    df_copy["bin"] = pd.cut(
        df_copy["success_rate"], bins=bins, labels=labels, right=True
    )

    counts = df_copy["bin"].value_counts().reindex(labels, fill_value=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(labels)), counts, color="#5C6BC0", edgecolor="black")

    for bar, count in zip(bars, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            str(count),
            ha="center",
            fontweight="bold",
        )

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_xlabel("Success Rate Bin")
    ax.set_ylabel("Number of Tasks")
    ax.set_title("Performance Distribution Across Tasks")

    mean_sr = df["success_rate"].mean()
    median_sr = df["success_rate"].median()
    pct_above_50 = (df["success_rate"] > 50).mean() * 100
    annotation = f"Mean: {mean_sr:.1f}%  |  Median: {median_sr:.1f}%  |  Tasks > 50%: {pct_above_50:.0f}%"
    ax.annotate(
        annotation,
        xy=(0.5, 0.95),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(out_dir / "performance_distribution.png", dpi=150)
    plt.close(fig)


def fig_summary_table(df: pd.DataFrame, out_dir: Path):
    col_labels = ["Category", "Mean", "Median", "Tasks at 0%", "# Tasks"]
    cell_text = []
    cell_colors = []

    for cat in CATEGORY_ORDER:
        cat_data = df[df["category"] == cat]["success_rate"]
        cell_text.append(
            [
                cat,
                f"{cat_data.mean():.1f}%",
                f"{cat_data.median():.1f}%",
                str(int((cat_data == 0).sum())),
                str(len(cat_data)),
            ]
        )
        cell_colors.append([CATEGORY_COLORS[cat] + "22"] * len(col_labels))

    overall = df["success_rate"]
    cell_text.append(
        [
            "Overall",
            f"{overall.mean():.1f}%",
            f"{overall.median():.1f}%",
            str(int((overall == 0).sum())),
            str(len(overall)),
        ]
    )
    cell_colors.append(["#ECF0F1"] * len(col_labels))

    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellColours=cell_colors,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    for col_idx in range(len(col_labels)):
        table[0, col_idx].set_facecolor("#2C3E50")
        table[0, col_idx].set_text_props(color="white", fontweight="bold")

    last_row = len(cell_text)
    for col_idx in range(len(col_labels)):
        table[last_row, col_idx].set_text_props(fontweight="bold")

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CCCCCC")
        cell.set_linewidth(0.5)

    ax.set_title("Summary by Category", fontsize=13, fontweight="bold", pad=14)

    plt.tight_layout()
    fig.savefig(out_dir / "summary_table.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_category_detail(df: pd.DataFrame, category: str, out_dir: Path):
    cat_df = df[df["category"] == category].sort_values("success_rate", ascending=True)
    color = CATEGORY_COLORS[category]

    fig, ax = plt.subplots(figsize=(10, max(6, len(cat_df) * 0.4)))
    ax.barh(range(len(cat_df)), cat_df["success_rate"], color=color)
    ax.set_yticks(range(len(cat_df)))
    ax.set_yticklabels(cat_df["task"], fontsize=9)
    ax.set_xlabel("Success Rate (%)")
    ax.set_title(f"{category} Tasks ({len(cat_df)} tasks)")

    mean_sr = cat_df["success_rate"].mean()
    median_sr = cat_df["success_rate"].median()
    n_zero = (cat_df["success_rate"] == 0).sum()
    annotation = (
        f"Mean: {mean_sr:.1f}%  |  Median: {median_sr:.1f}%  |  Tasks at 0%: {n_zero}"
    )
    ax.annotate(
        annotation,
        xy=(0.5, 0.02),
        xycoords="axes fraction",
        ha="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()
    fig.savefig(out_dir / f"detail_{category}.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Analyze eval results")
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Root experiment directory (contains evals/ subdirectory)",
    )
    args = parser.parse_args()

    experiment_dir = Path(args.experiment_dir)
    evals_dir = experiment_dir / "evals"
    figures_dir = experiment_dir / "figures"

    assert evals_dir.exists(), f"evals/ not found at {evals_dir}"

    if figures_dir.exists():
        shutil.rmtree(figures_dir)
    figures_dir.mkdir(parents=True)

    print(f"Loading results from {evals_dir} ...")
    df = load_results(evals_dir)
    print(f"Loaded {len(df)} tasks")
    print(f"\nOverall success rate: {df['success_rate'].mean():.1f}%")
    for cat in CATEGORY_ORDER:
        cat_df = df[df["category"] == cat]
        print(f"  {cat}: {cat_df['success_rate'].mean():.1f}% ({len(cat_df)} tasks)")

    print(f"\nGenerating figures in {figures_dir} ...")
    fig_ranked_success_rates(df, figures_dir)
    fig_category_breakdown(df, figures_dir)
    fig_sr_vs_horizon(df, figures_dir)
    fig_performance_distribution(df, figures_dir)
    fig_summary_table(df, figures_dir)
    for cat in CATEGORY_ORDER:
        fig_category_detail(df, cat, figures_dir)

    print(f"Done! {len(list(figures_dir.glob('*.png')))} figures saved.")


if __name__ == "__main__":
    main()
