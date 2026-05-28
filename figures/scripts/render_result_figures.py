from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COLORS = {
    "vit": "#8C96A8",
    "plucker": "#4E79A7",
    "prope": "#E15759",
    "action": "#159D8C",
    "nvs": "#E88412",
    "dp": "#6B7280",
    "maniflow": "#7B66FF",
    "grid": "#D9E0EA",
    "text": "#1F2937",
}


def setup() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 14,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.55,
            "grid.color": COLORS["grid"],
            "legend.frameon": False,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def read_csv(name: str) -> list[dict[str, str]]:
    with (DATA / name).open() as f:
        return list(csv.DictReader(f))


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(ROOT / f"{name}.png", dpi=450)
    fig.savefig(ROOT / f"{name}.svg")
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 1e-6 else f"{value:.1f}"


def render_libero() -> None:
    rows = read_csv("libero_mv_spatial.csv")
    baselines = [r for r in rows if r["plot"] == "baseline"]
    sweep = [r for r in rows if r["plot"] == "sweep"]

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(9.4, 4.2),
        gridspec_kw={"width_ratios": [0.92, 1.18]},
    )
    fig.patch.set_facecolor("white")

    labels = [r["method"] for r in baselines]
    values = [float(r["success"]) * 100 for r in baselines]
    colors = [COLORS["vit"], COLORS["plucker"], COLORS["prope"], COLORS["action"]]
    y = np.arange(len(labels))
    ax0.barh(y, values, color=colors, height=0.23, alpha=0.26)
    ax0.scatter(values, y, s=150, color=colors, zorder=3)
    for yi, v in zip(y, values):
        ax0.text(v + 0.8, yi, f"{v:.1f}", va="center", ha="left", fontsize=13, color=COLORS["text"])
    ax0.set_yticks(y)
    ax0.set_yticklabels(["ViT+CamID", "Plucker-GT", "PRoPE-GT", "E2E camera-pose\naction policy"], fontsize=13)
    ax0.invert_yaxis()
    ax0.set_xlim(50, 90)
    ax0.set_xlabel("Spatial success (%)")
    ax0.set_title("Spatial comparison", pad=12)
    ax0.grid(axis="x")
    ax0.grid(axis="y", visible=False)

    ratios = np.array([float(r["ratio"]) for r in sweep])
    sr = np.array([float(r["success"]) * 100 for r in sweep])
    ax1.plot(ratios, sr, color=COLORS["action"], linewidth=4.0, marker="o", markersize=9)
    ax1.axhline(84.2, color=COLORS["prope"], linestyle="--", linewidth=2.0, alpha=0.42)
    ax1.scatter([10], [84.2], marker="*", s=190, color=COLORS["prope"], zorder=4)
    ax1.annotate(
        "best E2E action\n86.5%",
        xy=(50, 86.5),
        xytext=(60, 87.8),
        arrowprops=dict(arrowstyle="-", color=COLORS["action"], lw=2.2),
        fontsize=13,
        color=COLORS["action"],
        ha="left",
    )
    ax1.text(12.5, 84.35, "PRoPE-GT 84.2%", color=COLORS["prope"], fontsize=12, va="bottom")
    ax1.set_xlim(5, 105)
    ax1.set_ylim(74, 90)
    ax1.set_xticks([10, 30, 50, 80, 100])
    ax1.set_yticks([75, 80, 85, 90])
    ax1.set_xlabel("GT pose labels used in training (%)")
    ax1.set_ylabel("Average success (%)")
    ax1.set_title("Pose-label ratio sweep", pad=12)
    ax1.grid(True)

    fig.suptitle("LIBERO-MV Spatial: E2E camera-pose action policy", fontsize=21, y=1.03, color=COLORS["text"])
    fig.tight_layout(w_pad=3.0)
    save(fig, "libero_mv_spatial_summary")


def render_mimicgen() -> None:
    rows = read_csv("mimicgen_multitask_results.csv")
    tasks = ["Stack Three", "Coffee", "Threading", "Stack"]
    methods = [
        ("Diffusion Policy", "Diffusion Policy\n(image)", COLORS["dp"]),
        ("ManiFlow", "ManiFlow\n(point cloud)", COLORS["maniflow"]),
        ("E2E camera-pose action policy", "E2E camera-pose\naction policy", COLORS["action"]),
        ("E2E camera-pose NVS policy", "E2E camera-pose\nNVS policy", COLORS["nvs"]),
    ]
    by_key = {(r["method"], r["task"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=True)
    fig.patch.set_facecolor("white")
    width = 0.18
    x = np.arange(len(tasks))
    offsets = np.linspace(-1.5 * width, 1.5 * width, len(methods))

    for ax, metric, title in zip(axes, ["train_success", "tight_success"], ["Train-camera evaluation", "Tight-camera evaluation"]):
        for mi, (method, label, color) in enumerate(methods):
            vals = []
            for task in tasks:
                raw = by_key[(method, task)][metric]
                vals.append(float(raw) * 100 if raw else np.nan)
            xpos = x + offsets[mi]
            bars = ax.bar(xpos, vals, width=width, label=label, color=color, alpha=0.88)
            for b, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                if v >= 8:
                    if v >= 90:
                        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, fmt_pct(v), ha="center", va="bottom", fontsize=8.5, rotation=90)
                    else:
                        ax.text(b.get_x() + b.get_width() / 2, v + 1.6, fmt_pct(v), ha="center", va="bottom", fontsize=9)
        ax.set_title(title, pad=10)
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=16, ha="right")
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Episode success rate (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.08), fontsize=10.5, columnspacing=1.2, handlelength=1.6)
    fig.suptitle("MimicGen 100-demo camera-shift evaluation", fontsize=21, y=1.22, color=COLORS["text"])
    fig.tight_layout(w_pad=2.4)
    save(fig, "mimicgen_multitask_results")


def render_realworld() -> None:
    rows = read_csv("realworld_blocks_bowl.csv")
    labels = [
        f"Camera Rays\nPlucker-GT\n(n={rows[0]['trials']})",
        f"Ours: E2E camera-pose\nNVS policy\n(n={rows[1]['trials']})",
    ]
    success = [float(rows[0]["success"]) * 100, float(rows[1]["success"]) * 100]
    progress = [float(rows[0]["progress"]) * 100 if rows[0]["progress"] else np.nan, np.nan]
    colors = [COLORS["plucker"], COLORS["nvs"]]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    fig.patch.set_facecolor("white")
    x = np.arange(2)
    bars = ax.bar(x, success, width=0.52, color=colors, alpha=0.88)
    for i, (b, s) in enumerate(zip(bars, success)):
        ax.text(b.get_x() + b.get_width() / 2, s + 3, f"{s:.0f}%", ha="center", va="bottom", fontsize=14, fontweight="bold", color=COLORS["text"])
        if not np.isnan(progress[i]) and progress[i] > s:
            ax.bar(x[i], progress[i] - s, width=0.52, bottom=s, fill=False, edgecolor=colors[i], linewidth=2.4)
            ax.text(x[i], progress[i] + 3, f"progress {progress[i]:.0f}%", ha="center", va="bottom", fontsize=10.5, color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=12)
    ax.set_ylabel("Full-task success rate (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("Real-world Blocks-in-Bowl", pad=12)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    save(fig, "realworld_blocks_bowl_comparison")


def main() -> None:
    setup()
    render_libero()
    render_mimicgen()
    render_realworld()


if __name__ == "__main__":
    main()
