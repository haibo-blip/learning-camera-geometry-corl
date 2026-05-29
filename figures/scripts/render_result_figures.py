from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

COLORS = {
    "image": "#6B7280",
    "geometry": "#3B82C4",
    "pose": "#7C5CC4",
    "action": "#17997F",
    "nvs": "#D97706",
    "gt": "#C73E1D",
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
    svg_path = ROOT / f"{name}.svg"
    fig.savefig(svg_path)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def fmt_pct(value: float) -> str:
    return f"{value:.0f}" if abs(value - round(value)) < 1e-6 else f"{value:.1f}"


def render_libero() -> None:
    rows = read_csv("libero_mv_spatial.csv")
    baselines = [r for r in rows if r["plot"] == "baseline"]
    sweep = [r for r in rows if r["plot"] == "sweep"]

    fig, ax0 = plt.subplots(figsize=(5.2, 3.8))
    fig.patch.set_facecolor("white")

    labels = [r["method"] for r in baselines]
    values = [float(r["success"]) * 100 for r in baselines]
    colors = [COLORS["image"], COLORS["geometry"], COLORS["pose"], COLORS["action"]]
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
    ax0.grid(axis="x")
    ax0.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, "libero_mv_spatial_summary")

    fig, ax1 = plt.subplots(figsize=(5.2, 3.8))
    fig.patch.set_facecolor("white")
    ratios = np.array([float(r["ratio"]) for r in sweep])
    sr = np.array([float(r["success"]) * 100 for r in sweep])
    ax1.plot(ratios, sr, color=COLORS["action"], linewidth=4.0, marker="o", markersize=9)
    ax1.axhline(84.2, color=COLORS["pose"], linestyle="--", linewidth=2.0, alpha=0.42)
    ax1.scatter([10], [84.2], marker="*", s=190, color=COLORS["pose"], zorder=4)
    ax1.annotate(
        "best E2E action\n86.5%",
        xy=(50, 86.5),
        xytext=(60, 87.8),
        arrowprops=dict(arrowstyle="-", color=COLORS["action"], lw=2.2),
        fontsize=13,
        color=COLORS["action"],
        ha="left",
    )
    ax1.text(12.5, 84.35, "PRoPE-GT 84.2%", color=COLORS["pose"], fontsize=12, va="bottom")
    ax1.set_xlim(5, 105)
    ax1.set_ylim(74, 90)
    ax1.set_xticks([10, 30, 50, 80, 100])
    ax1.set_yticks([75, 80, 85, 90])
    ax1.set_xlabel("GT pose labels used in training (%)")
    ax1.set_ylabel("Average success (%)")
    ax1.grid(True)
    fig.tight_layout()
    save(fig, "libero_mv_label_sweep")


def render_mimicgen() -> None:
    rows = read_csv("mimicgen_multitask_results.csv")
    tasks = ["Stack Three", "Coffee", "Threading", "Stack"]
    plot_labels = tasks + ["Mean"]
    methods = [
        ("Diffusion Policy", "Diffusion Policy\n(image)", COLORS["image"], None),
        ("E2E camera-pose action policy", "E2E camera-pose\naction policy", COLORS["action"], None),
        ("E2E camera-pose NVS policy", "E2E camera-pose\nNVS policy", COLORS["nvs"], None),
        ("ManiFlow", "ManiFlow\n(point cloud +\nGT extrinsics)", COLORS["geometry"], "////"),
    ]
    by_key = {(r["method"], r["task"]): r for r in rows}

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.25), sharey=True)
    fig.patch.set_facecolor("white")
    width = 0.16
    x = np.concatenate([np.arange(len(tasks)), np.array([len(tasks) + 0.72])])
    offsets = np.array([-1.75, -0.65, 0.45, 1.95]) * width

    for ax, metric in zip(axes, ["train_success", "tight_success"]):
        ax.axvspan(x[-1] - 0.52, x[-1] + 0.52, color="#F3F6FA", zorder=0)
        ax.axvline((x[-2] + x[-1]) / 2, color=COLORS["grid"], linestyle="--", linewidth=1.4, zorder=1)
        for mi, (method, label, color, hatch) in enumerate(methods):
            task_vals = []
            for task in tasks:
                raw = by_key[(method, task)][metric]
                task_vals.append(float(raw) * 100 if raw else np.nan)
            vals = task_vals + [float(np.nanmean(task_vals))]
            xpos = x + offsets[mi]
            bars = ax.bar(
                xpos,
                vals,
                width=width,
                label=label,
                color=color,
                alpha=0.88 if hatch is None else 0.76,
                edgecolor="white" if hatch is None else COLORS["text"],
                linewidth=0.5 if hatch is None else 1.1,
                hatch=hatch,
            )
            bars[-1].set_edgecolor(COLORS["text"])
            bars[-1].set_linewidth(0.8)
            for bi, (b, v) in enumerate(zip(bars, vals)):
                if np.isnan(v):
                    continue
                is_mean = bi == len(vals) - 1
                label_x = b.get_x() + b.get_width() / 2
                label_y = max(v + 1.3, 2.0)
                rotate = v < 8 or v >= 90 or is_mean
                ax.text(
                    label_x,
                    label_y,
                    fmt_pct(v),
                    ha="center",
                    va="bottom",
                    fontsize=8.2 if rotate else 8.8,
                    rotation=90 if rotate else 0,
                    color=COLORS["text"],
                    clip_on=False,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(plot_labels, rotation=16, ha="right")
        ax.set_ylim(0, 116)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Episode success rate (%)")
    fig.tight_layout(rect=(0, 0, 1, 0.87), w_pad=2.4)

    legend_ax = fig.add_axes([0.045, 0.885, 0.91, 0.105])
    legend_ax.set_axis_off()
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_items = [
        (0.02, COLORS["image"], None, "Diffusion Policy", "(image)", COLORS["text"], "normal"),
        (0.25, COLORS["action"], None, "E2E camera-pose", "action policy", COLORS["text"], "normal"),
        (0.49, COLORS["nvs"], None, "E2E camera-pose", "NVS policy", COLORS["text"], "normal"),
        (0.73, COLORS["geometry"], "////", "ManiFlow (point cloud)", "GT extrinsics", COLORS["gt"], "bold"),
    ]
    for x0, color, hatch, line1, line2, line2_color, line2_weight in legend_items:
        legend_ax.add_patch(
            Rectangle(
                (x0, 0.37),
                0.027,
                0.26,
                facecolor=color,
                edgecolor=COLORS["text"] if hatch else "white",
                linewidth=0.9 if hatch else 0.6,
                hatch=hatch,
                alpha=0.76 if hatch else 0.88,
            )
        )
        legend_ax.text(x0 + 0.038, 0.64, line1, ha="left", va="center", fontsize=9.7, color=COLORS["text"])
        legend_ax.text(
            x0 + 0.038,
            0.30,
            line2,
            ha="left",
            va="center",
            fontsize=9.7,
            color=line2_color,
            fontweight=line2_weight,
        )
    save(fig, "mimicgen_multitask_results")


def render_mimicgen_ablations() -> None:
    rows = read_csv("mimicgen_stack_three_ablations.csv")
    panels = [
        (
            "objective",
            "Objective",
            ["No E2E pose or NVS", "E2E pose only", "E2E pose + NVS"],
            ["No E2E pose\nor NVS", "E2E pose\nonly", "E2E pose\n+ NVS"],
        ),
        (
            "scene_token",
            "Scene tokens",
            ["NVS without scene tokens", "NVS with scene tokens"],
            ["NVS w/o\nscene tokens", "NVS w/\nscene tokens"],
        ),
    ]
    by_key = {(r["ablation"], r["variant"]): r for r in rows}
    evals = [
        ("train_success", "Train-camera", COLORS["geometry"]),
        ("tight_success", "Tight-camera", COLORS["nvs"]),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.05), sharey=True, gridspec_kw={"width_ratios": [1.25, 0.95]})
    fig.patch.set_facecolor("white")

    for ax, (ablation, _title, variants, labels) in zip(axes, panels):
        x = np.arange(len(variants))
        width = 0.30
        for ei, (metric, eval_label, color) in enumerate(evals):
            vals = [float(by_key[(ablation, variant)][metric]) * 100 for variant in variants]
            xpos = x + (ei - 0.5) * width
            bars = ax.bar(xpos, vals, width=width, label=eval_label, color=color, alpha=0.90)
            for b, v in zip(bars, vals):
                ax.text(
                    b.get_x() + b.get_width() / 2,
                    max(v + 1.2, 1.4),
                    fmt_pct(v),
                    ha="center",
                    va="bottom",
                    fontsize=8.8,
                    color=COLORS["text"],
                    clip_on=False,
                )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8.8)
        ax.set_ylim(0, 66)
        ax.set_yticks([0, 25, 50])
        ax.grid(axis="y")
        ax.grid(axis="x", visible=False)

    axes[0].set_ylabel("Success (%)", fontsize=10)
    axes[0].tick_params(axis="y", labelsize=9)
    axes[1].tick_params(axis="y", labelsize=9)
    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, loc="upper left", fontsize=8.6, handlelength=1.3, borderaxespad=0.2)
    fig.tight_layout(w_pad=1.5)
    save(fig, "mimicgen_stack_three_ablations")


def render_realworld() -> None:
    rows = read_csv("realworld_blocks_bowl.csv")
    labels = [
        f"Camera Rays\nPlucker-GT\n(n={rows[0]['trials']})",
        f"Ours: E2E camera-pose\nNVS policy\n(n={rows[1]['trials']})",
    ]
    success = [float(rows[0]["success"]) * 100, float(rows[1]["success"]) * 100]
    progress = [float(rows[0]["progress"]) * 100 if rows[0]["progress"] else np.nan, np.nan]
    colors = [COLORS["geometry"], COLORS["nvs"]]

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
    render_mimicgen_ablations()
    render_realworld()


if __name__ == "__main__":
    main()
