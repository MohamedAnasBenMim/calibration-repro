"""Figures in the style of Guo et al. (2017), Figures 1-4."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import bin_stats, ece_mce  # noqa: E402


def _conf_hist(ax, conf, correct, n_bins):
    ax.hist(
        conf,
        bins=np.linspace(0, 1, n_bins + 1),
        weights=np.ones_like(conf) / len(conf),
        color="tab:blue",
        edgecolor="black",
        linewidth=0.5,
    )
    acc, avg = correct.mean(), conf.mean()
    ax.axvline(acc, color="black", ls="--", lw=1.5, label=f"Accuracy ({100 * acc:.1f}%)")
    ax.axvline(avg, color="tab:red", ls="--", lw=1.5, label=f"Avg. confidence ({100 * avg:.1f}%)")
    ax.set_xlim(0, 1)
    ax.set_ylabel("% of samples")
    ax.legend(fontsize=7, loc="upper left")


def _reliability(ax, conf, correct, n_bins):
    s = bin_stats(conf, correct, n_bins)
    centers = (s.edges[:-1] + s.edges[1:]) / 2
    width = 1.0 / n_bins
    acc = np.nan_to_num(s.accuracy)
    # Gap is drawn between bin accuracy and the bin's *center* only for display;
    # ECE itself uses the mean confidence of the bin.
    ax.bar(centers, acc, width=width, color="tab:blue", edgecolor="black", lw=0.5, label="Outputs")
    gap_top = np.where(s.counts > 0, centers, 0)
    ax.bar(
        centers,
        gap_top - acc,
        bottom=acc,
        width=width,
        color="tab:red",
        alpha=0.3,
        edgecolor="tab:red",
        lw=0.5,
        hatch="//",
        label="Gap",
    )
    ax.plot([0, 1], [0, 1], ls="--", color="gray")
    ece, mce = ece_mce(conf, correct, n_bins)
    ax.text(
        0.97,
        0.05,
        f"ECE={100 * ece:.2f}%\nMCE={100 * mce:.1f}%",
        ha="right",
        va="bottom",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.8, lw=0.5),
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=7, loc="upper left")


def reliability_figure(
    panels: list[tuple[str, np.ndarray, np.ndarray]], path: Path, n_bins: int = 15, hist: bool = True
) -> None:
    """panels: list of (title, probs, labels). Fig. 1 layout: histogram over diagram."""
    n = len(panels)
    rows = 2 if hist else 1
    fig, axes = plt.subplots(rows, n, figsize=(3.2 * n, 3.0 * rows), squeeze=False)
    for j, (title, probs, labels) in enumerate(panels):
        conf = probs.max(1)
        correct = (probs.argmax(1) == labels).astype(float)
        if hist:
            _conf_hist(axes[0, j], conf, correct, n_bins)
            axes[0, j].set_title(title, fontsize=10)
        _reliability(axes[rows - 1, j], conf, correct, n_bins)
        if not hist:
            axes[0, j].set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def training_curves_figure(histories: dict[str, list[dict]], path: Path) -> None:
    """Fig. 3 style: test error (%) and test NLL vs epoch for each model."""
    fig, axes = plt.subplots(1, len(histories), figsize=(4.2 * len(histories), 3.2), squeeze=False)
    for ax, (name, hist) in zip(axes[0], histories.items(), strict=True):
        ep = [h["epoch"] for h in hist]
        ax.plot(
            ep, [h["test_error"] for h in hist], color="tab:blue", marker="o", ms=3, label="Test error (%)"
        )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Test error (%)", color="tab:blue")
        ax2 = ax.twinx()
        ax2.plot(ep, [h["test_nll"] for h in hist], color="tab:red", marker="s", ms=3, label="Test NLL")
        ax2.plot(ep, [h["train_loss"] for h in hist], color="tab:red", ls=":", lw=1, label="Train NLL")
        ax2.set_ylabel("NLL", color="tab:red")
        ax.set_title(name, fontsize=10)
        lines = ax.get_legend_handles_labels()
        lines2 = ax2.get_legend_handles_labels()
        ax.legend(lines[0] + lines2[0], lines[1] + lines2[1], fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def factors_figure(groups: dict[str, list[dict]], path: Path) -> None:
    """Fig. 2 style: error and ECE (%) as a function of each factor."""
    fig, axes = plt.subplots(1, len(groups), figsize=(3.4 * len(groups), 3.0), squeeze=False)
    for ax, (factor, rows) in zip(axes[0], groups.items(), strict=True):
        labels = [str(r["value"]) for r in rows]
        x = np.arange(len(rows))
        w = 0.38
        ax.bar(x - w / 2, [r["error"] for r in rows], w, label="Error", color="tab:blue")
        ax.bar(x + w / 2, [r["ece"] for r in rows], w, label="ECE", color="tab:red")
        ax.set_xticks(x, labels)
        ax.set_xlabel(factor)
        ax.set_ylabel("Error / ECE (%)")
        ax.set_title(f"Varying {factor}", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
