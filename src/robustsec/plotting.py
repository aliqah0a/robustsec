"""Figures from ``summary.csv`` produced by :func:`robustsec.benchmark.run_benchmark`."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# Colour, line style and marker all differ, so the figures also read in grayscale.
COLORS = {"FGSM": "#1f77b4", "PGD": "#d62728", "CW-L2": "#2ca02c",
          "RandomSearch": "#9467bd", "Greedy": "#8c564b", "Transfer-PGD": "#ff7f0e"}
MARKERS = {"FGSM": "o", "PGD": "s", "CW-L2": "D", "RandomSearch": "^", "Greedy": "x",
           "Transfer-PGD": "v"}
LINESTYLES = {"FGSM": ":", "PGD": "-", "CW-L2": "--", "RandomSearch": "-.",
              "Greedy": (0, (5, 1, 1, 1, 1, 1)), "Transfer-PGD": (0, (1, 1))}


def plot_metric(summary: pd.DataFrame, metric: str = "evasion_rate", out=None,
                ylabel: str = "Evasion rate"):
    """One panel per (threat model, model); one line per attack with ±1 s.d.

    Panels in a row share the y-axis; each row (threat model) gets its own
    range, so small rates under a constrained threat model stay visible.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tms = list(dict.fromkeys(summary["threat_model"]))
    models = list(dict.fromkeys(summary["model"]))
    fig, axes = plt.subplots(len(tms), len(models), figsize=(4.2 * len(models), 3.2 * len(tms)),
                             squeeze=False, sharey="row")
    for i, tm in enumerate(tms):
        for j, m in enumerate(models):
            ax = axes[i][j]
            sub = summary[(summary.threat_model == tm) & (summary.model == m)]
            for atk, d in sub.groupby("attack", sort=False):
                d = d.sort_values("eps")
                mu, sd = d[f"{metric}_mean"], d[f"{metric}_std"].fillna(0)
                ax.plot(d["eps"], mu, marker=MARKERS.get(atk, "o"), color=COLORS.get(atk),
                        ls=LINESTYLES.get(atk, "-"), label=atk, lw=1.6, ms=4)
                ax.fill_between(d["eps"], mu - sd, mu + sd, color=COLORS.get(atk), alpha=0.15)
            ax.set_title(f"{m}, {tm}", fontsize=10)
            ax.set_xlabel(r"budget $\varepsilon$")
            ax.grid(alpha=0.3)
            if j == 0:
                ax.set_ylabel(ylabel)
    for i, tm in enumerate(tms):        # one y-range per threat model (row)
        sub = summary[summary.threat_model == tm]
        top = float((sub[f"{metric}_mean"] + sub[f"{metric}_std"].fillna(0)).max())
        axes[i][0].set_ylim(0, min(1.02, max(top * 1.1, 0.01)))
    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, -0.02), handlelength=3)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=300, bbox_inches="tight")
    return fig
