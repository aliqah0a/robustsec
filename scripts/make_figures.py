"""Create the paper figures from finished benchmark runs.

Usage:
  python scripts/make_figures.py results/nsl_kdd paper/figures [results/nsl_kdd_ablation]

Figure 2: evasion rate per threat model and model (main run).
Figure 3: PGD robust accuracy, baseline vs adversarially trained.
Figure 4: ablation of the constraint components (if an ablation run is given).
All figures use distinct line styles, markers or hatches, so they read in grayscale.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from robustsec.plotting import plot_metric  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/nsl_kdd")
out = Path(sys.argv[2] if len(sys.argv) > 2 else run / "figures")
ablation = Path(sys.argv[3]) if len(sys.argv) > 3 else None
out.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9})
NAMES = {"baseline": "Baseline", "adv_trained": "Adversarially trained"}

s = pd.read_csv(run / "summary.csv")

# Figure 2
fig = plot_metric(s.replace({"model": NAMES}), "evasion_rate", out / "evasion_rate.pdf",
                  ylabel="Evasion rate")
plt.close(fig)

# Figure 3
fig, ax = plt.subplots(figsize=(4.6, 3.4))
style = {("baseline", "unconstrained"): ("#d62728", "-", "o"),
         ("baseline", "constrained"): ("#d62728", "--", "s"),
         ("adv_trained", "unconstrained"): ("#1f77b4", "-.", "^"),
         ("adv_trained", "constrained"): ("#1f77b4", ":", "D")}
for (m, tm), d in s[s.attack == "PGD"].groupby(["model", "threat_model"], sort=False):
    if (m, tm) not in style:
        continue
    c, ls, mk = style[(m, tm)]
    d = d.sort_values("eps")
    mu, sd = d["robust_accuracy_mean"], d["robust_accuracy_std"].fillna(0)
    ax.plot(d["eps"], mu, ls=ls, color=c, marker=mk, ms=4, lw=1.5,
            label=f"{NAMES.get(m, m)}, {tm}")
    ax.fill_between(d["eps"], mu - sd, mu + sd, color=c, alpha=0.12)
ax.set_xlabel(r"budget $\varepsilon$")
ax.set_ylabel("Robust accuracy (PGD)")
ax.set_ylim(0, 1)
ax.grid(alpha=0.3)
ax.legend(fontsize=7, frameon=False, ncol=2, handlelength=3.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.2))
fig.tight_layout()
fig.savefig(out / "robust_accuracy_pgd.pdf", bbox_inches="tight")
plt.close(fig)

# Figure 4: ablation, strongest attack per threat model at each budget
if ablation is not None:
    a = pd.read_csv(ablation / "summary.csv")
    a = a[a.model == "baseline"]
    best = (a.sort_values("evasion_rate_mean").groupby(["threat_model", "eps"], sort=False)
            .tail(1).sort_values(["threat_model", "eps"]))
    tms = list(dict.fromkeys(a.threat_model))
    epss = sorted(best.eps.unique())
    hatches = ["", "///", "..."]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    w = 0.8 / len(epss)
    for k, e in enumerate(epss):
        d = best[best.eps == e].set_index("threat_model").reindex(tms)
        ax.barh([i + k * w for i in range(len(tms))], d["evasion_rate_mean"].clip(lower=1e-4), height=w,
                xerr=d["evasion_rate_std"], color=str(0.25 + 0.3 * k), hatch=hatches[k % 3],
                edgecolor="black", lw=0.5, label=rf"$\varepsilon={e}$", capsize=2)
    ax.set_yticks([i + w * (len(epss) - 1) / 2 for i in range(len(tms))])
    ax.set_yticklabels([t.split("_", 1)[0] for t in tms])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Evasion rate, strongest attack (log scale)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(axis="x", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "ablation.pdf", bbox_inches="tight")
    plt.close(fig)

print("figures written to", out)
