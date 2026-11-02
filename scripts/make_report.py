"""Turn a results folder into a chart and a markdown table for your report.
Usage: python scripts/make_report.py reports/results/time_with_enso"""
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main(folder):
    folder = Path(folder)
    s = pd.read_csv(folder / "summary.csv")
    hs = sorted(s["horizon"].unique())
    fig, axes = plt.subplots(1, len(hs), figsize=(4.6 * len(hs), 4.2), sharey=True, squeeze=False)
    for ax, h in zip(axes[0], hs):
        d = s[s["horizon"] == h].sort_values("pr_auc")
        err = None
        if {"pr_auc_ci_low", "pr_auc_ci_high"} <= set(d.columns):
            err = [(d["pr_auc"] - d["pr_auc_ci_low"]).clip(lower=0), (d["pr_auc_ci_high"] - d["pr_auc"]).clip(lower=0)]
        ax.barh(d["model"], d["pr_auc"], xerr=err, color="#0d7a82", ecolor="#555", capsize=3)
        ax.set_title(f"{h}-week forecast")
        ax.set_xlabel("PR-AUC (higher is better)")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle(f"Model comparison: {folder.name}", fontweight="bold")
    fig.tight_layout()
    fig.savefig(folder / "pr_auc_by_model.png", dpi=160)
    cols = [c for c in ["horizon", "model", "pr_auc", "pr_auc_ci_low", "pr_auc_ci_high", "recall_at_prec", "brier", "ece"] if c in s.columns]
    (folder / "summary.md").write_text(s[cols].round(3).to_markdown(index=False) if hasattr(s, "to_markdown") else s[cols].round(3).to_string(index=False))
    print(f"wrote {folder/'pr_auc_by_model.png'} and {folder/'summary.md'}")


if __name__ == "__main__":
    main(sys.argv[1])
