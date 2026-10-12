"""Figures for the causal Echo Solution validation."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


ARMS = ("baseline", "s1_identity", "s2_retrieval_hygiene",
        "s3_reconsolidation", "s4_persistence", "full_guard")
COLORS = {
    "baseline": "#6B7280", "s1_identity": "#A78BFA",
    "s2_retrieval_hygiene": "#F59E0B", "s3_reconsolidation": "#2563EB",
    "s4_persistence": "#06B6D4", "full_guard": "#059669",
}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_solution_figures(output: Path, aggregates):
    folder = output / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    condition_rows = aggregates["condition_summary"]

    # Raw metrics retain absolute interpretability; causal reductions are in fig 2.
    for framework in sorted({r["framework"] for r in condition_rows}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        for ax, metric in zip(axes.flat, ("ED", "RRI", "WEP", "DCE")):
            width = 0.13
            for ai, arm in enumerate(ARMS):
                values = []
                for condition in ("control", "repetition", "update"):
                    matches = [r[metric] for r in condition_rows
                               if r["framework"] == framework
                               and r["solution_arm"] == arm
                               and r["condition"] == condition]
                    values.append(matches[0] if matches else 0)
                x = [i + (ai - (len(ARMS) - 1) / 2) * width for i in range(3)]
                ax.bar(x, values, width=width, label=arm, color=COLORS[arm])
            ax.set_xticks(range(3))
            ax.set_xticklabels(("Control", "Repetition", "Update"))
            ax.set_ylim(0, 1)
            ax.set_title(metric)
            ax.grid(axis="y", alpha=.2)
        axes[0, 0].legend(fontsize=8, ncol=2)
        fig.suptitle(f"{framework}: absolute Echo metrics by intervention")
        _save(fig, folder / f"Figure_1_{framework}_absolute_metrics.png")

    # Full guard vs baseline DiD, with paired bootstrap confidence intervals.
    rows = [r for r in aggregates["causal_estimands"]
            if r["treated_arm"] == "full_guard" and r["reference_arm"] == "baseline"
            and r["metric"] in ("ED", "RRI", "excess_WEP", "DCE")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, condition in zip(axes, ("repetition", "update")):
        selected = [r for r in rows if r["condition"] == condition]
        labels = [f"{r['framework']}\n{r['metric']}" for r in selected]
        values = [r["DiD"] for r in selected]
        lower = [v - r["ci_low"] for v, r in zip(values, selected)]
        upper = [r["ci_high"] - v for v, r in zip(values, selected)]
        ax.bar(range(len(values)), values, color="#059669")
        ax.errorbar(range(len(values)), values, yerr=[lower, upper], fmt="none",
                    ecolor="black", capsize=3)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=35, ha="right")
        ax.set_title(f"{condition}: FullGuard − Baseline DiD")
        ax.set_ylabel("Change in treatment effect (negative = mitigation)")
        ax.grid(axis="y", alpha=.2)
    _save(fig, folder / "Figure_2_causal_effect_reduction.png")

    # Continuous delay curves are the empirical basis of WEP.
    delay_rows = aggregates["per_delay_summary"]
    frameworks = sorted({r["framework"] for r in delay_rows})
    fig, axes = plt.subplots(1, len(frameworks), figsize=(12, 4.5), sharey=True)
    if len(frameworks) == 1:
        axes = [axes]
    for ax, framework in zip(axes, frameworks):
        for arm in ("baseline", "s2_retrieval_hygiene", "s3_reconsolidation",
                    "s4_persistence", "full_guard"):
            selected = sorted(
                (r for r in delay_rows if r["framework"] == framework
                 and r["solution_arm"] == arm and r["condition"] == "update"),
                key=lambda r: r["delay"],
            )
            ax.plot([r["delay"] for r in selected], [r["ED"] for r in selected],
                    label=arm, color=COLORS[arm], linewidth=2)
        ax.set_title(framework)
        ax.set_xlabel("Sessions after reinforcement stopped")
        ax.set_ylabel("Echo Density")
        ax.set_ylim(0, 1)
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    _save(fig, folder / "Figure_3_continuous_WEP_curves.png")

    # Stage accuracy checks that mitigation does not erase useful memory.
    stage_rows = aggregates["per_stage_summary"]
    stages = ("encoding", "retrieval_practice", "updating_reconsolidation",
              "delayed_retrieval", "final_decision")
    fig, axes = plt.subplots(1, len(frameworks), figsize=(13, 4.8), sharey=True)
    if len(frameworks) == 1:
        axes = [axes]
    for ax, framework in zip(axes, frameworks):
        for arm in ("baseline", "s2_retrieval_hygiene", "s3_reconsolidation",
                    "s4_persistence", "full_guard"):
            values = []
            for stage in stages:
                matches = [r["target_accuracy"] for r in stage_rows
                           if r["framework"] == framework and r["solution_arm"] == arm
                           and r["condition"] == "update" and r["stage"] == stage]
                values.append(matches[0] if matches else 0)
            ax.plot(range(len(stages)), values, marker="o", label=arm,
                    color=COLORS[arm])
        ax.set_xticks(range(len(stages)))
        ax.set_xticklabels(("Encode", "Practice", "Update", "Delay", "Decision"),
                           rotation=25)
        ax.set_ylim(0, 1.03)
        ax.set_title(framework)
        ax.set_ylabel("Stage target accuracy")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    _save(fig, folder / "Figure_4_stage_accuracy.png")
