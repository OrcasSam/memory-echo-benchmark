"""Required figures for the lifecycle benchmark."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

from . import CONDITIONS, STAGES


COLORS = {"control": "#607D8B", "repetition": "#FF9800", "update": "#C62828"}


def _save(fig, path):
    fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches="tight"); plt.close(fig)


def write_figures(output: Path, aggregates, samples):
    folder = output / "figures"; folder.mkdir(parents=True, exist_ok=True)
    stage_rows = aggregates["per_stage_summary"]
    condition_rows = aggregates["condition_summary"]
    qa_rows = aggregates["per_qa_type_summary"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    for condition in CONDITIONS:
        values = []
        for stage in STAGES:
            matches = [r["ED"] for r in stage_rows if r["condition"] == condition
                       and r["stage"] == stage and r["ED"] is not None]
            values.append(sum(matches) / len(matches) if matches else 0)
        ax.plot(range(len(STAGES)), values, marker="o", label=condition,
                color=COLORS[condition])
    ax.set_title("ED across all lifecycle stages")
    ax.set_xticks(range(len(STAGES))); ax.set_xticklabels(
        [s.replace("_", "\n") for s in STAGES], fontsize=8)
    ax.set_ylim(0, 1); ax.grid(alpha=.2); ax.legend()

    panels = (
        (axes[0, 1], "RRI", "RRI at Updating / Reconsolidation"),
        (axes[1, 0], "WEP", "WEP at Delayed Retrieval"),
        (axes[1, 1], "DCE", "DCE at Final Decision"),
    )
    for ax, metric, title in panels:
        values = []
        for condition in CONDITIONS:
            matches = [r[metric] for r in condition_rows if r["condition"] == condition
                       and r[metric] is not None]
            values.append(sum(matches) / len(matches) if matches else 0)
        ax.bar(CONDITIONS, values, color=[COLORS[c] for c in CONDITIONS])
        ax.set_title(title); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=20)
    _save(fig, folder / "Figure_1_metrics_by_stage.png")

    metrics = ("ED", "RRI", "WEP", "DCE")
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, metric in zip(axes, metrics):
        values = []
        for condition in CONDITIONS:
            rows = [r[metric] for r in condition_rows if r["condition"] == condition
                    and r[metric] is not None]
            values.append(sum(rows) / len(rows) if rows else 0)
        ax.bar(CONDITIONS, values, color=[COLORS[c] for c in CONDITIONS])
        ax.set_title(metric); ax.tick_params(axis="x", rotation=25); ax.set_ylim(0, 1)
    _save(fig, folder / "Figure_2_condition_comparison.png")

    qa_types = sorted({r["qa_type"] for r in qa_rows})
    fig, ax = plt.subplots(figsize=(12, 5))
    width = .25
    for ci, condition in enumerate(CONDITIONS):
        values = []
        for qa_type in qa_types:
            rows = [r["answer_similarity"] for r in qa_rows if r["condition"] == condition
                    and r["qa_type"] == qa_type and r["answer_similarity"] is not None]
            values.append(sum(rows) / len(rows) if rows else 0)
        ax.bar([i + (ci - 1) * width for i in range(len(qa_types))], values,
               width=width, label=condition, color=COLORS[condition])
    ax.set_xticks(range(len(qa_types))); ax.set_xticklabels(qa_types, rotation=30, ha="right")
    ax.set_ylabel("Answer similarity"); ax.set_ylim(0, 1); ax.legend()
    _save(fig, folder / "Figure_3_qa_type_scores.png")

    delay_values = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            for point in run["delay_series"]:
                if point["ED"] is not None:
                    delay_values[(run["condition"], point["delay"])].append(point["ED"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in CONDITIONS:
        delays = sorted({key[1] for key in delay_values if key[0] == condition})
        means = [sum(delay_values[(condition, d)]) / len(delay_values[(condition, d)]) for d in delays]
        ax.plot(delays, means, marker="o", label=condition, color=COLORS[condition])
    ax.set_xlabel("Delay (sessions)"); ax.set_ylabel("Echo Density"); ax.set_title("Consecutive delay curve used to calculate WEP")
    ax.grid(alpha=.2); ax.legend()
    _save(fig, folder / "Figure_4_delay_and_wep.png")

    similarities = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            similarities[run["condition"]].extend(t["answer_similarity"] for t in run["trajectory"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for condition in CONDITIONS:
        ax.hist(similarities[condition], bins=15, alpha=.45, label=condition,
                color=COLORS[condition], density=True)
    ax.set_xlabel("Answer similarity to canonical answer"); ax.set_ylabel("Density"); ax.legend()
    _save(fig, folder / "Figure_5_answer_similarity_distribution.png")

    source_rows = aggregates.get("per_source_summary", [])
    sources = sorted({row["source_dataset"] for row in source_rows})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric in zip(axes, ("ED", "DCE")):
        width = .25
        for ci, condition in enumerate(CONDITIONS):
            values = []
            for source in sources:
                matches = [row[metric] for row in source_rows
                           if row["condition"] == condition and row["source_dataset"] == source
                           and row[metric] is not None]
                values.append(sum(matches) / len(matches) if matches else 0)
            ax.bar([i + (ci - 1) * width for i in range(len(sources))], values,
                   width=width, color=COLORS[condition], label=condition)
        ax.set_xticks(range(len(sources))); ax.set_xticklabels(sources, rotation=20)
        ax.set_ylim(0, 1); ax.set_title(f"{metric} by source dataset")
    axes[0].legend()
    _save(fig, folder / "Figure_6_source_dataset_comparison.png")
