"""Publication-ready figures for the Generic Memory Echo Benchmark."""
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONDITION_LABELS = {
    "control": "Control",
    "historical_repetition": "Retrieval repetition",
    "historical_repetition_update": "Retrieval + update",
}


def _grouped_bars(ax, categories, series, ylabel, title):
    width = 0.8 / max(1, len(series))
    centers = list(range(len(categories)))
    for index, (label, values) in enumerate(series.items()):
        offset = (index - (len(series) - 1) / 2) * width
        ax.bar(
            [center + offset for center in centers],
            values,
            width=width,
            label=label,
        )
    ax.set_xticks(centers, categories, rotation=18, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axhline(0, color="#444444", linewidth=0.7)
    ax.legend(fontsize=8)


def _protocol_ed_figure(output, protocol_rows):
    protocols = sorted({row["protocol"] for row in protocol_rows})
    frameworks = sorted({row["framework"] for row in protocol_rows})
    fig, axes = plt.subplots(1, len(frameworks), figsize=(13, 4.8), sharey=True)
    if len(frameworks) == 1:
        axes = [axes]
    for ax, framework in zip(axes, frameworks):
        series = {}
        for condition, label in CONDITION_LABELS.items():
            lookup = {
                row["protocol"]: row["ED"]
                for row in protocol_rows
                if row["framework"] == framework
                and row["condition"] == condition
            }
            series[label] = [lookup.get(protocol, 0) or 0 for protocol in protocols]
        _grouped_bars(
            ax, protocols, series, "ED",
            f"{framework}: standardized historical-memory occupancy",
        )
        ax.set_ylim(0, 1)
    fig.suptitle("Figure 1. Echo Density by protocol and condition")
    fig.tight_layout()
    fig.savefig(output / "Figure_1_protocol_echo_density.png", dpi=180)
    plt.close(fig)


def _four_metric_figure(output, protocol_rows, paired_rows):
    protocols = sorted({
        row["protocol"] for row in paired_rows
        if row["protocol"] != "ALL_PROTOCOLS"
    })
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    definitions = (
        ("ED_lift", "update_minus_control", "ED lift", "Formation"),
        ("UIR_drop", "update_minus_repetition", "UIR drop", "Amplification"),
        ("EPS", None, "EPS", "Retention"),
        ("DBI_lift", "update_minus_control", "DBI lift", "Influence"),
    )
    frameworks = sorted({row["framework"] for row in protocol_rows})
    for ax, (metric, contrast, ylabel, title) in zip(axes.flat, definitions):
        series = {}
        for framework in frameworks:
            if metric == "EPS":
                lookup = {
                    row["protocol"]: row["EPS"]
                    for row in protocol_rows
                    if row["framework"] == framework
                    and row["condition"] == "historical_repetition_update"
                }
            else:
                lookup = {
                    row["protocol"]: row[metric]
                    for row in paired_rows
                    if row["framework"] == framework
                    and row["contrast"] == contrast
                    and row["protocol"] != "ALL_PROTOCOLS"
                }
            series[framework] = [
                lookup.get(protocol, 0) or 0 for protocol in protocols
            ]
        _grouped_bars(
            ax, protocols, series, ylabel,
            f"{title}: {ylabel}",
        )
        if metric in ("ED_lift", "UIR_drop", "DBI_lift"):
            ax.axhline(0.05, color="#b22222", linestyle="--", linewidth=1)
        elif metric == "EPS":
            ax.axhline(0.80, color="#b22222", linestyle="--", linewidth=1)
    fig.suptitle("Figure 2. Four complementary Memory Echo measurements")
    fig.tight_layout()
    fig.savefig(output / "Figure_2_four_metric_protocol_comparison.png", dpi=180)
    plt.close(fig)


def _phase_distribution_figure(output, samples):
    phases = (
        "echo_seeding",
        "update_adaptation",
        "echo_retention",
        "decision_influence",
    )
    values = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            for turn in run["memory_trajectory"]:
                value = turn["echo_density"]
                if value is not None:
                    values[(run["condition"], turn["phase"])].append(value)
    data, labels, colors = [], [], []
    palette = ("#7f8c8d", "#4c78a8", "#e45756")
    for phase in phases:
        for condition, color in zip(CONDITION_LABELS, palette):
            data.append(values[(condition, phase)])
            labels.append(
                f"{phase.replace('_', ' ')}\n"
                f"{CONDITION_LABELS[condition]}"
            )
            colors.append(color)
    fig, ax = plt.subplots(figsize=(16, 6))
    box = ax.boxplot(
        data, tick_labels=labels, patch_artist=True, showfliers=False
    )
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_ylabel("Turn-level ED")
    ax.set_ylim(0, 1)
    ax.set_title("Figure 3. ED distributions across longitudinal phases")
    ax.tick_params(axis="x", rotation=28, labelsize=8)
    fig.tight_layout()
    fig.savefig(output / "Figure_3_phase_ED_distributions.png", dpi=180)
    plt.close(fig)


def _qa_sensitivity_figure(output, qa_rows):
    lookup = {
        (
            row["framework"], row["protocol"], row["condition"],
            row["qa_type"],
        ): row
        for row in qa_rows
    }
    drops = defaultdict(list)
    for key, control in lookup.items():
        framework, protocol, condition, qa_type = key
        if condition != "control":
            continue
        update = lookup.get((
            framework, protocol, "historical_repetition_update", qa_type
        ))
        if update:
            drops[(framework, qa_type)].append(
                control["answer_score"] - update["answer_score"]
            )
    qa_types = sorted({key[1] for key in drops})
    frameworks = sorted({key[0] for key in drops})
    series = {
        framework: [
            sum(drops[(framework, qa_type)])
            / len(drops[(framework, qa_type)])
            for qa_type in qa_types
        ]
        for framework in frameworks
    }
    fig, ax = plt.subplots(figsize=(10.5, 5))
    _grouped_bars(
        ax, qa_types, series, "Control − retrieval+update answer score",
        "Figure 4. QA-type sensitivity to Echo reinforcement",
    )
    fig.tight_layout()
    fig.savefig(output / "Figure_4_QA_type_sensitivity.png", dpi=180)
    plt.close(fig)


def _memory_type_figure(output, memory_rows):
    rows = [
        row for row in memory_rows
        if row["condition"] == "historical_repetition_update"
    ]
    memory_types = sorted({row["memory_unit_type"] for row in rows})
    frameworks = sorted({row["framework"] for row in rows})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for ax, metric in zip(axes.flat, ("ED", "UIR", "EPS", "DBI")):
        series = {}
        for framework in frameworks:
            lookup = {
                row["memory_unit_type"]: row[metric]
                for row in rows if row["framework"] == framework
            }
            series[framework] = [
                lookup.get(memory_type, 0) or 0
                for memory_type in memory_types
            ]
        _grouped_bars(
            ax, memory_types, series, metric,
            f"Retrieval+update condition: {metric}",
        )
    fig.suptitle("Figure 5. Memory-unit-type differences")
    fig.tight_layout()
    fig.savefig(output / "Figure_5_memory_unit_type_comparison.png", dpi=180)
    plt.close(fig)


def write_generic_figures(
    output_dir: Path,
    run_aggregates,
    turn_aggregates,
    paired_rows,
    samples,
):
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    _protocol_ed_figure(
        figure_dir, run_aggregates["protocol_summary"]
    )
    _four_metric_figure(
        figure_dir, run_aggregates["protocol_summary"], paired_rows
    )
    _phase_distribution_figure(figure_dir, samples)
    _qa_sensitivity_figure(
        figure_dir, turn_aggregates["qa_type_summary"]
    )
    _memory_type_figure(
        figure_dir, run_aggregates["memory_type_summary"]
    )
