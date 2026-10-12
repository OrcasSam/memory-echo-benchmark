"""Aggregate cognitive Memory Echo metrics by condition, QA type, and phase."""
import csv
import json
from collections import defaultdict
from pathlib import Path


def _mean(values):
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate_by_qa_type(samples):
    grouped = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            for turn in run["memory_trajectory"]:
                grouped[(run["framework"], run["condition"], turn["qa_type"])].append(turn)
    rows = []
    for (framework, condition, qa_type), turns in sorted(grouped.items()):
        evaluations = [t["answer_evaluation"] for t in turns]
        rows.append({
            "framework": framework,
            "condition": condition,
            "qa_type": qa_type,
            "qa_count": len(turns),
            "echo_density": _mean(t["echo_density"] for t in turns),
            "semantic_similarity": _mean(e["semantic_similarity"] for e in evaluations),
            "answer_score": _mean(e["answer_score"] for e in evaluations),
            "target_match_rate": _mean(e["target_id_match"] for e in evaluations),
            "fully_correct_rate": _mean(e["semantic_grade"] == "fully_correct" for e in evaluations),
            "partial_or_better_rate": _mean(e["semantic_grade"] != "incorrect" for e in evaluations),
            "decision_score": _mean(e["answer_score"] for t, e in zip(turns, evaluations)
                                    if t["qa_type"] == "decision") if qa_type == "decision" else "",
        })
    return rows


def aggregate_by_phase_and_qa_type(samples):
    """Joint aggregation required by the sixth benchmark."""
    grouped = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            for turn in run["memory_trajectory"]:
                key = (run["framework"], run["condition"], turn["phase"], turn["qa_type"])
                grouped[key].append(turn)
    rows = []
    for (framework, condition, phase, qa_type), turns in sorted(grouped.items()):
        evaluations = [t["answer_evaluation"] for t in turns]
        decisions = [e["answer_score"] for t, e in zip(turns, evaluations)
                     if t["qa_type"] == "decision"]
        rows.append({
            "framework": framework,
            "condition": condition,
            "phase": phase,
            "qa_type": qa_type,
            "qa_count": len(turns),
            "echo_density": _mean(t["echo_density"] for t in turns),
            "semantic_similarity": _mean(e["semantic_similarity"] for e in evaluations),
            "answer_score": _mean(e["answer_score"] for e in evaluations),
            "target_match_rate": _mean(e["target_id_match"] for e in evaluations),
            "fully_correct_rate": _mean(e["semantic_grade"] == "fully_correct" for e in evaluations),
            "semantically_correct_or_better_rate": _mean(
                e["semantic_grade"] in ("fully_correct", "semantically_correct") for e in evaluations
            ),
            "partial_or_better_rate": _mean(e["semantic_grade"] != "incorrect" for e in evaluations),
            "decision_score": _mean(decisions) if decisions else "",
        })
    return rows


def aggregate_by_phase(samples):
    grouped = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            for turn in run["memory_trajectory"]:
                grouped[(run["framework"], run["condition"], turn["phase"])].append(turn)
    rows = []
    for (framework, condition, phase), turns in sorted(grouped.items()):
        evaluations = [t["answer_evaluation"] for t in turns]
        decisions = [e["answer_score"] for t, e in zip(turns, evaluations)
                     if t["qa_type"] == "decision"]
        rows.append({
            "framework": framework,
            "condition": condition,
            "phase": phase,
            "qa_count": len(turns),
            "echo_density": _mean(t["echo_density"] for t in turns),
            "semantic_similarity": _mean(e["semantic_similarity"] for e in evaluations),
            "answer_score": _mean(e["answer_score"] for e in evaluations),
            "target_match_rate": _mean(e["target_id_match"] for e in evaluations),
            "decision_score": _mean(decisions) if decisions else "",
        })
    controls = {(r["framework"], r["phase"]): r for r in rows if r["condition"] == "control"}
    for row in rows:
        control = controls[(row["framework"], row["phase"])]
        row["phase_ed_lift_vs_control"] = row["echo_density"] - control["echo_density"]
        row["phase_answer_score_drop_vs_control"] = control["answer_score"] - row["answer_score"]
    return rows


def aggregate_conditions(samples):
    grouped = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            grouped[(run["framework"], run["condition"])].extend(run["memory_trajectory"])
    rows = []
    for (framework, condition), turns in sorted(grouped.items()):
        consequential = [t for t in turns if t["phase"] in
                         ("updating_reconsolidation", "delayed_retrieval", "final_decision")]
        practice = [t for t in turns if t["phase"] == "retrieval_practice"]
        delayed = [t for t in turns if t["phase"] == "delayed_retrieval"]
        decisions = [t for t in turns if t["qa_type"] == "decision" and t["phase"] == "final_decision"]
        rows.append({
            "framework": framework,
            "condition": condition,
            "samples": len({t["sample_id"] for t in turns}),
            "echo_density": _mean(t["echo_density"] for t in consequential),
            "formation_ed": _mean(t["echo_density"] for t in practice),
            "delayed_ed": _mean(t["echo_density"] for t in delayed),
            "echo_persistence_score": (_mean(t["echo_density"] for t in delayed) /
                                       _mean(t["echo_density"] for t in practice)) if practice else 0.0,
            "semantic_similarity": _mean(t["answer_evaluation"]["semantic_similarity"] for t in consequential),
            "answer_score": _mean(t["answer_evaluation"]["answer_score"] for t in consequential),
            "target_match_rate": _mean(t["answer_evaluation"]["target_id_match"] for t in consequential),
            "decision_score": _mean(t["answer_evaluation"]["answer_score"] for t in decisions),
        })
    controls = {r["framework"]: r for r in rows if r["condition"] == "control"}
    for row in rows:
        control = controls[row["framework"]]
        row["ed_lift_ear"] = row["echo_density"] - control["echo_density"]
        row["answer_score_drop"] = control["answer_score"] - row["answer_score"]
        row["decision_score_drop"] = control["decision_score"] - row["decision_score"]
    return rows


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def write_figures(output_dir: Path, qa_rows, condition_rows, samples, phase_rows=None):
    import matplotlib.pyplot as plt
    figdir = output_dir / "figures"; figdir.mkdir(exist_ok=True)
    labels = [f"{r['framework']}\n{r['condition']}" for r in condition_rows]
    plt.figure(figsize=(11, 5))
    plt.bar(labels, [r["echo_density"] for r in condition_rows])
    plt.ylabel("Echo Density"); plt.title("ED: Control vs Retrieval Practice vs Retrieval + Update")
    plt.xticks(rotation=18, ha="right"); plt.tight_layout()
    plt.savefig(figdir / "Figure_1_ED_Comparison.png", dpi=160); plt.close()

    qa_types = sorted({r["qa_type"] for r in qa_rows})
    conditions = ["control", "historical_repetition", "historical_repetition_update"]
    plt.figure(figsize=(12, 5))
    x = list(range(len(qa_types))); width = 0.25
    for j, condition in enumerate(conditions):
        vals = [_mean(r["answer_score"] for r in qa_rows
                      if r["condition"] == condition and r["qa_type"] == q) for q in qa_types]
        plt.bar([i + (j - 1) * width for i in x], vals, width=width, label=condition)
    plt.xticks(x, qa_types, rotation=25, ha="right"); plt.ylabel("Mean graded semantic answer score")
    plt.title("Semantic Answer Score by QA Type"); plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "Figure_2_Semantic_Score_by_QA_Type.png", dpi=160); plt.close()

    values = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            values[run["condition"]].extend(
                t["answer_evaluation"]["semantic_similarity"] for t in run["memory_trajectory"]
            )
    plt.figure(figsize=(10, 5))
    bins = [i / 20 for i in range(21)]
    for condition in conditions:
        plt.hist(values[condition], bins=bins, alpha=0.45, density=True, label=condition)
    plt.xlabel("Raw semantic similarity"); plt.ylabel("Density")
    plt.title("Answer Similarity Distribution"); plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "Figure_3_Answer_Similarity_Distribution.png", dpi=160); plt.close()

    if phase_rows:
        phases = ["encoding", "retrieval_practice", "updating_reconsolidation",
                  "delayed_retrieval", "final_decision"]
        plt.figure(figsize=(12, 5))
        x = list(range(len(phases))); width = 0.25
        for j, condition in enumerate(conditions):
            vals = [_mean(r["answer_score"] for r in phase_rows
                          if r["condition"] == condition and r["phase"] == phase)
                    for phase in phases]
            plt.bar([i + (j - 1) * width for i in x], vals, width=width, label=condition)
        plt.xticks(x, phases, rotation=20, ha="right")
        plt.ylabel("Mean graded semantic answer score")
        plt.title("Mean Answer Score by Cognitive Phase")
        plt.legend(); plt.tight_layout()
        plt.savefig(figdir / "Figure_4_Mean_Score_by_Phase.png", dpi=160); plt.close()


def build_conclusion(condition_rows, qa_rows, phase_rows=None):
    update_rows = [r for r in condition_rows if r["condition"] == "historical_repetition_update"]
    mean_lift = _mean(r["ed_lift_ear"] for r in update_rows)
    mean_drop = _mean(r["answer_score_drop"] for r in update_rows)
    effects = defaultdict(list)
    lookup = {(r["framework"], r["condition"], r["qa_type"]): r for r in qa_rows}
    for qa_type in {r["qa_type"] for r in qa_rows}:
        for framework in {r["framework"] for r in qa_rows}:
            control = lookup[(framework, "control", qa_type)]["answer_score"]
            update = lookup[(framework, "historical_repetition_update", qa_type)]["answer_score"]
            effects[qa_type].append(control - update)
    qa_effects = {k: _mean(v) for k, v in effects.items()}
    most_affected = max(qa_effects, key=qa_effects.get)
    observed = mean_lift > 0.05
    result = {
        "memory_echo_observed": observed,
        "mean_ed_lift_repetition_update_vs_control": mean_lift,
        "mean_answer_score_drop": mean_drop,
        "most_affected_qa_type": most_affected,
        "qa_type_answer_score_drops": qa_effects,
        "interpretation": (
            "Memory Echo was observed: retrieval-writeback increased historical-memory occupancy "
            "and reduced graded semantic task performance."
            if observed else
            "No material Memory Echo was observed under the predefined ED-lift criterion."
        ),
    }
    if phase_rows:
        update_phase_rows = [r for r in phase_rows
                             if r["condition"] == "historical_repetition_update"]
        phase_effects = defaultdict(list)
        phase_score_drops = defaultdict(list)
        for row in update_phase_rows:
            phase_effects[row["phase"]].append(row["phase_ed_lift_vs_control"])
            phase_score_drops[row["phase"]].append(row["phase_answer_score_drop_vs_control"])
        mean_phase_lifts = {k: _mean(v) for k, v in phase_effects.items()}
        mean_phase_drops = {k: _mean(v) for k, v in phase_score_drops.items()}
        result.update({
            "most_echo_prone_phase": max(mean_phase_lifts, key=mean_phase_lifts.get),
            "phase_ed_lifts": mean_phase_lifts,
            "phase_answer_score_drops": mean_phase_drops,
            "semantic_threshold_interpretation": {
                "fully_correct": "answer_score >= 0.90",
                "semantically_correct": "0.80 <= answer_score < 0.90",
                "partially_correct": "0.65 <= answer_score < 0.80",
                "incorrect": "answer_score < 0.65",
            },
        })
    return result
