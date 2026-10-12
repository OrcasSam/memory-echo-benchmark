"""Compare the 10x100 benchmark with the scaled 30x200 experiment."""
import csv
import json
from pathlib import Path


METRICS = (
    "echo_density", "echo_persistence_score", "answer_score",
    "decision_score", "ed_lift_ear", "answer_score_drop", "decision_score_drop",
)


def _key(row):
    return row["framework"], row["condition"]


def compare_condition_rows(old_rows, new_rows, matched_new_rows,
                           matched_old_rows=None, matched_count=None):
    old = {_key(r): r for r in old_rows}
    matched_old = {
        _key(r): r for r in (matched_old_rows if matched_old_rows is not None else old_rows)
    }
    outputs = []
    matched_label = (
        f"matched_{matched_count}_shared_instructions"
        if matched_count is not None else "matched_samples"
    )
    for scope, rows, baseline in (
        ("overall_10x100_vs_30x200", new_rows, old),
        (matched_label, matched_new_rows, matched_old),
    ):
        for new in rows:
            prior = baseline[_key(new)]
            row = {
                "comparison_scope": scope,
                "framework": new["framework"],
                "condition": new["condition"],
                "old_samples": int(prior["samples"]),
                "old_qa_per_sample": 100,
                "new_samples": int(new["samples"]),
                "new_qa_per_sample": 200,
            }
            for metric in METRICS:
                old_value = float(prior[metric])
                new_value = float(new[metric])
                row[f"old_{metric}"] = old_value
                row[f"new_{metric}"] = new_value
                row[f"delta_{metric}"] = new_value - old_value
            outputs.append(row)
    return outputs


def compare_phase_rows(old_rows, new_rows):
    old = {
        (r["framework"], r["condition"], r["phase"]): r
        for r in old_rows
    }
    outputs = []
    for new in new_rows:
        prior = old[(new["framework"], new["condition"], new["phase"])]
        outputs.append({
            "framework": new["framework"],
            "condition": new["condition"],
            "phase": new["phase"],
            "old_echo_density": float(prior["echo_density"]),
            "new_echo_density": float(new["echo_density"]),
            "delta_echo_density": float(new["echo_density"]) - float(prior["echo_density"]),
            "old_answer_score": float(prior["answer_score"]),
            "new_answer_score": float(new["answer_score"]),
            "delta_answer_score": float(new["answer_score"]) - float(prior["answer_score"]),
        })
    return outputs


def load_prior_results(prior_dir: Path):
    aggregate = json.loads((prior_dir / "aggregate_results.json").read_text(encoding="utf-8"))
    condition_rows = aggregate["condition_summary"]
    phase_rows = aggregate.get("by_framework_condition_phase")
    if phase_rows is None:
        with (prior_dir / "aggregate_by_phase.csv").open(
            encoding="utf-8-sig", newline=""
        ) as f:
            phase_rows = list(csv.DictReader(f))
    return condition_rows, phase_rows


def write_comparison_figures(output_dir, condition_comparison, phase_comparison):
    import matplotlib.pyplot as plt

    figdir = Path(output_dir) / "comparison_figures"
    figdir.mkdir(exist_ok=True)
    overall = [r for r in condition_comparison
               if r["comparison_scope"] == "overall_10x100_vs_30x200"]
    labels = [f"{r['framework']}\n{r['condition']}" for r in overall]
    x = list(range(len(labels)))
    width = 0.36
    plt.figure(figsize=(12, 5))
    plt.bar([i - width / 2 for i in x], [r["old_echo_density"] for r in overall],
            width=width, label="10x100")
    plt.bar([i + width / 2 for i in x], [r["new_echo_density"] for r in overall],
            width=width, label="30x200")
    plt.xticks(x, labels, rotation=18, ha="right")
    plt.ylabel("Echo Density")
    plt.title("Scale Comparison: Echo Density")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "Comparison_1_ED_Old_vs_New.png", dpi=160)
    plt.close()

    update = [r for r in overall if r["condition"] == "historical_repetition_update"]
    labels = [r["framework"] for r in update]
    x = list(range(len(labels)))
    plt.figure(figsize=(8, 5))
    plt.bar([i - width / 2 for i in x], [r["old_answer_score"] for r in update],
            width=width, label="10x100")
    plt.bar([i + width / 2 for i in x], [r["new_answer_score"] for r in update],
            width=width, label="30x200")
    plt.xticks(x, labels); plt.ylabel("Answer score")
    plt.title("Update Condition: Answer Score at Two Scales")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "Comparison_2_Update_Answer_Score.png", dpi=160)
    plt.close()

    update_phase = [r for r in phase_comparison
                    if r["condition"] == "historical_repetition_update"]
    phases = ["encoding", "retrieval_practice", "updating_reconsolidation",
              "delayed_retrieval", "final_decision"]
    old_vals = [
        sum(r["old_echo_density"] for r in update_phase if r["phase"] == p) / 2
        for p in phases
    ]
    new_vals = [
        sum(r["new_echo_density"] for r in update_phase if r["phase"] == p) / 2
        for p in phases
    ]
    x = list(range(len(phases)))
    plt.figure(figsize=(11, 5))
    plt.plot(x, old_vals, marker="o", label="10x100")
    plt.plot(x, new_vals, marker="o", label="30x200")
    plt.xticks(x, phases, rotation=20, ha="right")
    plt.ylabel("Echo Density")
    plt.title("Update Condition: ED Across Cognitive Phases")
    plt.legend(); plt.tight_layout()
    plt.savefig(figdir / "Comparison_3_Phase_ED.png", dpi=160)
    plt.close()
