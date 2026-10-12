"""Merge and validate matched top-k=5 A/B/C/D Memory Echo experiments."""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_memory_echo_abcd_top5"
SOLUTIONS = "ABCD"
FRAMEWORKS = ("A-MEM", "Mem0")
CONDITIONS = ("control", "repetition", "update")
METRICS = ("ED", "RRI", "WEP", "DCE")
STAGES = ("encoding", "retrieval_practice", "updating_reconsolidation",
          "delayed_retrieval", "final_decision")


def read_jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def bootstrap(values, seed, n=5000):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(n, len(array)), replace=True), axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def load_solution(solution):
    directory = ROOT / f"results_memory_echo_solution_{solution}_top5"
    return read_jsonl(directory / "sample_level_decisions.jsonl")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = []
    by_solution = {}
    for solution in SOLUTIONS:
        rows = load_solution(solution)
        by_solution[solution] = rows
        all_rows.extend(rows)

    aggregate = []
    for solution in SOLUTIONS:
        for framework in FRAMEWORKS:
            for condition in CONDITIONS:
                rows = [r for r in by_solution[solution]
                        if r["framework"] == framework and r["condition"] == condition]
                aggregate.append({
                    "solution": solution, "framework": framework,
                    "condition": condition, "n": len(rows),
                    **{metric: float(np.mean([r[metric] for r in rows])) for metric in METRICS},
                    "contamination_rate": float(np.mean([r["contaminated"] for r in rows])),
                    "gold_preference_rate": float(np.mean([r["gold_preferred"] for r in rows])),
                })
    write_csv(OUT / "aggregate_results.csv", aggregate)
    (OUT / "aggregate_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    comparisons = []
    for solution in "BCD":
        for framework in FRAMEWORKS:
            for condition in CONDITIONS:
                a = {r["sample_id"]: r for r in by_solution["A"]
                     if r["framework"] == framework and r["condition"] == condition}
                b = {r["sample_id"]: r for r in by_solution[solution]
                     if r["framework"] == framework and r["condition"] == condition}
                ids = sorted(set(a) & set(b))
                for metric_index, metric in enumerate(METRICS):
                    differences = [float(b[i][metric]) - float(a[i][metric]) for i in ids]
                    low, high = bootstrap(
                        differences,
                        seed=20260826 + 100 * ord(solution) + 10 * METRICS.index(metric) + metric_index,
                    )
                    comparisons.append({
                        "solution": solution, "reference": "A", "framework": framework,
                        "condition": condition, "metric": metric, "n": len(ids),
                        "solution_mean": float(np.mean([b[i][metric] for i in ids])),
                        "A_mean": float(np.mean([a[i][metric] for i in ids])),
                        "paired_delta_solution_minus_A": float(np.mean(differences)),
                        "ci95_low": low, "ci95_high": high,
                        "ci_excludes_zero": bool(low > 0 or high < 0),
                    })
    write_csv(OUT / "paired_comparison_vs_A.csv", comparisons)

    validations = []
    for solution in SOLUTIONS:
        directory = ROOT / f"results_memory_echo_solution_{solution}_top5"
        runs_count = unique_runs = max_k = over_k = 0
        delays = set()
        run_ids = set()
        with (directory / "sample_level.jsonl").open("r", encoding="utf-8") as handle:
            for line in handle:
                run = json.loads(line); runs_count += 1; run_ids.add(run["run_id"])
                for turn in run["trajectory"]:
                    count = int(turn["retrieval_count"])
                    max_k = max(max_k, count); over_k += int(count > 5)
                    if turn["stage"] == "delayed_retrieval":
                        delays.add(int(turn["delay"]))
        predictions = by_solution[solution]
        cases = json.loads((directory / "stage_cases.json").read_text(encoding="utf-8"))
        case_counts = Counter(row["stage"] for row in cases)
        validations.append({
            "solution": solution, "runs": runs_count, "unique_runs": len(run_ids),
            "decision_answers": len(predictions),
            "empty_decision_answers": sum(not r["freeform_answer"].strip() for r in predictions),
            "unique_DCE_values": len({round(float(r["DCE"]), 10) for r in predictions}),
            "maximum_retrieval_count": max_k, "retrievals_over_top5": over_k,
            "delay_points": len(delays), "delay_1_to_30_complete": delays == set(range(1, 31)),
            "case_count": len(cases),
            "five_cases_per_stage": all(case_counts[s] == 5 for s in STAGES),
            "affected_cases": sum(bool(r["echo_affected"]) for r in cases),
            "correct_cases": sum(bool(r["answer_correct"]) for r in cases),
            "incorrect_cases": sum(not bool(r["answer_correct"]) for r in cases),
            "passed": bool(
                runs_count == 1200 and len(run_ids) == 1200 and len(predictions) == 1200
                and over_k == 0 and max_k <= 5 and delays == set(range(1, 31))
                and not any(not r["freeform_answer"].strip() for r in predictions)
                and len(cases) == 25 and all(case_counts[s] == 5 for s in STAGES)
            ),
        })
    write_csv(OUT / "protocol_validation.csv", validations)
    (OUT / "protocol_validation.json").write_text(
        json.dumps(validations, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    case_summary = []
    all_cases = json.loads((OUT / "all_stage_cases.json").read_text(encoding="utf-8"))
    for solution in SOLUTIONS:
        for stage in STAGES:
            rows = [r for r in all_cases if r["solution"] == solution and r["stage"] == stage]
            case_summary.append({
                "solution": solution, "stage": stage, "n": len(rows),
                "correct": sum(bool(r["answer_correct"]) for r in rows),
                "incorrect": sum(not bool(r["answer_correct"]) for r in rows),
                "echo_affected": sum(bool(r["echo_affected"]) for r in rows),
            })
    write_csv(OUT / "stage_case_summary.csv", case_summary)
    manifest = {
        "experiment": "Memory Echo solutions A/B/C/D matched comparison",
        "samples": 200, "runs_per_solution": 1200, "top_k": 5,
        "frameworks": list(FRAMEWORKS), "conditions": list(CONDITIONS),
        "delay_points": list(range(1, 31)), "freeform_model": "deepseek-chat",
        "semantic_evaluator": "sentence-transformers/all-MiniLM-L6-v2",
        "case_policy": "5 per stage per solution; real correct/error/affected outcomes retained",
        "validation_passed": all(r["passed"] for r in validations),
    }
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"aggregate_rows": len(aggregate), "comparisons": len(comparisons),
                      "validations": validations}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

