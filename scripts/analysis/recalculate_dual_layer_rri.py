"""Recalculate the unified dual-layer RRI for Baseline and solutions A/B/C/D.

The locked replay does not mutate the underlying memory store for each
solution.  Therefore every solution is evaluated against the same matched
no-solution storage state.  The solution-specific legacy RRI is retained as
Serving-RRI, and the new benchmark RRI is:

    RRI* = 0.5 * Storage-RRI + 0.5 * Serving-RRI

This prevents a hard serving gate from being misreported as complete memory
reconsolidation while keeping all methods on one formula and one sample set.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_memory_echo_dce_v3_all"
OUTPUT = ROOT / "results_memory_echo_rri_star"
LOCKED = SOURCE / "paired_sample_locked_190.csv"
BASELINE = SOURCE / "baseline_before_dce_v3.jsonl"
SOLUTION_FILES = {
    solution: SOURCE / f"solution_{solution}_after_dce_v3.jsonl"
    for solution in "ABCD"
}
CASES = SOURCE / "stage_cases_all_dce_v3_enriched.json"

FRAMEWORKS = ("A-MEM", "Mem0")
CONDITIONS = ("control", "repetition", "update")
METRICS = ("ED", "RRI_star", "WEP", "DCE_v3")
EPS = 1e-12


def read_jsonl(path: Path) -> dict[tuple[str, str, str], dict]:
    rows = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["sample_id"], row["framework"], row["condition"])
            if key in rows:
                raise RuntimeError(f"duplicate key in {path.name}: {key}")
            rows[key] = row
    return rows


def mean(values):
    values = [float(v) for v in values]
    return sum(values) / len(values)


def bootstrap_ci(values, seed=20260831, iterations=5000):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    rng = np.random.default_rng(seed)
    means = np.empty(iterations, dtype=float)
    for index in range(iterations):
        means[index] = float(np.mean(rng.choice(values, size=values.size, replace=True)))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def write_csv(path: Path, rows: list[dict], fields: list[str]):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with LOCKED.open("r", encoding="utf-8", newline="") as handle:
        locked_ids = {row["sample_id"] for row in csv.DictReader(handle)}
    if len(locked_ids) != 190:
        raise RuntimeError(f"expected 190 locked samples, found {len(locked_ids)}")

    baseline = read_jsonl(BASELINE)
    solutions = {name: read_jsonl(path) for name, path in SOLUTION_FILES.items()}
    sample_rows = []
    index = {}

    for sample_id in sorted(locked_ids):
        for framework in FRAMEWORKS:
            for condition in CONDITIONS:
                key = (sample_id, framework, condition)
                before = baseline[key]
                storage_rri = float(before["RRI"])

                baseline_row = {
                    "sample_id": sample_id,
                    "dataset": before["dataset"],
                    "ability": before["ability"],
                    "framework": framework,
                    "condition": condition,
                    "solution": "Baseline",
                    "ED": float(before["ED"]),
                    "RRI_storage": storage_rri,
                    "RRI_serving": storage_rri,
                    "RRI_star": storage_rri,
                    "WEP": float(before["WEP"]),
                    "DCE_v3": float(before["DCE_v3"]),
                    "contaminated": bool(before["contaminated"]),
                    "gold_preferred": bool(before["gold_preferred"]),
                }
                sample_rows.append(baseline_row)
                index[("Baseline",) + key] = baseline_row

                for solution in "ABCD":
                    after = solutions[solution][key]
                    serving_rri = float(after["RRI"])
                    rri_star = 0.5 * storage_rri + 0.5 * serving_rri
                    row = {
                        "sample_id": sample_id,
                        "dataset": after["dataset"],
                        "ability": after["ability"],
                        "framework": framework,
                        "condition": condition,
                        "solution": solution,
                        "ED": float(after["ED"]),
                        "RRI_storage": storage_rri,
                        "RRI_serving": serving_rri,
                        "RRI_star": rri_star,
                        "WEP": float(after["WEP"]),
                        "DCE_v3": float(after["DCE_v3"]),
                        "contaminated": bool(after["contaminated"]),
                        "gold_preferred": bool(after["gold_preferred"]),
                    }
                    sample_rows.append(row)
                    index[(solution,) + key] = row

    fields = [
        "sample_id", "dataset", "ability", "framework", "condition", "solution",
        "ED", "RRI_storage", "RRI_serving", "RRI_star", "WEP", "DCE_v3",
        "contaminated", "gold_preferred",
    ]
    write_csv(OUTPUT / "sample_level_rri_star_locked_190.csv", sample_rows, fields)

    groups = defaultdict(list)
    for row in sample_rows:
        groups[(row["solution"], row["framework"], row["condition"])].append(row)
    aggregate = []
    for solution in ("Baseline", "A", "B", "C", "D"):
        for framework in FRAMEWORKS:
            for condition in CONDITIONS:
                rows = groups[(solution, framework, condition)]
                aggregate.append({
                    "solution": solution,
                    "framework": framework,
                    "condition": condition,
                    "n": len(rows),
                    "ED": mean(r["ED"] for r in rows),
                    "RRI_storage": mean(r["RRI_storage"] for r in rows),
                    "RRI_serving": mean(r["RRI_serving"] for r in rows),
                    "RRI_star": mean(r["RRI_star"] for r in rows),
                    "WEP": mean(r["WEP"] for r in rows),
                    "DCE_v3": mean(r["DCE_v3"] for r in rows),
                    "contamination_rate": mean(float(r["contaminated"]) for r in rows),
                    "gold_preference_rate": mean(float(r["gold_preferred"]) for r in rows),
                })
    agg_fields = [
        "solution", "framework", "condition", "n", "ED", "RRI_storage",
        "RRI_serving", "RRI_star", "WEP", "DCE_v3", "contamination_rate",
        "gold_preference_rate",
    ]
    write_csv(OUTPUT / "aggregate_rri_star_locked_190.csv", aggregate, agg_fields)

    echo_summary = []
    for solution in ("Baseline", "A", "B", "C", "D"):
        rows = [r for r in sample_rows
                if r["solution"] == solution and r["condition"] in {"repetition", "update"}]
        echo_summary.append({
            "solution": solution,
            "n": len(rows),
            "ED": mean(r["ED"] for r in rows),
            "RRI_storage": mean(r["RRI_storage"] for r in rows),
            "RRI_serving": mean(r["RRI_serving"] for r in rows),
            "RRI_star": mean(r["RRI_star"] for r in rows),
            "WEP": mean(r["WEP"] for r in rows),
            "DCE_v3": mean(r["DCE_v3"] for r in rows),
            "contamination_rate": mean(float(r["contaminated"]) for r in rows),
            "gold_preference_rate": mean(float(r["gold_preferred"]) for r in rows),
        })
    write_csv(OUTPUT / "echo_summary_rri_star_locked_190.csv", echo_summary,
              ["solution", "n", "ED", "RRI_storage", "RRI_serving", "RRI_star",
               "WEP", "DCE_v3", "contamination_rate", "gold_preference_rate"])

    paired = []
    for solution in "ABCD":
        for framework in FRAMEWORKS:
            for condition in CONDITIONS:
                for metric in METRICS:
                    differences = []
                    for sample_id in sorted(locked_ids):
                        before = index[("Baseline", sample_id, framework, condition)][metric]
                        after = index[(solution, sample_id, framework, condition)][metric]
                        differences.append(float(after) - float(before))
                    low, high = bootstrap_ci(
                        differences,
                        seed=20260831 + ord(solution) + CONDITIONS.index(condition),
                    )
                    paired.append({
                        "solution": solution,
                        "framework": framework,
                        "condition": condition,
                        "metric": metric,
                        "n": len(differences),
                        "mean_after_minus_before": mean(differences),
                        "ci95_low": low,
                        "ci95_high": high,
                    })
    write_csv(OUTPUT / "paired_effects_rri_star_locked_190.csv", paired,
              ["solution", "framework", "condition", "metric", "n",
               "mean_after_minus_before", "ci95_low", "ci95_high"])

    # A-vs-comparator inference uses a cluster bootstrap over source samples;
    # the four repeated observations (2 frameworks x 2 Echo conditions) are
    # averaged within each sample before resampling.
    comparisons = []
    for comparator in "BCD":
        for metric in METRICS:
            sample_differences = []
            for sample_id in sorted(locked_ids):
                cells = []
                for framework in FRAMEWORKS:
                    for condition in ("repetition", "update"):
                        left = index[("A", sample_id, framework, condition)][metric]
                        right = index[(comparator, sample_id, framework, condition)][metric]
                        cells.append(float(left) - float(right))
                sample_differences.append(mean(cells))
            low, high = bootstrap_ci(
                sample_differences,
                seed=20260901 + ord(comparator) + METRICS.index(metric),
            )
            comparisons.append({
                "comparison": f"A_vs_{comparator}",
                "metric": metric,
                "cluster_n": len(sample_differences),
                "delta_A_minus_comparator": mean(sample_differences),
                "ci95_low": low,
                "ci95_high": high,
            })
    write_csv(OUTPUT / "paired_A_vs_BCD_rri_star.csv", comparisons,
              ["comparison", "metric", "cluster_n", "delta_A_minus_comparator",
               "ci95_low", "ci95_high"])

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    for case in cases:
        key = (case["sample_id"], case["framework"], case["condition"])
        before_source = baseline[key]
        after_source = solutions[case["solution"]][key]
        storage_rri = float(before_source["RRI"])
        serving_rri = float(after_source["RRI"])
        rri_star = 0.5 * storage_rri + 0.5 * serving_rri
        case["RRI"] = rri_star
        case["RRI_storage"] = storage_rri
        case["RRI_serving"] = serving_rri
        case.setdefault("before", {})["RRI"] = storage_rri
        case["before"]["RRI_storage"] = storage_rri
        case["before"]["RRI_serving"] = storage_rri
        case.setdefault("after", {})["RRI"] = rri_star
        case["after"]["RRI_storage"] = storage_rri
        case["after"]["RRI_serving"] = serving_rri
    (OUTPUT / "stage_cases_all_rri_star.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    validation = {
        "formula": "RRI*=0.5*Storage-RRI+0.5*Serving-RRI",
        "storage_protocol": "matched no-solution Baseline memory state for every replay solution",
        "serving_protocol": "solution-specific post-gate top-k RRI",
        "locked_samples": len(locked_ids),
        "sample_rows": len(sample_rows),
        "all_rri_in_unit_interval": all(-EPS <= r["RRI_star"] <= 1 + EPS for r in sample_rows),
        "group_n_values": sorted({row["n"] for row in aggregate}),
        "note": "This is a replay-compatible dual-layer RRI; provenance-noise stress testing is not included.",
    }
    (OUTPUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(validation, ensure_ascii=False))
    for row in echo_summary:
        print(row["solution"], f"RRI_storage={row['RRI_storage']:.6f}",
              f"RRI_serving={row['RRI_serving']:.6f}", f"RRI_star={row['RRI_star']:.6f}")


if __name__ == "__main__":
    main()
