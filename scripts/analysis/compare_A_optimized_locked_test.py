"""Locked-test paired comparison of optimized A against frozen B/C/D.

The first 10 streamed source samples were used for implementation smoke checks
and are excluded.  All reported claims are computed on the remaining 190
source samples with matched framework/condition keys.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import ijson
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_memory_echo_solution_A_optimized_comparison"
SOURCE = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
DIRS = {
    "A": ROOT / "results_memory_echo_solution_A_optimized_top5",
    "B": ROOT / "results_memory_echo_solution_B_top5",
    "C": ROOT / "results_memory_echo_solution_C_top5",
    "D": ROOT / "results_memory_echo_solution_D_top5",
}
METRICS = ("ED", "RRI", "WEP", "DCE")
SEED = 20260826
BOOTSTRAPS = 10_000


def jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with SOURCE.open("rb") as handle:
        items = ijson.items(handle, "item")
        development_ids = [next(items)["sample_id"] for _ in range(10)]
    development_ids = set(development_ids)

    data = {}
    for solution, directory in DIRS.items():
        rows = jsonl(directory / "sample_level_decisions.jsonl")
        data[solution] = {
            (r["sample_id"], r["framework"], r["condition"]): r
            for r in rows if r["sample_id"] not in development_ids
        }
        if len(data[solution]) != 190 * 2 * 3:
            raise RuntimeError(f"{solution}: expected 1140 locked rows, got {len(data[solution])}")

    keys = set(data["A"])
    if any(set(data[s]) != keys for s in "BCD"):
        raise RuntimeError("A/B/C/D locked-test keys are not identical")

    aggregate = []
    for solution in "ABCD":
        for framework in ("A-MEM", "Mem0"):
            for condition in ("control", "repetition", "update"):
                subset = [
                    data[solution][k] for k in sorted(keys)
                    if k[1] == framework and k[2] == condition
                ]
                row = {
                    "solution": solution, "framework": framework,
                    "condition": condition, "n": len(subset),
                }
                for metric in METRICS:
                    row[metric] = float(np.mean([r[metric] for r in subset]))
                row["contamination_rate"] = float(np.mean([r["contaminated"] for r in subset]))
                row["gold_preference_rate"] = float(np.mean([r["gold_preferred"] for r in subset]))
                aggregate.append(row)

    rng = np.random.default_rng(SEED)
    echo_keys = sorted(k for k in keys if k[2] in {"repetition", "update"})
    paired = []
    for comparator in "BCD":
        for metric in METRICS:
            a = np.asarray([data["A"][k][metric] for k in echo_keys], dtype=float)
            b = np.asarray([data[comparator][k][metric] for k in echo_keys], dtype=float)
            delta = a - b
            epsilon = 1e-6
            non_tie = np.abs(delta) > epsilon
            indices = rng.integers(0, len(delta), size=(BOOTSTRAPS, len(delta)))
            boot = delta[indices].mean(axis=1)
            paired.append({
                "comparison": f"A_vs_{comparator}", "metric": metric,
                "scope": "locked_190_repetition_plus_update",
                "paired_n": len(delta), "A_mean": float(a.mean()),
                "comparator_mean": float(b.mean()),
                "delta_A_minus_comparator": float(delta.mean()),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "A_lower_rate_all": float(np.mean(delta < -epsilon)),
                "near_tie_rate": float(np.mean(np.abs(delta) <= epsilon)),
                "A_lower_rate_non_tie": float(
                    np.mean(delta[non_tie] < 0) if np.any(non_tie) else 0.0
                ),
            })

    echo_summary = []
    for solution in "ABCD":
        subset = [data[solution][k] for k in echo_keys]
        echo_summary.append({
            "solution": solution, "n": len(subset),
            **{metric: float(np.mean([r[metric] for r in subset])) for metric in METRICS},
            "contamination_rate": float(np.mean([r["contaminated"] for r in subset])),
            "gold_preference_rate": float(np.mean([r["gold_preferred"] for r in subset])),
        })

    control_keys = sorted(k for k in keys if k[2] == "control")
    control_summary = []
    for solution in "ABCD":
        subset = [data[solution][k] for k in control_keys]
        control_summary.append({
            "solution": solution, "n": len(subset),
            **{metric: float(np.mean([r[metric] for r in subset])) for metric in METRICS},
            "contamination_rate": float(np.mean([r["contaminated"] for r in subset])),
            "gold_preference_rate": float(np.mean([r["gold_preferred"] for r in subset])),
        })

    validation = {
        "development_samples_excluded": 10,
        "locked_test_samples": 190,
        "runs_per_solution": 1140,
        "echo_condition_pairs_per_solution": len(echo_keys),
        "top_k": 5,
        "frameworks": ["A-MEM", "Mem0"],
        "conditions": ["control", "repetition", "update"],
        "same_model_protocol": "deepseek-chat, temperature=0.0",
        "A_ranking_uses_benchmark_label": False,
        "bootstrap_seed": SEED,
        "bootstrap_replicates": BOOTSTRAPS,
        "passed": True,
    }

    write_csv(OUT / "locked_test_aggregate.csv", aggregate)
    write_csv(OUT / "locked_test_echo_summary.csv", echo_summary)
    write_csv(OUT / "locked_test_control_summary.csv", control_summary)
    write_csv(OUT / "paired_A_vs_BCD.csv", paired)
    (OUT / "validation.json").write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "results.json").write_text(
        json.dumps({"aggregate": aggregate, "echo_summary": echo_summary,
                    "control_summary": control_summary,
                    "paired_comparisons": paired, "validation": validation},
                   ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(echo_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
