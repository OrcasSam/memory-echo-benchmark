"""Merge matched A-MEM and sharded Mem0 lifecycle runs with strict validation."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[3] / ".matplotlib"))

from memory_echo_lifecycle.aggregator import aggregate, write_csv
from memory_echo_lifecycle.figures import write_figures


CONDITIONS = ("control", "repetition", "update")
METRICS = ("ED", "RRI", "WEP", "DCE")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def metric_value(run, metric):
    if metric == "ED":
        return run["metrics"]["ED"]["retrieval_practice"]
    return run["metrics"][metric]


def scenario_payload(sample):
    return {key: value for key, value in sample.items() if key != "runs"}


def validate_run(run, framework):
    if run["framework"] != framework:
        raise ValueError(f"Expected {framework}, found {run['framework']}")
    delays = [point["delay"] for point in run["delay_series"]]
    if delays != list(range(1, 31)):
        raise ValueError("Every run must contain consecutive delay points 1..30")
    if run["condition"] not in CONDITIONS:
        raise ValueError(f"Unknown condition: {run['condition']}")
    for metric in METRICS:
        value = metric_value(run, metric)
        if value is None or not -1e-6 <= value <= 1.0 + 1e-6:
            raise ValueError(f"Illegal {metric}={value}")


def bootstrap_summary(rows, conditions=CONDITIONS, draws=10000, seed=20260821,
                      value_suffix="mem0_minus_amem"):
    rng = np.random.default_rng(seed)
    output = []
    for condition in conditions:
        condition_rows = [row for row in rows if row["condition"] == condition]
        for metric in METRICS:
            values = np.asarray(
                [row[f"{metric}_{value_suffix}"] for row in condition_rows], dtype=float
            )
            indices = rng.integers(0, len(values), size=(draws, len(values)))
            boot = values[indices].mean(axis=1)
            output.append({
                "condition": condition,
                "metric": metric,
                "n": len(values),
                f"mean_{value_suffix}": float(values.mean()),
                f"median_{value_suffix}": float(np.median(values)),
                "ci95_low": float(np.quantile(boot, 0.025)),
                "ci95_high": float(np.quantile(boot, 0.975)),
                "bootstrap_draws": draws,
            })
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amem_dir", required=True, type=Path)
    parser.add_argument("--mem0_dirs", required=True, type=Path, nargs="+")
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()

    amem_samples = read_json(args.amem_dir / "sample_level.json")
    if len(amem_samples) != 200:
        raise ValueError(f"Expected 200 A-MEM samples, found {len(amem_samples)}")
    amem_by_id = {sample["sample_id"]: sample for sample in amem_samples}
    if len(amem_by_id) != 200:
        raise ValueError("Duplicate A-MEM sample IDs")

    mem0_samples = []
    for directory in args.mem0_dirs:
        mem0_samples.extend(read_json(directory / "sample_level.json"))
    mem0_by_id = {sample["sample_id"]: sample for sample in mem0_samples}
    if len(mem0_samples) != 200 or len(mem0_by_id) != 200:
        raise ValueError("Mem0 shards must contain exactly 200 unique samples")
    if set(amem_by_id) != set(mem0_by_id):
        raise ValueError("A-MEM and Mem0 sample IDs differ")

    combined = []
    framework_differences = []
    framework_echo_effect_differences = []
    for sample_id in sorted(amem_by_id):
        amem = amem_by_id[sample_id]
        mem0 = mem0_by_id[sample_id]
        if scenario_payload(amem) != scenario_payload(mem0):
            raise ValueError(f"Scenario/QA mismatch for {sample_id}")
        if len(amem["runs"]) != 3 or len(mem0["runs"]) != 3:
            raise ValueError(f"Expected three condition runs per framework for {sample_id}")
        for run in amem["runs"]: validate_run(run, "A-MEM")
        for run in mem0["runs"]: validate_run(run, "Mem0")
        amem_runs = {run["condition"]: run for run in amem["runs"]}
        mem0_runs = {run["condition"]: run for run in mem0["runs"]}
        if set(amem_runs) != set(CONDITIONS) or set(mem0_runs) != set(CONDITIONS):
            raise ValueError(f"Missing condition for {sample_id}")
        for condition in CONDITIONS:
            row = {"sample_id": sample_id, "condition": condition}
            for metric in METRICS:
                av = metric_value(amem_runs[condition], metric)
                mv = metric_value(mem0_runs[condition], metric)
                row[f"{metric}_amem"] = av
                row[f"{metric}_mem0"] = mv
                row[f"{metric}_mem0_minus_amem"] = mv - av
            framework_differences.append(row)
        for condition in ("repetition", "update"):
            row = {"sample_id": sample_id, "condition": condition}
            for metric in METRICS:
                amem_effect = (metric_value(amem_runs[condition], metric)
                               - metric_value(amem_runs["control"], metric))
                mem0_effect = (metric_value(mem0_runs[condition], metric)
                               - metric_value(mem0_runs["control"], metric))
                row[f"{metric}_amem_effect"] = amem_effect
                row[f"{metric}_mem0_effect"] = mem0_effect
                row[f"{metric}_interaction_mem0_minus_amem"] = mem0_effect - amem_effect
            framework_echo_effect_differences.append(row)
        combined.append({**scenario_payload(amem), "runs": amem["runs"] + mem0["runs"]})

    aggregates = aggregate(combined)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "sample_level.json", combined)
    write_json(output / "aggregate_results.json", aggregates)
    write_csv(output / "aggregate_results.csv", aggregates["condition_summary"])
    write_csv(output / "per_stage_summary.csv", aggregates["per_stage_summary"])
    write_csv(output / "per_qa_type_summary.csv", aggregates["per_qa_type_summary"])
    write_csv(output / "per_delay_summary.csv", aggregates["per_delay_summary"])
    write_csv(output / "per_source_summary.csv", aggregates["per_source_summary"])
    write_csv(output / "per_ability_summary.csv", aggregates["per_ability_summary"])
    write_csv(output / "paired_effects.csv", aggregates["paired_effects"])
    write_csv(output / "framework_paired_differences.csv", framework_differences)
    framework_summary = bootstrap_summary(framework_differences)
    write_csv(output / "framework_paired_summary.csv", framework_summary)
    write_json(output / "framework_paired_summary.json", framework_summary)
    write_csv(output / "framework_echo_effect_differences.csv",
              framework_echo_effect_differences)
    interaction_summary = bootstrap_summary(
        framework_echo_effect_differences,
        conditions=("repetition", "update"),
        value_suffix="interaction_mem0_minus_amem",
    )
    write_csv(output / "framework_echo_effect_summary.csv", interaction_summary)
    write_json(output / "framework_echo_effect_summary.json", interaction_summary)
    shutil.copy2(args.amem_dir / "rewritten_dataset.json", output / "rewritten_dataset.json")
    frozen_dataset_bytes = (args.amem_dir / "rewritten_dataset.json").read_bytes()
    write_json(output / "dataset_manifest.json", {
        "benchmark": "Matched A-MEM vs official Mem0 Memory Echo Lifecycle Benchmark",
        "samples": 200,
        "runs": 1200,
        "delay_observations": 36000,
        "frameworks": ["A-MEM", "Mem0"],
        "conditions": list(CONDITIONS),
        "delay_points": list(range(1, 31)),
        "pairing_key": "sample_id × condition",
        "framework_difference_direction": "Mem0 minus A-MEM",
        "bootstrap_draws": 10000,
        "rewritten_dataset_sha256": hashlib.sha256(frozen_dataset_bytes).hexdigest(),
        "input_equivalence_verified": True,
        "mem0ai_version": importlib.metadata.version("mem0ai"),
        "mem0_execution": {
            "implementation": "official OSS mem0.Memory",
            "methods": ["Memory.add", "Memory.search", "Memory.update"],
            "infer": False,
            "vector_store": "local Qdrant",
            "embedder": "sentence-transformers/all-MiniLM-L6-v2, 384 dimensions",
        },
        "amem_source": str(args.amem_dir),
        "mem0_sources": [str(path) for path in args.mem0_dirs],
    })
    write_figures(output, aggregates, combined)
    condition_rows = {
        (row["framework"], row["condition"]): row
        for row in aggregates["condition_summary"]
    }
    lines = [
        "Matched A-MEM vs official Mem0 experiment completed.",
        "All comparisons use identical sample_id, QA plan, condition, top-k, and delays 1..30.",
        "Framework sensitivity must be read from framework_echo_effect_summary.csv (difference-in-differences),",
        "not from raw framework means alone.",
    ]
    for framework in ("A-MEM", "Mem0"):
        control = condition_rows[(framework, "control")]
        for condition in ("repetition", "update"):
            row = condition_rows[(framework, condition)]
            effects = ", ".join(
                f"{metric}={row[metric] - control[metric]:.6f}" for metric in METRICS
            )
            lines.append(f"{framework} {condition}-control: {effects}")
    (output / "conclusion.txt").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output), "samples": len(combined),
        "runs": sum(len(sample["runs"]) for sample in combined),
        "framework_paired_rows": len(framework_differences),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
