"""Paired GPT-5.6 ablation of raw model answers vs A-v3 answer constraint.

This experiment does not regenerate answers.  It takes the locked cross-model
predictions, exposes both the preserved raw answer and the final constrained
answer, and evaluates both with the same blinded two-pass judge.  Identical
judge requests can be deduplicated by ``run_deduplicated_judge.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from memory_echo_multimodel.providers import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "results_memory_echo_cross_model_gpt56_valid"
DEFAULT_OUTPUT = ROOT / "results_memory_echo_raw_constraint_ablation"
SEED = 20260930


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare(source: Path, output: Path) -> dict[str, Any]:
    rows = [r for r in read_jsonl(source / "all_predictions.jsonl") if r["solution"] == "A"]
    expected = 190 * 2 * 3 * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} locked A rows, found {len(rows)}")
    predictions: list[dict[str, Any]] = []
    changed = defaultdict(int)
    for row in rows:
        raw_answer = str(row.get("raw_freeform_answer") or row["freeform_answer"]).strip()
        final_answer = str(row["freeform_answer"]).strip()
        original_id = row["evaluation_id"]
        for variant, answer in (("raw", raw_answer), ("constraint", final_answer)):
            item = dict(row)
            item["original_evaluation_id"] = original_id
            item["answer_variant"] = variant
            item["evaluation_id"] = f"{variant}|{original_id}"
            item["freeform_answer"] = answer
            predictions.append(item)
        changed[row["generator_provider"]] += int(raw_answer != final_answer)
    write_jsonl(output / "all_predictions.jsonl", predictions)
    manifest = {
        "protocol": "locked_raw_vs_authority_constraint_v1",
        "source": str(source),
        "locked_samples": 190,
        "solutions": ["A"],
        "generators": ["deepseek", "kimi"],
        "frameworks": ["A-MEM", "Mem0"],
        "conditions": ["control", "repetition", "update"],
        "answer_variants": ["raw", "constraint"],
        "rows": len(predictions),
        "changed_answers": dict(changed),
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def bootstrap_ci(values: list[float], seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(n_boot, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def aggregate(output: Path) -> dict[str, Any]:
    rows = read_jsonl(output / "judged_answers.jsonl")
    expected = 190 * 2 * 3 * 2 * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} judged rows, found {len(rows)}")

    metrics = (
        "DCE_gpt56", "judge_correctness_score", "judge_evidence_faithfulness",
        "judge_echo_contaminated", "judge_contamination_severity",
    )
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["generator_provider"], row["framework"], row["condition"], row["answer_variant"])].append(row)
    summary: list[dict[str, Any]] = []
    for key, subset in sorted(grouped.items()):
        record: dict[str, Any] = {
            "generator": key[0], "framework": key[1], "condition": key[2],
            "answer_variant": key[3], "n": len(subset),
        }
        for metric in metrics:
            record[metric] = float(np.mean([float(r[metric]) for r in subset]))
        summary.append(record)
    write_csv(output / "aggregate_by_variant.csv", summary)

    paired_lookup = {
        (r["generator_provider"], r["framework"], r["condition"], r["sample_id"], r["answer_variant"]): r
        for r in rows
    }
    effects: list[dict[str, Any]] = []
    groups = sorted({key[:3] for key in paired_lookup})
    for index, (generator, framework, condition) in enumerate(groups):
        sample_ids = sorted({key[3] for key in paired_lookup if key[:3] == (generator, framework, condition)})
        for metric in metrics:
            deltas = []
            for sample_id in sample_ids:
                raw = paired_lookup[(generator, framework, condition, sample_id, "raw")]
                constrained = paired_lookup[(generator, framework, condition, sample_id, "constraint")]
                deltas.append(float(constrained[metric]) - float(raw[metric]))
            low, high = bootstrap_ci(deltas, SEED + index * 10 + metrics.index(metric))
            effects.append({
                "generator": generator, "framework": framework, "condition": condition,
                "metric": metric, "paired_samples": len(deltas),
                "mean_constraint_minus_raw": float(np.mean(deltas)),
                "ci95_low": low, "ci95_high": high,
                "constraint_improves_sample_rate": float(np.mean(np.asarray(deltas) < 0))
                if metric in {"DCE_gpt56", "judge_echo_contaminated", "judge_contamination_severity"}
                else float(np.mean(np.asarray(deltas) > 0)),
            })
    write_csv(output / "paired_variant_effects.csv", effects)
    result = {"aggregate": summary, "paired_effects": effects}
    (output / "aggregate_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("prepare", "aggregate", "all"), default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase in {"prepare", "all"}:
        print(json.dumps(prepare(args.source, args.output), ensure_ascii=False), flush=True)
    if args.phase in {"aggregate", "all"}:
        aggregate(args.output)
        print("aggregate_complete", flush=True)


if __name__ == "__main__":
    main()
