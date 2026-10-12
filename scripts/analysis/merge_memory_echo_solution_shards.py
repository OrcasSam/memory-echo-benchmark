"""Stream-merge disjoint Solution shards and recompute paired statistics."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from memory_echo_solution.aggregator import (
    causal_estimands_from_condition_effects,
    summarize_utility_effects,
    write_csv,
)
from memory_echo_solution.figures import write_solution_figures
from scripts.legacy.run_memory_echo_solution_validation import (
    _json, _protocol_checks, _strict_validation, decision_challenge_analysis_from_effects,
    probe_analysis_from_effects,
)

ROOT = Path(__file__).resolve().parents[2]


def _weighted_merge(rows, keys):
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    merged = []
    for key_values, group in sorted(grouped.items()):
        result = dict(zip(keys, key_values))
        result["n"] = sum(int(row["n"]) for row in group)
        value_keys = [key for key in group[0] if key not in {*keys, "n"}]
        for value_key in value_keys:
            available = [(row.get(value_key), int(row["n"])) for row in group
                         if row.get(value_key) is not None]
            result[value_key] = (None if not available else
                sum(float(value) * n for value, n in available) /
                sum(n for _, n in available))
        merged.append(result)
    return merged


def _copy_array_payload(source: Path, target, prepend_comma: bool):
    """Copy a JSON-array payload without materializing it in memory."""
    size = source.stat().st_size
    with source.open("rb") as handle:
        start = 0
        while start < size:
            handle.seek(start)
            if handle.read(1) not in b" \t\r\n[":
                break
            start += 1
        end = size - 1
        while end >= 0:
            handle.seek(end)
            if handle.read(1) not in b" \t\r\n]":
                break
            end -= 1
        if end < start:
            return False
        if prepend_comma:
            target.write(b",")
        handle.seek(start)
        remaining = end - start + 1
        while remaining:
            chunk = handle.read(min(8 * 1024 * 1024, remaining))
            if not chunk:
                raise IOError(f"Unexpected EOF while merging {source}")
            target.write(chunk)
            remaining -= len(chunk)
        return True


def _stream_concat_sample_arrays(shard_dirs, output_file):
    with output_file.open("wb") as target:
        target.write(b"[")
        wrote = False
        for folder in shard_dirs:
            wrote_now = _copy_array_payload(folder / "sample_level.json", target, wrote)
            wrote = wrote or wrote_now
        target.write(b"]")


def _combine_cases(shard_dirs, count=12):
    cases = []
    for folder in shard_dirs:
        cases.extend(json.loads((folder / "representative_cases.json").read_text(encoding="utf-8")))
    def improvement(row):
        return row["baseline"]["metrics"]["DCE"] - row["full_guard"]["metrics"]["DCE"]
    return sorted(cases, key=improvement, reverse=True)[:count]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("shards", nargs="+")
    parser.add_argument("--output_dir", default="results_memory_echo_solution_validation_200")
    args = parser.parse_args()
    shard_dirs, shard_aggregates, manifests = [], [], []
    for shard in args.shards:
        folder = Path(shard)
        if not folder.is_absolute():
            folder = ROOT / folder
        required = ("sample_level.json", "aggregate_results.json", "dataset_manifest.json",
                    "representative_cases.json")
        missing = [name for name in required if not (folder / name).exists()]
        if missing:
            raise FileNotFoundError(f"{folder}: missing {missing}")
        shard_dirs.append(folder)
        shard_aggregates.append(json.loads(
            (folder / "aggregate_results.json").read_text(encoding="utf-8")))
        manifests.append(json.loads(
            (folder / "dataset_manifest.json").read_text(encoding="utf-8")))

    condition_summary = _weighted_merge(
        [r for a in shard_aggregates for r in a["condition_summary"]],
        ("framework", "solution_arm", "condition"))
    stage_summary = _weighted_merge(
        [r for a in shard_aggregates for r in a["per_stage_summary"]],
        ("framework", "solution_arm", "condition", "stage"))
    qa_summary = _weighted_merge(
        [r for a in shard_aggregates for r in a["per_qa_type_summary"]],
        ("framework", "solution_arm", "condition", "qa_type"))
    ability_summary = _weighted_merge(
        [r for a in shard_aggregates for r in a["per_ability_summary"]],
        ("framework", "solution_arm", "condition", "source_dataset", "source_ability"))
    delay_summary = _weighted_merge(
        [r for a in shard_aggregates for r in a["per_delay_summary"]],
        ("framework", "solution_arm", "condition", "delay"))
    condition_effects = [r for a in shard_aggregates for r in a["condition_effects"]]
    sample_ids = {r["sample_id"] for r in condition_effects}
    if len(sample_ids) != sum(m["samples"] for m in manifests):
        raise ValueError("sample IDs overlap across shards or effects are incomplete")
    framework_count = len(manifests[0]["frameworks"])
    arm_count = len(manifests[0]["solution_arms"])
    expected_effect_rows = len(sample_ids) * framework_count * arm_count * 2
    if len(condition_effects) != expected_effect_rows:
        raise ValueError(f"expected {expected_effect_rows} condition effects, got {len(condition_effects)}")

    causal = causal_estimands_from_condition_effects(condition_effects)
    utility_effects = [r for a in shard_aggregates for r in a["utility_effects"]]
    utility = summarize_utility_effects(utility_effects)
    probe_summary = _weighted_merge(
        [r for a in shard_aggregates for r in a.get("probe_summary", [])],
        ("framework", "probe"))
    probe_effects = [r for a in shard_aggregates for r in a.get("probe_effects", [])]
    probe_contrasts = probe_analysis_from_effects(probe_effects)
    decision_challenge_effects = [
        r for a in shard_aggregates for r in a.get("decision_challenge_effects", [])
    ]
    decision_challenge_summary = decision_challenge_analysis_from_effects(
        decision_challenge_effects
    )
    aggregates = {
        "condition_summary": condition_summary,
        "per_stage_summary": stage_summary,
        "per_qa_type_summary": qa_summary,
        "per_ability_summary": ability_summary,
        "per_delay_summary": delay_summary,
        "condition_effects": condition_effects,
        "causal_estimands": causal,
        "utility_effects": utility_effects,
        "utility_noninferiority": utility,
        "probe_summary": probe_summary,
        "probe_effects": probe_effects,
        "probe_contrasts": probe_contrasts,
        "decision_challenge_effects": decision_challenge_effects,
        "decision_challenge_summary": decision_challenge_summary,
    }
    checks = _protocol_checks(aggregates)
    validation = _strict_validation(aggregates)
    aggregates["protocol_checks"] = checks
    aggregates["validation_summary"] = validation

    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **manifests[0], "samples": len(sample_ids),
        "sample_range": [min(m["sample_range"][0] for m in manifests),
                         max(m["sample_range"][1] for m in manifests)],
        "merged_shards": [str(folder) for folder in shard_dirs],
        "merge_mode": "streamed sample array + recomputed compact paired estimands",
    }
    _json(output / "dataset_manifest.json", manifest)
    _stream_concat_sample_arrays(shard_dirs, output / "sample_level.json")
    _json(output / "aggregate_results.json", aggregates)
    _json(output / "representative_cases.json", _combine_cases(shard_dirs))
    for name in ("condition_summary", "per_stage_summary", "per_qa_type_summary",
                 "per_ability_summary", "per_delay_summary", "condition_effects",
                 "causal_estimands", "utility_effects", "utility_noninferiority",
                 "validation_summary", "protocol_checks", "probe_summary",
                 "probe_contrasts", "probe_effects"):
        write_csv(output / f"{name}.csv", aggregates[name])
    for name in ("decision_challenge_summary", "decision_challenge_effects"):
        write_csv(output / f"{name}.csv", aggregates[name])
    write_solution_figures(output, aggregates)
    passed = sum(r["target_met"] for r in validation)
    utility_pass = sum(r["noninferior"] for r in utility)
    check_pass = sum(r["passed"] for r in checks)
    probe_pass = sum(r["passed"] for r in probe_contrasts)
    decision_pass = sum(r["passed"] for r in decision_challenge_summary)
    (output / "conclusion.txt").write_text(
        "Strict Memory Echo lifecycle solution validation\n"
        f"Pre-registered metric targets: {passed}/{len(validation)}.\n"
        f"Control utility non-inferiority: {utility_pass}/{len(utility)}.\n"
        f"Protocol checks: {check_pass}/{len(checks)}.\n"
        f"Positive/negative probe contrasts: {probe_pass}/{len(probe_contrasts)}.\n"
        f"Counterfactual decision challenges: {decision_pass}/{len(decision_challenge_summary)}.\n",
        encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "samples": len(sample_ids),
                      "runs": len(sample_ids) * framework_count * 3 * arm_count},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
