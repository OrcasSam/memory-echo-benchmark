"""Validate formal Generic Memory Echo output invariants."""
import argparse
import json
from collections import Counter
from pathlib import Path

from memory_echo_generic import MEMORY_UNIT_TYPES, PROTOCOLS
from memory_echo_generic.generic_trajectory_builder import CONDITIONS


def _close(left, right, tolerance=1e-12):
    if left is None or right is None:
        return left is right
    return abs(float(left) - float(right)) <= tolerance


def validate(output_dir: Path):
    sample_path = output_dir / "sample_level.json"
    raw = sample_path.read_text(encoding="utf-8")
    samples = json.loads(raw)
    protocol_counts = Counter(sample["protocol"] for sample in samples)
    seed_counts = Counter(sample["random_seed"] for sample in samples)
    catch_counts = Counter(
        sample["protocol"] for sample in samples
        if not sample["should_update"]
    )
    qa_total = 0
    condition_run_total = 0
    framework_equal_metric_runs = 0
    framework_equal_all_metric_runs = 0
    zero_writeback_sample_ids = set()
    errors = []

    def check(assertion, message):
        if not assertion:
            errors.append(message)

    check(len(samples) == 400, f"expected 400 samples, got {len(samples)}")
    check(
        all(protocol_counts[protocol] == 100 for protocol in PROTOCOLS),
        f"protocol counts are {dict(protocol_counts)}",
    )
    check(
        {sample["memory_unit_type"] for sample in samples}
        == set(MEMORY_UNIT_TYPES),
        "not all memory unit types are represented",
    )
    check(set(seed_counts) == {11, 29, 47}, f"seed counts are {seed_counts}")
    check('"supplier_id"' not in raw, "legacy supplier_id leaked into output")

    for sample in samples:
        runs = sample["runs"]
        check(len(runs) == 6, f"sample {sample['sample_id']} run count")
        qa_total += sum(len(run["memory_trajectory"]) for run in runs)
        condition_run_total += len(runs)
        for framework in ("A-MEM", "Mem0"):
            framework_runs = {
                run["condition"]: run for run in runs
                if run["framework"] == framework
            }
            check(
                set(framework_runs) == set(CONDITIONS),
                f"sample {sample['sample_id']} {framework} conditions",
            )
            if set(framework_runs) != set(CONDITIONS):
                continue
            counts = {
                run["distinct_memory_count"]
                for run in framework_runs.values()
            }
            check(
                len(counts) == 1,
                f"sample {sample['sample_id']} {framework} capacity mismatch",
            )
            check(
                framework_runs["control"]["exposure_retrieval_count"] == 0,
                f"sample {sample['sample_id']} control exposure",
            )
            check(
                framework_runs["historical_repetition"][
                    "exposure_retrieval_count"
                ] > 0,
                f"sample {sample['sample_id']} repetition exposure",
            )
            check(
                0 <= framework_runs["historical_repetition_update"][
                    "target_writeback_count"
                ] <= sample["writeback_budget"],
                f"sample {sample['sample_id']} target writeback range",
            )
            check(
                (
                    framework_runs["historical_repetition_update"][
                        "target_writeback_count"
                    ]
                    + framework_runs["historical_repetition_update"][
                        "capacity_matching_write_count"
                    ]
                    == sample["writeback_budget"]
                ),
                f"sample {sample['sample_id']} matched write budget",
            )
            if (
                framework_runs["historical_repetition_update"][
                    "target_writeback_count"
                ] == 0
            ):
                zero_writeback_sample_ids.add(sample["sample_id"])
            for condition, run in framework_runs.items():
                check(
                    len(run["memory_trajectory"]) == 20,
                    f"sample {sample['sample_id']} {framework}/{condition} QA",
                )
                check(
                    all(
                        event.get("phase") in (None, "update_adaptation")
                        for event in run["memory_events"]
                        if event.get("kind")
                        == "retrieval_update_feedback"
                    ),
                    f"sample {sample['sample_id']} feedback phase",
                )
                probes = {}
                for turn in run["memory_trajectory"]:
                    if turn.get("persistence_probe_id"):
                        probes[turn["phase"]] = turn["query"]
                check(
                    probes.get("update_adaptation")
                    == probes.get("echo_retention"),
                    f"sample {sample['sample_id']} unmatched EPS probes",
                )
                metrics = run["metrics"]
                for metric in ("ED", "UIR", "DBI"):
                    value = metrics[metric]
                    check(
                        value is None or 0 <= value <= 1,
                        f"sample {sample['sample_id']} {metric} range",
                    )
                check(
                    metrics["EPS"] is None or metrics["EPS"] >= 0,
                    f"sample {sample['sample_id']} EPS range",
                )
                decision_turns = [
                    turn for turn in run["memory_trajectory"]
                    if turn["phase"] == "decision_influence"
                    and turn["qa_type"] == "decision"
                    and turn["decision_bias_eligible"]
                ]
                recomputed_dbi = (
                    sum(
                        turn["decision_bias_flag"]
                        for turn in decision_turns
                    ) / len(decision_turns)
                    if decision_turns else None
                )
                check(
                    _close(metrics["DBI"], recomputed_dbi),
                    f"sample {sample['sample_id']} DBI mismatch",
                )

        by_key = {
            (run["framework"], run["condition"]): run
            for run in runs
        }
        if all(
            _close(
                by_key[("A-MEM", condition)]["metrics"]["ED"],
                by_key[("Mem0", condition)]["metrics"]["ED"],
            )
            for condition in CONDITIONS
        ):
            framework_equal_metric_runs += 1
        if all(
            all(
                _close(
                    by_key[("A-MEM", condition)]["metrics"][metric],
                    by_key[("Mem0", condition)]["metrics"][metric],
                )
                for metric in ("ED", "UIR", "EPS", "DBI")
            )
            for condition in CONDITIONS
        ):
            framework_equal_all_metric_runs += 1

    check(qa_total == 48_000, f"expected 48000 QA, got {qa_total}")
    check(
        condition_run_total == 2_400,
        f"expected 2400 condition runs, got {condition_run_total}",
    )
    figures = sorted((output_dir / "figures").glob("*.png"))
    check(len(figures) >= 4, f"expected >=4 figures, got {len(figures)}")
    report = {
        "valid": not errors,
        "errors": errors,
        "base_trajectory_count": len(samples),
        "condition_trajectory_count": condition_run_total,
        "executed_qa_count": qa_total,
        "protocol_counts": dict(protocol_counts),
        "seed_counts": dict(seed_counts),
        "catch_trial_counts": dict(catch_counts),
        "framework_equal_ED_sample_count": framework_equal_metric_runs,
        "framework_equal_all_four_metrics_sample_count": (
            framework_equal_all_metric_runs
        ),
        "zero_target_writeback_sample_count": len(
            zero_writeback_sample_ids
        ),
        "memory_unit_types": sorted({
            sample["memory_unit_type"] for sample in samples
        }),
        "figure_count": len(figures),
        "legacy_business_field_present": '"supplier_id"' in raw,
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir", default="results_memory_echo_generic"
    )
    args = parser.parse_args()
    output = Path(args.output_dir)
    report = validate(output)
    (output / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
