"""Low-memory structural validation for a full Solution experiment output."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _stream_counts(path, patterns):
    counts = {pattern: 0 for pattern in patterns}
    carry = b""
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            data = carry + block
            for pattern in patterns:
                start = 0
                while True:
                    found = data.find(pattern, start)
                    if found < 0:
                        break
                    # Count cross-boundary matches once, but do not recount a
                    # complete match retained in the overlap buffer.
                    if found + len(pattern) > len(carry):
                        counts[pattern] += 1
                    start = found + 1
            carry = data[-128:]
    return counts


def _csv(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir")
    args = parser.parse_args()
    folder = Path(args.results_dir)
    sample_file = folder / "sample_level.json"
    patterns = (
        b'"qa_plan"', b'"delay_series"', b'"memory_audit"',
        b'"LoCoMo"', b'"LongMemEval"', b'"HotpotQA"', b'"BEAM"',
    )
    counts = _stream_counts(sample_file, patterns)
    condition_effects = _csv(folder / "condition_effects.csv")
    condition_summary = _csv(folder / "condition_summary.csv")
    delay_summary = _csv(folder / "per_delay_summary.csv")
    checks = _csv(folder / "protocol_checks.csv")
    validation = _csv(folder / "validation_summary.csv")
    probes = _csv(folder / "probe_contrasts.csv")
    probe_effects = _csv(folder / "probe_effects.csv")
    decision_challenges = _csv(folder / "decision_challenge_summary.csv")
    decision_effects = _csv(folder / "decision_challenge_effects.csv")
    utility = _csv(folder / "utility_noninferiority.csv")
    with sample_file.open("rb") as handle:
        first = handle.read(1)
        handle.seek(-1, 2)
        last = handle.read(1)
    report = {
        "sample_file_bytes": sample_file.stat().st_size,
        "json_array_closed": first == b"[" and last == b"]",
        "stream_marker_counts": {key.decode(): value for key, value in counts.items()},
        "unique_effect_sample_ids": len({row["sample_id"] for row in condition_effects}),
        "condition_effect_rows": len(condition_effects),
        "condition_summary_rows": len(condition_summary),
        "delay_summary_rows": len(delay_summary),
        "protocol_checks_passed": sum(row["passed"].lower() == "true" for row in checks),
        "protocol_checks_total": len(checks),
        "probe_contrasts_passed": sum(row["passed"].lower() == "true" for row in probes),
        "probe_contrasts_total": len(probes),
        "probe_effect_rows": len(probe_effects),
        "decision_challenges_passed": sum(row["passed"].lower() == "true" for row in decision_challenges),
        "decision_challenges_total": len(decision_challenges),
        "decision_effect_rows": len(decision_effects),
        "utility_noninferior": sum(row["noninferior"].lower() == "true" for row in utility),
        "utility_total": len(utility),
        "targets_met": sum(row["target_met"].lower() == "true" for row in validation),
        "targets_total": len(validation),
    }
    expected = {
        "json_array_closed": True,
        "qa_plan": 200,
        "delay_series": 7200,
        "memory_audit": 8800,
        "unique_effect_sample_ids": 200,
        "condition_effect_rows": 4800,
        "condition_summary_rows": 36,
        "delay_summary_rows": 1080,
        "probe_effect_rows": 2400,
        "decision_effect_rows": 400,
    }
    report["passed"] = bool(
        report["json_array_closed"]
        and counts[b'"qa_plan"'] == expected["qa_plan"]
        and counts[b'"delay_series"'] == expected["delay_series"]
        and counts[b'"memory_audit"'] == expected["memory_audit"]
        and all(counts[f'"{name}"'.encode()] == 50
                for name in ("LoCoMo", "LongMemEval", "HotpotQA", "BEAM"))
        and report["unique_effect_sample_ids"] == expected["unique_effect_sample_ids"]
        and report["condition_effect_rows"] == expected["condition_effect_rows"]
        and report["condition_summary_rows"] == expected["condition_summary_rows"]
        and report["delay_summary_rows"] == expected["delay_summary_rows"]
        and report["probe_effect_rows"] == expected["probe_effect_rows"]
        and report["decision_effect_rows"] == expected["decision_effect_rows"]
        and report["protocol_checks_passed"] == report["protocol_checks_total"]
        and report["probe_contrasts_passed"] == report["probe_contrasts_total"]
        and report["decision_challenges_passed"] == report["decision_challenges_total"]
        and report["utility_noninferior"] == report["utility_total"]
        and report["targets_met"] == report["targets_total"]
    )
    (folder / "integrity_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
