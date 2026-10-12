"""Aggregate ED, UIR, EPS, and DBI at trajectory level."""
import csv
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


METRICS = ("ED", "UIR", "EPS", "DBI")


def _mean(values):
    values = [float(v) for v in values if v is not None and v != ""]
    return sum(values) / len(values) if values else None


def _median(values):
    values = [float(v) for v in values if v is not None and v != ""]
    return statistics.median(values) if values else None


def _bootstrap_ci(values, seed=2027, iterations=500):
    values = [float(v) for v in values if v is not None and v != ""]
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed + len(values))
    means = []
    for _ in range(iterations):
        means.append(sum(rng.choice(values) for _ in values) / len(values))
    means.sort()
    return means[int(0.025 * iterations)], means[int(0.975 * iterations) - 1]


def _run_records(samples: List[Dict]) -> List[Dict]:
    records = []
    for sample in samples:
        for run in sample["runs"]:
            decision_turns = [
                turn for turn in run.get("memory_trajectory", [])
                if turn.get("phase") == "decision_influence"
                and turn.get("qa_type") == "decision"
            ]
            catch_turns = (
                decision_turns if not sample["should_update"] else []
            )
            records.append({
                "sample_id": sample["sample_id"],
                "protocol_sample_id": sample["protocol_sample_id"],
                "protocol": sample["protocol"],
                "memory_unit_type": sample["memory_unit_type"],
                "random_seed": sample["random_seed"],
                "random_seed_replicate_id": sample.get(
                    "random_seed_replicate_id"
                ),
                "evidence_strength": sample["conflicting_evidence"]["strength"],
                "noise_level": sample["noise_level"],
                "should_update": sample["should_update"],
                "framework": run["framework"],
                "condition": run["condition"],
                **run["metrics"],
                "catch_correct_retention_rate": _mean(
                    float(turn.get("predicted_memory_label") == "OLD")
                    for turn in catch_turns
                ),
                "decision_accuracy": _mean(
                    float(
                        turn.get("predicted_memory_label")
                        == turn.get("canonical_target_label")
                    )
                    for turn in decision_turns
                ),
            })
    return records


def _aggregate_records(records: List[Dict], group_fields: Tuple[str, ...]):
    grouped = defaultdict(list)
    for record in records:
        grouped[tuple(record[field] for field in group_fields)].append(record)
    rows = []
    for key, group in sorted(grouped.items()):
        row = dict(zip(group_fields, key))
        row["trajectory_count"] = len(group)
        row["update_applicable_count"] = sum(r["should_update"] for r in group)
        for metric_index, metric in enumerate(METRICS):
            values = [
                r.get(metric) for r in group
                if not (metric in ("UIR", "DBI") and not r["should_update"])
                and r[metric] is not None
            ]
            low, high = _bootstrap_ci(
                values, seed=2027 + metric_index * 1009
            )
            row[metric] = _mean(values)
            row[f"{metric}_median"] = _median(values)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_valid_n"] = len(values)
        row["semantic_answer_score"] = _mean(
            r.get("semantic_answer_score") for r in group
        )
        row["relevant_retrieval_coverage"] = _mean(
            r.get("relevant_retrieval_coverage") for r in group
        )
        row["empty_retrieval_rate"] = _mean(
            r.get("empty_retrieval_rate") for r in group
        )
        row["decision_accuracy"] = _mean(
            r.get("decision_accuracy") for r in group
        )
        row["catch_correct_retention_rate"] = _mean(
            r.get("catch_correct_retention_rate") for r in group
        )
        rows.append(row)
    return rows


def aggregate_run_metrics(samples: List[Dict]):
    records = _run_records(samples)
    return {
        "condition_summary": _aggregate_records(
            records, ("framework", "condition")
        ),
        "protocol_summary": _aggregate_records(
            records, ("framework", "condition", "protocol")
        ),
        "memory_type_summary": _aggregate_records(
            records, ("framework", "condition", "memory_unit_type")
        ),
        "seed_summary": _aggregate_records(
            records, ("framework", "condition", "random_seed")
        ),
        "evidence_strength_summary": _aggregate_records(
            records, ("framework", "condition", "evidence_strength")
        ),
        "run_records": records,
    }


def aggregate_turn_metrics(samples: List[Dict]):
    phase_groups = defaultdict(list)
    qa_groups = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            for turn in run["memory_trajectory"]:
                base = (
                    run["framework"], run["condition"], sample["protocol"]
                )
                phase_groups[base + (turn["phase"],)].append(
                    (sample, run, turn)
                )
                qa_groups[base + (turn["qa_type"],)].append(
                    (sample, run, turn)
                )

    def make_rows(groups, final_field):
        rows = []
        for key, items in sorted(groups.items()):
            turns = [item[2] for item in items]
            row = {
                "framework": key[0],
                "condition": key[1],
                "protocol": key[2],
                final_field: key[3],
                "qa_count": len(turns),
                "ED": _mean(t["echo_density"] for t in turns),
                "UIR": _mean(
                    t["update_incorporation_rate"] for sample, _, t in items
                    if sample["should_update"]
                    and t["phase"] != "echo_seeding"
                ),
                "DBI": _mean(
                    t["decision_bias_flag"] for t in turns
                    if t["decision_bias_eligible"]
                ),
                "semantic_similarity": _mean(
                    t["answer_evaluation"]["semantic_similarity"]
                    for t in turns
                ),
                "answer_score": _mean(
                    t["answer_evaluation"]["answer_score"] for t in turns
                ),
                "target_match_rate": _mean(
                    t["answer_evaluation"]["target_id_match"] for t in turns
                ),
                "relevant_retrieval_coverage": _mean(
                    t.get("relevant_retrieval_coverage") for t in turns
                ),
            }
            # EPS is a trajectory-level retention metric. It is reported only
            # on the retention phase rather than copied to every phase.
            if final_field == "phase" and key[3] == "echo_retention":
                unique_runs = {}
                for sample, run, _ in items:
                    unique_runs[(sample["sample_id"], run["framework"],
                                 run["condition"])] = run
                row["EPS"] = _mean(
                    run["metrics"].get("EPS")
                    for run in unique_runs.values()
                )
            else:
                row["EPS"] = ""
            rows.append(row)
        return rows

    return {
        "phase_summary": make_rows(phase_groups, "phase"),
        "qa_type_summary": make_rows(qa_groups, "qa_type"),
    }


def _paired_delta(current: Dict, baseline: Dict):
    def delta(metric):
        left, right = current.get(metric), baseline.get(metric)
        return left - right if left is not None and right is not None else None

    return {
        "ED_lift": delta("ED"),
        "UIR_drop": (
            baseline.get("UIR") - current.get("UIR")
            if current["should_update"]
            and baseline.get("UIR") is not None
            and current.get("UIR") is not None else None
        ),
        "EPS_lift": delta("EPS"),
        "DBI_lift": (
            current.get("DBI") - baseline.get("DBI")
            if current["should_update"]
            and current.get("DBI") is not None
            and baseline.get("DBI") is not None else None
        ),
    }


def _aggregate_paired_groups(grouped):
    rows = []
    for key, effects in sorted(grouped.items()):
        framework, protocol, contrast, random_seed = key
        row = {
            "framework": framework,
            "protocol": protocol,
            "contrast": contrast,
            "random_seed": random_seed,
            "paired_trajectory_count": len(effects),
        }
        for i, metric in enumerate(
            ("ED_lift", "UIR_drop", "EPS_lift", "DBI_lift")
        ):
            values = [e[metric] for e in effects if e[metric] is not None]
            low, high = _bootstrap_ci(values, 9001 + i * 503)
            row[metric] = _mean(values)
            row[f"{metric}_ci95_low"] = low
            row[f"{metric}_ci95_high"] = high
            row[f"{metric}_valid_n"] = len(values)
        rows.append(row)
    return rows


def paired_condition_effects(samples: List[Dict]):
    records = _run_records(samples)
    lookup = {
        (
            r["sample_id"], r["framework"], r["condition"]
        ): r for r in records
    }
    comparisons = (
        ("historical_repetition_update", "control", "update_minus_control"),
        ("historical_repetition", "control", "repetition_minus_control"),
        (
            "historical_repetition_update",
            "historical_repetition",
            "update_minus_repetition",
        ),
    )
    grouped = defaultdict(list)
    seed_grouped = defaultdict(list)
    for record in records:
        for current_name, baseline_name, contrast in comparisons:
            if record["condition"] != current_name:
                continue
            baseline = lookup.get(
                (record["sample_id"], record["framework"], baseline_name)
            )
            if baseline is None:
                continue
            effects = _paired_delta(record, baseline)
            grouped[(
                record["framework"], record["protocol"], contrast, ""
            )].append(effects)
            grouped[(
                record["framework"], "ALL_PROTOCOLS", contrast, ""
            )].append(effects)
            seed_grouped[(
                record["framework"], record["protocol"], contrast,
                record["random_seed"],
            )].append(effects)
            seed_grouped[(
                record["framework"], "ALL_PROTOCOLS", contrast,
                record["random_seed"],
            )].append(effects)
    return _aggregate_paired_groups(grouped)


def paired_condition_effects_by_seed(samples: List[Dict]):
    records = _run_records(samples)
    lookup = {
        (r["sample_id"], r["framework"], r["condition"]): r
        for r in records
    }
    comparisons = (
        ("historical_repetition_update", "control", "update_minus_control"),
        ("historical_repetition", "control", "repetition_minus_control"),
        (
            "historical_repetition_update",
            "historical_repetition",
            "update_minus_repetition",
        ),
    )
    grouped = defaultdict(list)
    for record in records:
        for current_name, baseline_name, contrast in comparisons:
            if record["condition"] != current_name:
                continue
            baseline = lookup.get(
                (record["sample_id"], record["framework"], baseline_name)
            )
            if baseline is None:
                continue
            grouped[(
                record["framework"], record["protocol"], contrast,
                record["random_seed"],
            )].append(_paired_delta(record, baseline))
    return _aggregate_paired_groups(grouped)


def write_csv(path: Path, rows: Iterable[Dict]):
    rows = list(rows)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
