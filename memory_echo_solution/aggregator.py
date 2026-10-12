"""Paired causal estimands and uncertainty for Solution validation."""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

from memory_echo_lifecycle import STAGES


METRICS = ("ED", "RRI", "WEP", "DCE")


def _mean(values):
    values = [float(v) for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _run_metric(run, metric):
    if metric == "ED":
        return run["metrics"]["ED"]["retrieval_practice"]
    return run["metrics"][metric]


def _bootstrap_ci(values, seed=20260824, iterations=10000):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if len(values) == 0:
        return None, None
    if np.allclose(values, values[0]):
        return float(values[0]), float(values[0])
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for start in range(0, iterations, 1000):
        n = min(1000, iterations - start)
        draws = rng.integers(0, len(values), size=(n, len(values)))
        means[start:start + n] = values[draws].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def _p_value(values):
    values = np.asarray([v for v in values if v is not None], dtype=float)
    if not len(values) or np.allclose(values, 0):
        return 1.0
    try:
        return float(wilcoxon(values, zero_method="zsplit").pvalue)
    except ValueError:
        return 1.0


def _holm(rows):
    ordered = sorted(enumerate(rows), key=lambda pair: pair[1]["p_value"])
    adjusted = [1.0] * len(rows)
    running = 0.0
    m = len(rows)
    for rank, (index, row) in enumerate(ordered):
        value = min(1.0, (m - rank) * row["p_value"])
        running = max(running, value)
        adjusted[index] = running
    for row, value in zip(rows, adjusted):
        row["p_holm"] = value


def _excess_wep(run, control):
    by_delay = {p["delay"]: p["ED"] for p in control["delay_series"]}
    excess = []
    for point in run["delay_series"]:
        base = by_delay.get(point["delay"])
        if point["ED"] is not None and base is not None:
            excess.append(max(0.0, point["ED"] - base))
    return _mean(excess)


def _delay_value(run, delay=30):
    point = next((p for p in run["delay_series"] if int(p["delay"]) == delay), None)
    return None if point is None else float(point["ED"])


def causal_estimands_from_condition_effects(condition_effects):
    """Compute paired component/full DiD from compact sample-level effects."""
    by_effect = {(r["sample_id"], r["framework"], r["solution_arm"], r["condition"]): r
                 for r in condition_effects}
    sample_ids = sorted({r["sample_id"] for r in condition_effects})
    frameworks = sorted({r["framework"] for r in condition_effects})
    comparisons = (
        ("s1_identity", "baseline", "S1 family identity and provenance"),
        ("s2_retrieval_hygiene", "s1_identity", "S2 retrieval exposure without fact write"),
        ("s3_reconsolidation", "s2_retrieval_hygiene", "S3 independent-write and correction binding"),
        ("s4_persistence", "s3_reconsolidation", "S4 activation decay and window remediation"),
        ("full_guard", "s4_persistence", "Full counterfactual decision gate"),
        ("full_guard", "baseline", "complete lifecycle guard"),
    )
    did_rows = []
    for framework in frameworks:
        for condition in ("repetition", "update"):
            for treated, reference, component in comparisons:
                for metric in (*METRICS, "excess_WEP", "final_accuracy_effect"):
                    key = (metric if metric.endswith("effect") or metric == "excess_WEP"
                           else f"{metric}_effect")
                    diffs = []
                    for sid in sample_ids:
                        left = by_effect.get((sid, framework, treated, condition))
                        right = by_effect.get((sid, framework, reference, condition))
                        if left and right:
                            diffs.append(float(left[key]) - float(right[key]))
                    if not diffs:
                        continue
                    low, high = _bootstrap_ci(diffs)
                    sd = float(np.std(diffs, ddof=1)) if len(diffs) > 1 else 0.0
                    mean = float(np.mean(diffs))
                    did_rows.append({
                        "framework": framework, "condition": condition,
                        "treated_arm": treated, "reference_arm": reference,
                        "component": component, "metric": metric, "n": len(diffs),
                        "DiD": mean, "ci_low": low, "ci_high": high,
                        "cohens_dz": mean / sd if sd > 1e-12 else 0.0,
                        "p_value": _p_value(diffs),
                    })
    _holm(did_rows)
    return did_rows


def summarize_utility_effects(utility_effects):
    rows = []
    grouped = defaultdict(list)
    for row in utility_effects:
        grouped[(row["framework"], row["solution_arm"])].append(
            float(row["control_final_accuracy_delta"])
        )
    for (framework, arm), diffs in sorted(grouped.items()):
        low, high = _bootstrap_ci(diffs)
        rows.append({
            "framework": framework, "solution_arm": arm, "n": len(diffs),
            "control_final_accuracy_delta": _mean(diffs),
            "ci_low": low, "ci_high": high,
            "noninferior_margin": -0.02,
            "noninferior": bool(low is not None and low > -0.02),
        })
    return rows


def aggregate_solution(samples):
    groups = defaultdict(list)
    stage_groups = defaultdict(list)
    qa_groups = defaultdict(list)
    delay_groups = defaultdict(list)
    ability_groups = defaultdict(list)
    for sample in samples:
        for run in sample["runs"]:
            key = (run["framework"], run["solution_arm"], run["condition"])
            groups[key].append(run)
            source = sample["source"]["dataset"]
            ability = sample["source"]["ability"]
            ability_groups[key + (source, ability)].append(run)
            for turn in run["trajectory"]:
                stage_groups[key + (turn["stage"],)].append(turn)
                qa_groups[key + (turn["qa_type"],)].append(turn)
            for point in run["delay_series"]:
                delay_groups[key + (point["delay"],)].append(point)

    condition_summary = []
    for (framework, arm, condition), runs in sorted(groups.items()):
        condition_summary.append({
            "framework": framework, "solution_arm": arm,
            "condition": condition, "n": len(runs),
            **{metric: _mean(_run_metric(run, metric) for run in runs)
               for metric in METRICS},
            "contamination_rate": _mean(run["metrics"]["decision_contamination_rate"]
                                        for run in runs),
            "final_accuracy": _mean(run["metrics"]["final_accuracy"] for run in runs),
            "encoding_accuracy": _mean(run["metrics"]["stage_accuracy"]["encoding"]
                                       for run in runs),
            "executed_writes": _mean(run["memory_audit"]["executed_writes"] for run in runs),
            "blocked_derived_writes": _mean(
                run["memory_audit"]["blocked_derived_writes"] for run in runs),
            "blocked_derived_adds": _mean(
                run["memory_audit"].get("blocked_derived_adds", 0) for run in runs),
            "blocked_derived_updates": _mean(
                run["memory_audit"].get("blocked_derived_updates", 0) for run in runs),
            "family_pair_recall": _mean(
                run["memory_audit"].get("family_pair_recall") for run in runs),
            "provenance_inheritance_accuracy": _mean(
                run["memory_audit"].get("provenance_inheritance_accuracy") for run in runs),
            "max_confidence_drift_on_search": _mean(
                run["memory_audit"].get("max_confidence_drift_on_search") for run in runs),
            "remediation_triggers": _mean(
                run["memory_audit"].get("remediation_triggers", 0) for run in runs),
            "superseded_claim_records": _mean(
                run["memory_audit"].get("superseded_claim_records", 0) for run in runs),
            "suppressed_claim_records": _mean(
                run["memory_audit"].get("suppressed_claim_records", 0) for run in runs),
            "counterfactual_flip_rate": _mean(
                run["metrics"].get("counterfactual_flip_rate", 0) for run in runs),
            "pre_guard_DCE": _mean(
                run["metrics"].get("pre_guard_DCE") for run in runs),
        })

    stage_summary = []
    for key, turns in sorted(stage_groups.items()):
        framework, arm, condition, stage = key
        stage_summary.append({
            "framework": framework, "solution_arm": arm,
            "condition": condition, "stage": stage, "n": len(turns),
            "ED": _mean(t["ED"] for t in turns),
            "target_accuracy": _mean(float(t["target_correct"]) for t in turns),
            "answer_similarity": _mean(t["answer_similarity"] for t in turns),
            "DCE": _mean(t["DCE"] for t in turns),
        })

    qa_summary = []
    for key, turns in sorted(qa_groups.items()):
        framework, arm, condition, qa_type = key
        qa_summary.append({
            "framework": framework, "solution_arm": arm,
            "condition": condition, "qa_type": qa_type, "n": len(turns),
            "ED": _mean(t["ED"] for t in turns),
            "target_accuracy": _mean(float(t["target_correct"]) for t in turns),
            "answer_similarity": _mean(t["answer_similarity"] for t in turns),
            "DCE": _mean(t["DCE"] for t in turns),
        })

    ability_summary = []
    for key, runs in sorted(ability_groups.items()):
        framework, arm, condition, source, ability = key
        ability_summary.append({
            "framework": framework, "solution_arm": arm,
            "condition": condition, "source_dataset": source,
            "source_ability": ability, "n": len(runs),
            **{metric: _mean(_run_metric(run, metric) for run in runs)
               for metric in METRICS},
            "final_accuracy": _mean(run["metrics"]["final_accuracy"] for run in runs),
        })

    delay_summary = []
    for key, points in sorted(delay_groups.items()):
        framework, arm, condition, delay = key
        delay_summary.append({
            "framework": framework, "solution_arm": arm,
            "condition": condition, "delay": delay, "n": len(points),
            "ED": _mean(p["ED"] for p in points),
            "target_accuracy": _mean(float(p["target_correct"]) for p in points),
        })

    # First create sample-level condition effects inside each arm.
    condition_effects = []
    indexed = {}
    for sample in samples:
        for run in sample["runs"]:
            indexed[(sample["sample_id"], run["framework"], run["solution_arm"],
                     run["condition"])] = run
    frameworks = sorted({key[1] for key in indexed})
    arms = sorted({key[2] for key in indexed})
    for sample in samples:
        sid = sample["sample_id"]
        for framework in frameworks:
            for arm in arms:
                control = indexed.get((sid, framework, arm, "control"))
                if not control:
                    continue
                for condition in ("repetition", "update"):
                    run = indexed.get((sid, framework, arm, condition))
                    if not run:
                        continue
                    condition_effects.append({
                        "sample_id": sid, "framework": framework,
                        "solution_arm": arm, "condition": condition,
                        **{f"{metric}_effect": _run_metric(run, metric) - _run_metric(control, metric)
                           for metric in METRICS},
                        "excess_WEP": _excess_wep(run, control),
                        "delay30_effect": _delay_value(run) - _delay_value(control),
                        "delay30_excess": max(0.0, _delay_value(run) - _delay_value(control)),
                        "final_accuracy_effect": (run["metrics"]["final_accuracy"]
                                                  - control["metrics"]["final_accuracy"]),
                    })

    did_rows = causal_estimands_from_condition_effects(condition_effects)

    # Normal-ability non-inferiority uses matched Control trajectories.
    utility_effects = []
    for framework in frameworks:
        for arm in arms:
            if arm == "baseline":
                continue
            for sample in samples:
                treated = indexed.get((sample["sample_id"], framework, arm, "control"))
                baseline = indexed.get((sample["sample_id"], framework, "baseline", "control"))
                if treated and baseline:
                    utility_effects.append({
                        "sample_id": sample["sample_id"], "framework": framework,
                        "solution_arm": arm,
                        "control_final_accuracy_delta": (
                            treated["metrics"]["final_accuracy"]
                            - baseline["metrics"]["final_accuracy"]
                        ),
                    })
    utility_rows = summarize_utility_effects(utility_effects)

    return {
        "condition_summary": condition_summary,
        "per_stage_summary": stage_summary,
        "per_qa_type_summary": qa_summary,
        "per_ability_summary": ability_summary,
        "per_delay_summary": delay_summary,
        "condition_effects": condition_effects,
        "causal_estimands": did_rows,
        "utility_effects": utility_effects,
        "utility_noninferiority": utility_rows,
    }


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
