"""Recalculate parameter-free DCE v3 for baseline and A/B/C/D outputs.

The script never regenerates model answers and never overwrites legacy result
files.  It applies one frozen formula to every existing answer, writes
sample-level audit data, and creates full-200 plus locked-190 comparisons.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np

from memory_echo_abcd.dce import (
    CONTAMINATION_MARGIN,
    LEGACY_CONTAMINATION_THRESHOLD,
    LEGACY_TAU,
    is_contaminated,
    relative_distance_dce,
)


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results_memory_echo_dce_v3_all"
SOURCE = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
LEGACY_BASELINE = (
    ROOT / "results_memory_echo_solution_freeform_dce_200" /
    "sample_level_dce_v2.jsonl"
)
BASELINE = (
    ROOT / "results_memory_echo_baseline_equal_protocol_top5" /
    "sample_level_decisions.jsonl"
)
SOLUTION_DIRS = {
    "A": ROOT / "results_memory_echo_solution_A_v3_top5",
    "B": ROOT / "results_memory_echo_solution_B_top5",
    "C": ROOT / "results_memory_echo_solution_C_top5",
    "D": ROOT / "results_memory_echo_solution_D_top5",
}
PREVIOUS_A_DIR = ROOT / "results_memory_echo_solution_A_optimized_top5"
LATEST_CASES = (
    ROOT / "results_memory_echo_solution_A_optimized_comparison" /
    "latest_ABCD_paired_stage_cases.json"
)
FRAMEWORKS = ("A-MEM", "Mem0")
CONDITIONS = ("control", "repetition", "update")
ECHO_CONDITIONS = {"repetition", "update"}
SEED = 20260831
BOOTSTRAPS = 10_000


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows):
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def score_row(row, state: str, solution: str):
    row = dict(row)
    if "echo_similarity" in row:
        echo_similarity = float(row["echo_similarity"])
        gold_similarity = float(row["gold_similarity"])
        legacy = float(row.get("DCE_v2", row["DCE"]))
        legacy_contaminated = bool(row.get("contaminated", False))
    else:
        echo_similarity = float(row["similarity_echo"])
        gold_similarity = float(row["similarity_gold"])
        legacy = float(row["DCE_v2"])
        legacy_contaminated = bool(row.get("contaminated_v2", False))

    dce_v3 = relative_distance_dce(echo_similarity, gold_similarity)
    contaminated = is_contaminated(echo_similarity, gold_similarity)
    row.update({
        "evaluation_state": state,
        "evaluation_solution": solution,
        "echo_similarity": echo_similarity,
        "gold_similarity": gold_similarity,
        "DCE_v2": legacy,
        "DCE_v3": dce_v3,
        "DCE": dce_v3,
        "contamination_margin": CONTAMINATION_MARGIN,
        "contaminated_v3_protocol": contaminated,
        "legacy_contamination_match": contaminated == legacy_contaminated,
    })
    return row


def key(row):
    return row["sample_id"], row["framework"], row["condition"]


def development_ids():
    with SOURCE.open("rb") as handle:
        items = ijson.items(handle, "item")
        return {next(items)["sample_id"] for _ in range(10)}


def aggregate(rows, label: str, state: str, scope: str):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["framework"], row["condition"])].append(row)
    output = []
    for framework in FRAMEWORKS:
        for condition in CONDITIONS:
            values = grouped[(framework, condition)]
            if not values:
                raise RuntimeError(f"missing group: {label}/{framework}/{condition}/{scope}")
            output.append({
                "scope": scope,
                "state": state,
                "solution": label,
                "framework": framework,
                "condition": condition,
                "n": len(values),
                "DCE_v2_legacy": float(np.mean([r["DCE_v2"] for r in values])),
                "DCE_v3": float(np.mean([r["DCE_v3"] for r in values])),
                "DCE_v3_std": float(np.std([r["DCE_v3"] for r in values], ddof=1)),
                "contamination_rate": float(np.mean([
                    r["contaminated_v3_protocol"] for r in values
                ])),
                "gold_preference_rate": float(np.mean([
                    r["gold_similarity"] > r["echo_similarity"] for r in values
                ])),
                "mean_echo_similarity": float(np.mean([
                    r["echo_similarity"] for r in values
                ])),
                "mean_gold_similarity": float(np.mean([
                    r["gold_similarity"] for r in values
                ])),
            })
    return output


def bootstrap_ci(values, rng):
    values = np.asarray(values, dtype=float)
    indices = rng.integers(0, len(values), size=(BOOTSTRAPS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def before_after_table(baseline_rows, after_by_solution, scope: str, rng):
    baseline = {key(row): row for row in baseline_rows}
    output = []
    paired = []
    for solution in "ABCD":
        after = {key(row): row for row in after_by_solution[solution]}
        if set(after) != set(baseline):
            raise RuntimeError(f"unmatched before/after keys: {solution}/{scope}")
        for framework in FRAMEWORKS:
            for condition in CONDITIONS:
                keys = sorted(
                    k for k in baseline if k[1] == framework and k[2] == condition
                )
                before = np.asarray([baseline[k]["DCE_v3"] for k in keys])
                final = np.asarray([after[k]["DCE_v3"] for k in keys])
                reduction = before - final
                low, high = bootstrap_ci(reduction, rng)
                before_mean = float(before.mean())
                after_mean = float(final.mean())
                reduction_mean = float(reduction.mean())
                output.append({
                    "scope": scope,
                    "solution": solution,
                    "framework": framework,
                    "condition": condition,
                    "paired_n": len(keys),
                    "DCE_before": before_mean,
                    "DCE_after": after_mean,
                    "signed_change_after_minus_before": after_mean - before_mean,
                    "DCE_reduction_before_minus_after": reduction_mean,
                    "relative_reduction": (
                        reduction_mean / before_mean if before_mean else 0.0
                    ),
                    "reduction_ci95_low": low,
                    "reduction_ci95_high": high,
                })
                paired.extend({
                    "scope": scope,
                    "solution": solution,
                    "sample_id": k[0],
                    "framework": k[1],
                    "condition": k[2],
                    "DCE_before": float(baseline[k]["DCE_v3"]),
                    "DCE_after": float(after[k]["DCE_v3"]),
                    "signed_change_after_minus_before": float(
                        after[k]["DCE_v3"] - baseline[k]["DCE_v3"]
                    ),
                    "DCE_reduction_before_minus_after": float(
                        baseline[k]["DCE_v3"] - after[k]["DCE_v3"]
                    ),
                } for k in keys)
    return output, paired


def echo_summary(rows_by_solution, scope: str):
    output = []
    for solution in "ABCD":
        rows = [r for r in rows_by_solution[solution]
                if r["condition"] in ECHO_CONDITIONS]
        output.append({
            "scope": scope,
            "solution": solution,
            "n": len(rows),
            "DCE_v2_legacy": float(np.mean([r["DCE_v2"] for r in rows])),
            "DCE_v3": float(np.mean([r["DCE_v3"] for r in rows])),
            "contamination_rate": float(np.mean([
                r["contaminated_v3_protocol"] for r in rows
            ])),
            "gold_preference_rate": float(np.mean([
                r["gold_similarity"] > r["echo_similarity"] for r in rows
            ])),
        })
    return output


def baseline_and_solution_echo_summary(baseline_rows, rows_by_solution, scope: str):
    before = [r for r in baseline_rows if r["condition"] in ECHO_CONDITIONS]
    output = [{
        "scope": scope,
        "state": "before",
        "solution": "Baseline",
        "n": len(before),
        "DCE_v2_legacy": float(np.mean([r["DCE_v2"] for r in before])),
        "DCE_v3": float(np.mean([r["DCE_v3"] for r in before])),
        "contamination_rate": float(np.mean([
            r["contaminated_v3_protocol"] for r in before
        ])),
        "gold_preference_rate": float(np.mean([
            r["gold_similarity"] > r["echo_similarity"] for r in before
        ])),
    }]
    for row in echo_summary(rows_by_solution, scope):
        output.append({"scope": row["scope"], "state": "after", **{
            k: v for k, v in row.items() if k != "scope"
        }})
    return output


def recalculate_stage_cases():
    cases = json.loads(LATEST_CASES.read_text(encoding="utf-8"))
    # Replace the previous A cases with cases generated from the enhanced A-v3
    # run; frozen B/C/D cases remain unchanged.
    cases = [row for row in cases if row.get("solution") != "A"]
    cases.extend(json.loads(
        (SOLUTION_DIRS["A"] / "stage_cases.json").read_text(encoding="utf-8")
    ))
    for row in cases:
        old_dce = float(row["DCE"])
        new_dce = relative_distance_dce(
            row["echo_similarity"], row["gold_similarity"]
        )
        row["DCE_v2"] = old_dce
        row["DCE_v3"] = new_dce
        row["DCE"] = new_dce
        row["contamination_margin"] = CONTAMINATION_MARGIN
        row["contaminated_v3_protocol"] = is_contaminated(
            row["echo_similarity"], row["gold_similarity"]
        )
        if isinstance(row.get("after"), dict):
            row["after"]["DCE_v2"] = row["after"].get("DCE")
            row["after"]["DCE_v3"] = new_dce
            row["after"]["DCE"] = new_dce
            row["after"]["contaminated"] = row["contaminated_v3_protocol"]
    return cases


def compare_a_to_bcd(rows_by_solution, scope: str, rng):
    maps = {solution: {key(r): r for r in rows}
            for solution, rows in rows_by_solution.items()}
    echo_keys = sorted(k for k in maps["A"] if k[2] in ECHO_CONDITIONS)
    output = []
    for comparator in "BCD":
        a = np.asarray([maps["A"][k]["DCE_v3"] for k in echo_keys])
        b = np.asarray([maps[comparator][k]["DCE_v3"] for k in echo_keys])
        difference = a - b
        low, high = bootstrap_ci(difference, rng)
        output.append({
            "scope": scope,
            "comparison": f"A_vs_{comparator}",
            "paired_n": len(echo_keys),
            "A_DCE_v3": float(a.mean()),
            "comparator_DCE_v3": float(b.mean()),
            "difference_A_minus_comparator": float(difference.mean()),
            "ci95_low": low,
            "ci95_high": high,
            "A_lower_rate": float(np.mean(difference < 0.0)),
        })
    return output


def compare_a_versions(previous_rows, enhanced_rows, scope: str, rng):
    previous = {key(row): row for row in previous_rows}
    enhanced = {key(row): row for row in enhanced_rows}
    if set(previous) != set(enhanced):
        raise RuntimeError("A-v2/A-v3 keys are not identical")
    output = []
    groups = [
        (framework, condition)
        for framework in FRAMEWORKS for condition in CONDITIONS
    ] + [("ALL", "echo_conditions")]
    for framework, condition in groups:
        keys = sorted(k for k in previous if (
            (framework == "ALL" and k[2] in ECHO_CONDITIONS)
            or (k[1] == framework and k[2] == condition)
        ))
        old = np.asarray([previous[k]["DCE_v3"] for k in keys])
        new = np.asarray([enhanced[k]["DCE_v3"] for k in keys])
        reduction = old - new
        low, high = bootstrap_ci(reduction, rng)
        output.append({
            "scope": scope,
            "framework": framework,
            "condition": condition,
            "paired_n": len(keys),
            "A_v2_DCE_v3": float(old.mean()),
            "A_enhanced_DCE_v3": float(new.mean()),
            "reduction_v2_minus_enhanced": float(reduction.mean()),
            "relative_reduction_vs_v2": float(
                reduction.mean() / old.mean() if old.mean() else 0.0
            ),
            "ci95_low": low,
            "ci95_high": high,
            "enhanced_lower_rate": float(np.mean(new < old)),
            "near_tie_rate": float(np.mean(np.abs(new - old) <= 1e-12)),
        })
    return output


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    dev = development_ids()

    baseline_source = [r for r in read_jsonl(BASELINE)
                       if r.get("solution_arm") == "baseline"]
    baseline_full = [score_row(r, "before", "Baseline") for r in baseline_source]
    if len(baseline_full) != 1200:
        raise RuntimeError(f"expected 1200 baseline rows, got {len(baseline_full)}")
    legacy_baseline_full = [
        score_row(r, "before_legacy_temperature_0.3", "Baseline_legacy")
        for r in read_jsonl(LEGACY_BASELINE)
        if r.get("solution_arm") == "baseline"
    ]

    after_full = {}
    for solution, directory in SOLUTION_DIRS.items():
        source_rows = read_jsonl(directory / "sample_level_decisions.jsonl")
        after_full[solution] = [score_row(r, "after", solution) for r in source_rows]
        if len(after_full[solution]) != 1200:
            raise RuntimeError(
                f"expected 1200 solution {solution} rows, got {len(after_full[solution])}"
            )

    previous_a_full = [
        score_row(r, "after_previous", "A_v2")
        for r in read_jsonl(PREVIOUS_A_DIR / "sample_level_decisions.jsonl")
    ]
    if len(previous_a_full) != 1200:
        raise RuntimeError(f"expected 1200 previous A rows, got {len(previous_a_full)}")

    # The binary boundary is kept mathematically equivalent to the v2 protocol.
    all_scored = baseline_full + sum((after_full[s] for s in "ABCD"), [])
    mismatches = [r for r in all_scored if not r["legacy_contamination_match"]]
    if mismatches:
        raise RuntimeError(
            f"legacy contamination boundary mismatch in {len(mismatches)} rows"
        )
    if any(not 0.0 <= r["DCE_v3"] <= 1.0 for r in all_scored):
        raise RuntimeError("DCE v3 escaped [0, 1]")

    baseline_locked = [r for r in baseline_full if r["sample_id"] not in dev]
    after_locked = {
        solution: [r for r in rows if r["sample_id"] not in dev]
        for solution, rows in after_full.items()
    }
    previous_a_locked = [r for r in previous_a_full if r["sample_id"] not in dev]
    if len(baseline_locked) != 1140 or any(
        len(rows) != 1140 for rows in after_locked.values()
    ):
        raise RuntimeError("locked-test row count is not 1140 per arm")

    write_jsonl(OUTPUT / "baseline_before_dce_v3.jsonl", baseline_full)
    write_jsonl(
        OUTPUT / "baseline_legacy_temperature_0.3_reference_dce_v3.jsonl",
        legacy_baseline_full,
    )
    write_jsonl(OUTPUT / "solution_A_v2_reference_dce_v3.jsonl", previous_a_full)
    for solution in "ABCD":
        write_jsonl(OUTPUT / f"solution_{solution}_after_dce_v3.jsonl",
                    after_full[solution])
    write_jsonl(OUTPUT / "sample_level_dce_v3_all.jsonl", all_scored)

    rng = np.random.default_rng(SEED)
    full_aggregate = aggregate(baseline_full, "Baseline", "before", "full_200")
    locked_aggregate = aggregate(
        baseline_locked, "Baseline", "before", "locked_190"
    )
    for solution in "ABCD":
        full_aggregate.extend(aggregate(
            after_full[solution], solution, "after", "full_200"
        ))
        locked_aggregate.extend(aggregate(
            after_locked[solution], solution, "after", "locked_190"
        ))
    write_csv(OUTPUT / "aggregate_full_200.csv", full_aggregate)
    write_csv(OUTPUT / "aggregate_locked_190.csv", locked_aggregate)

    full_comparison, full_pairs = before_after_table(
        baseline_full, after_full, "full_200", rng
    )
    locked_comparison, locked_pairs = before_after_table(
        baseline_locked, after_locked, "locked_190", rng
    )
    write_csv(OUTPUT / "before_after_full_200.csv", full_comparison)
    write_csv(OUTPUT / "before_after_locked_190.csv", locked_comparison)
    write_csv(OUTPUT / "paired_sample_full_200.csv", full_pairs)
    write_csv(OUTPUT / "paired_sample_locked_190.csv", locked_pairs)

    full_echo = echo_summary(after_full, "full_200")
    locked_echo = echo_summary(after_locked, "locked_190")
    write_csv(OUTPUT / "echo_summary_full_200.csv", full_echo)
    write_csv(OUTPUT / "echo_summary_locked_190.csv", locked_echo)
    write_csv(
        OUTPUT / "echo_before_after_summary_full_200.csv",
        baseline_and_solution_echo_summary(baseline_full, after_full, "full_200"),
    )
    write_csv(
        OUTPUT / "echo_before_after_summary_locked_190.csv",
        baseline_and_solution_echo_summary(
            baseline_locked, after_locked, "locked_190"
        ),
    )
    a_vs_bcd = compare_a_to_bcd(after_locked, "locked_190_echo_conditions", rng)
    write_csv(OUTPUT / "paired_A_vs_BCD_locked_190.csv", a_vs_bcd)
    a_version_comparison = compare_a_versions(
        previous_a_locked, after_locked["A"], "locked_190", rng
    )
    write_csv(OUTPUT / "paired_A_v2_vs_enhanced_locked_190.csv", a_version_comparison)

    stage_cases = recalculate_stage_cases()
    (OUTPUT / "stage_cases_all_dce_v3.json").write_text(
        json.dumps(stage_cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for solution in "ABCD":
        subset = [row for row in stage_cases if row["solution"] == solution]
        (OUTPUT / f"stage_cases_{solution}_dce_v3.json").write_text(
            json.dumps(subset, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    formula = {
        "metric": "DCE v3 - Relative Semantic Distance",
        "range": [0.0, 1.0],
        "formula": "(1-cos(answer,gold))/((1-cos(answer,gold))+(1-cos(answer,echo)))",
        "anchors": {"gold": 0.0, "equidistant": 0.5, "echo": 1.0},
        "zero_denominator_rule": 0.5,
        "legacy_formula": (
            "sigmoid((cos(answer,echo)-cos(answer,gold))/0.20)"
        ),
        "legacy_tau": LEGACY_TAU,
        "binary_contamination_rule": (
            "cos(answer,echo)-cos(answer,gold) >= "
            f"{CONTAMINATION_MARGIN:.15f}"
        ),
        "binary_rule_note": (
            "This is exactly equivalent to legacy DCE-v2 >= "
            f"{LEGACY_CONTAMINATION_THRESHOLD:.2f}; it is kept fixed so the "
            "contamination rate remains comparable."
        ),
        "difference_reporting": {
            "signed_change": "DCE_after-DCE_before; natural range [-1,1]",
            "reduction": "DCE_before-DCE_after; positive means improvement",
            "relative_reduction": "(DCE_before-DCE_after)/DCE_before",
        },
    }
    (OUTPUT / "formula_definition.json").write_text(
        json.dumps(formula, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results = {
        "formula": formula,
        "validation": {
            "baseline_rows": len(baseline_full),
            "baseline_model_protocol": "deepseek-chat, temperature=0.0",
            "legacy_baseline_reference_rows": len(legacy_baseline_full),
            "rows_per_solution": {s: len(after_full[s]) for s in "ABCD"},
            "previous_A_reference_rows": len(previous_a_full),
            "all_scored_rows": len(all_scored),
            "stage_cases_recalculated": len(stage_cases),
            "locked_rows_per_arm": len(baseline_locked),
            "development_samples_excluded": len(dev),
            "legacy_contamination_mismatches": len(mismatches),
            "all_DCE_v3_bounded": True,
            "bootstrap_seed": SEED,
            "bootstrap_replicates": BOOTSTRAPS,
        },
        "locked_echo_summary": locked_echo,
        "locked_A_vs_BCD": a_vs_bcd,
        "locked_A_v2_vs_enhanced": a_version_comparison,
        "locked_before_after": locked_comparison,
    }
    (OUTPUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    baseline_echo = [r for r in baseline_locked if r["condition"] in ECHO_CONDITIONS]
    before_mean = float(np.mean([r["DCE_v3"] for r in baseline_echo]))
    lines = [
        "DCE v3 全量重算完成。",
        f"锁定测试集 Echo 条件加入前 DCE v3：{before_mean:.6f}",
    ]
    for row in locked_echo:
        reduction = before_mean - row["DCE_v3"]
        lines.append(
            f"方案 {row['solution']} 加入后：{row['DCE_v3']:.6f}；"
            f"降低量：{reduction:.6f}；相对降低：{reduction / before_mean:.2%}"
        )
    lines.extend([
        "DCE v3 无温度参数，0 表示更接近 Gold，0.5 表示等距，1 表示更接近 Echo。",
        "有符号差值仍应报告在 [-1,1]；同时列出正向降低量，避免改变统计含义。",
    ])
    (OUTPUT / "conclusion.txt").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
