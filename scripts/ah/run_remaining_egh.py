"""Run and aggregate all remaining cells for Memory Echo Tables D--H."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

from memory_echo_multimodel.providers import (
    apply_authority_answer_constraint, generate_answers, read_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results_memory_echo_remaining_egh"
FINAL = ROOT / "results_memory_echo_ah_complete"
ECHO_NEW = {"read_only_repetition", "retrieve_writeback"}
ECHO_OLD = {"repetition", "update"}


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def accuracy(rows) -> float:
    return mean(float(row["judge_correctness_score"] >= 0.75) for row in rows)


def old_rate(rows) -> float:
    return mean(float(row["judge_echo_contaminated"]) for row in rows)


def metric_value(row: dict[str, Any], key: str) -> float:
    """Read lifecycle metrics from either the legacy top level or metrics block."""
    if key in row:
        return float(row[key])
    metrics = row.get("metrics") or {}
    aliases = {"ED": "pre_correction_ED", "WEP": "WEP30"}
    nested_key = aliases.get(key, key)
    if nested_key in metrics:
        return float(metrics[nested_key])
    raise KeyError(f"missing metric {key} in evaluation {row.get('evaluation_id')}")


def prepare_raw_authority_rows(output: Path) -> int:
    source = read_jsonl(ROOT / "results_memory_echo_com_lang_methods" / "judged_answers.jsonl")
    rows = []
    for row in source:
        if (row["generator_provider"] != "deepseek" or row["method"] != "FPA"
                or row["condition"] not in ECHO_NEW):
            continue
        raw = str(row.get("raw_freeform_answer") or row["freeform_answer"])
        item = dict(row)
        item["experiment"] = "E"
        item["configuration"] = "F+P+Authority"
        item["evaluation_id"] = "E-raw|" + row["evaluation_id"]
        item["freeform_answer"] = raw
        for key in list(item):
            if key.startswith("judge_") or key == "DCE_gpt56":
                item.pop(key, None)
        rows.append(item)
    if len(rows) != 760:
        raise RuntimeError(f"expected 760 CoM/Lang raw FPA rows, found {len(rows)}")
    write_jsonl(output / "e_raw_authority_predictions.jsonl", rows)
    return len(rows)


def combine_new(output: Path) -> None:
    deepseek = [apply_authority_answer_constraint(row)
                for row in read_jsonl(output / "deepseek_predictions.jsonl")]
    kimi = read_jsonl(output / "kimi_baseline_predictions.jsonl")
    raw = read_jsonl(output / "e_raw_authority_predictions.jsonl")
    if len(deepseek) != 11840 or len(kimi) != 1140 or len(raw) != 760:
        raise RuntimeError(
            f"prediction count mismatch deepseek={len(deepseek)} kimi={len(kimi)} raw={len(raw)}"
        )
    combined = deepseek + kimi + raw
    if len({row["evaluation_id"] for row in combined}) != len(combined):
        raise RuntimeError("duplicate evaluation IDs in remaining predictions")
    write_jsonl(output / "all_predictions.jsonl", combined)


def table_d() -> list[dict[str, Any]]:
    baseline = [row for row in read_jsonl(
        ROOT / "results_memory_echo_ah_answers" / "judged_answers.jsonl"
    ) if row["condition"] in ECHO_NEW]
    old = [row for row in read_jsonl(
        ROOT / "results_memory_echo_cross_model_gpt56_valid" / "judged_answers.jsonl"
    ) if row["generator_provider"] == "deepseek" and row["condition"] in ECHO_OLD]
    new = [row for row in read_jsonl(
        ROOT / "results_memory_echo_com_lang_methods" / "judged_answers.jsonl"
    ) if row["generator_provider"] == "deepseek" and row["condition"] in ECHO_NEW]
    mapping = {"A": "FPA", "B": "Budget", "C": "CRAG-M", "D": "PairLR"}
    records = []
    for row in baseline:
        item = dict(row); item["method"] = "Baseline"; records.append(item)
    for row in old:
        item = dict(row); item["method"] = mapping[row["solution"]]; records.append(item)
    # The strict lifecycle file already supplies the paired Baseline rows for
    # all four frameworks.  The CoM/LangMem method run is used only for the
    # four mitigation methods here, otherwise Baseline would be counted twice.
    records.extend(row for row in new if row["method"] != "Baseline")
    output = []
    for framework in ("A-MEM", "Mem0", "CoM", "LangMem"):
        for method in ("Baseline", "Budget", "CRAG-M", "PairLR", "FPA"):
            subset = [r for r in records if r["framework"] == framework and r["method"] == method]
            if len(subset) != 380:
                raise RuntimeError(f"D {framework}/{method}: expected 380, found {len(subset)}")
            output.append({
                "framework": framework, "method": method, "n": len(subset),
                "accuracy": accuracy(subset), "old_answer_rate": old_rate(subset),
                "ED": mean(metric_value(r, "ED") for r in subset),
                "WEP": mean(metric_value(r, "WEP") for r in subset),
                "DCE": mean(r["DCE_gpt56"] for r in subset),
            })
    return output


def table_e(new_judged) -> list[dict[str, Any]]:
    records = []
    # Baseline from the strict four-framework lifecycle.
    for row in read_jsonl(ROOT / "results_memory_echo_ah_answers" / "judged_answers.jsonl"):
        if row["condition"] in ECHO_NEW:
            item = dict(row); item["configuration"] = "Baseline"; records.append(item)
    # Missing independent components generated in this run.
    records.extend(row for row in new_judged if row.get("experiment") == "E")
    # A-MEM/Mem0 raw vs constrained FPA ablation.
    for row in read_jsonl(ROOT / "results_memory_echo_raw_constraint_ablation" / "judged_answers.jsonl"):
        if row["generator_provider"] != "deepseek" or row["condition"] not in ECHO_OLD:
            continue
        item = dict(row)
        item["configuration"] = (
            "F+P+Authority" if row["answer_variant"] == "raw" else "Full FPA"
        )
        records.append(item)
    # CoM/LangMem full FPA is already judged in the five-method experiment.
    for row in read_jsonl(ROOT / "results_memory_echo_com_lang_methods" / "judged_answers.jsonl"):
        if (row["generator_provider"] == "deepseek" and row["method"] == "FPA"
                and row["condition"] in ECHO_NEW):
            item = dict(row); item["configuration"] = "Full FPA"; records.append(item)
    configs = ("Baseline", "Only-F", "Only-P", "F+P", "F+P+Authority", "Full FPA")
    output = []
    for config in configs:
        subset = [row for row in records if row["configuration"] == config]
        if len(subset) != 1520:
            raise RuntimeError(f"E {config}: expected 1520, found {len(subset)}")
        output.append({
            "configuration": config, "n": len(subset), "accuracy": accuracy(subset),
            "old_answer_rate": old_rate(subset),
            "DCE": mean(row["DCE_gpt56"] for row in subset),
        })
    return output


def bootstrap_ci(values, seed: int, n_boot: int = 10000):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(n_boot, len(array)), replace=True), axis=1)
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def table_f(new_judged) -> list[dict[str, Any]]:
    records = []
    records.extend(
        row for row in new_judged
        if row.get("experiment") == "F" and row["condition"] in ECHO_NEW
    )
    for row in read_jsonl(ROOT / "results_memory_echo_cross_model_gpt56_valid" / "judged_answers.jsonl"):
        if (row["generator_provider"] == "kimi" and row["solution"] == "A"
                and row["condition"] in ECHO_OLD):
            item = dict(row); item["method"] = "FPA"; records.append(item)
    for row in read_jsonl(ROOT / "results_memory_echo_com_lang_methods" / "judged_answers.jsonl"):
        if (row["generator_provider"] == "kimi" and row["method"] in {"Baseline", "FPA"}
                and row["condition"] in ECHO_NEW):
            records.append(row)
    output = []
    for index, framework in enumerate(("A-MEM", "Mem0", "CoM", "LangMem")):
        base = [r for r in records if r["framework"] == framework and r["method"] == "Baseline"]
        fpa = [r for r in records if r["framework"] == framework and r["method"] == "FPA"]
        if len(base) != 380 or len(fpa) != 380:
            raise RuntimeError(f"F {framework}: baseline={len(base)} fpa={len(fpa)}")
        def per_sample(rows):
            grouped = defaultdict(list)
            for row in rows:
                grouped[row["sample_id"]].append(float(row["judge_correctness_score"] >= .75))
            return {key: mean(value) for key, value in grouped.items()}
        b, f = per_sample(base), per_sample(fpa)
        deltas = [f[key] - b[key] for key in sorted(b)]
        low, high = bootstrap_ci(deltas, 20260930 + index)
        output.append({
            "framework": framework, "n_samples": len(deltas),
            "baseline_accuracy": accuracy(base), "FPA_accuracy": accuracy(fpa),
            "paired_improvement": mean(deltas), "ci95_low": low, "ci95_high": high,
        })
    return output


def table_g(new_judged) -> list[dict[str, Any]]:
    rows = [row for row in new_judged if row.get("experiment") == "G"]
    output = []
    for task in ("shopping", "medical", "travel"):
        base = [r for r in rows if r["task"] == task and r["method"] == "Baseline"]
        fpa = [r for r in rows if r["task"] == task and r["method"] == "FPA"]
        if len(base) != 200 or len(fpa) != 200:
            raise RuntimeError(f"G {task}: baseline={len(base)} fpa={len(fpa)}")
        output.append({
            "task": task, "n_per_method": len(base),
            "baseline_accuracy": accuracy(base), "FPA_accuracy": accuracy(fpa),
            "old_answer_rate_before": old_rate(base), "old_answer_rate_after": old_rate(fpa),
            "old_answer_rate_change": old_rate(fpa) - old_rate(base),
        })
    return output


def table_h(new_judged) -> list[dict[str, Any]]:
    rows = [row for row in new_judged if row.get("experiment") == "H"]
    output = []
    for condition in (
        "complete", "family_noise", "correction_missing_or_noisy", "old_fact_nonconflicting",
    ):
        base = [r for r in rows if r["source_condition"] == condition and r["method"] == "Baseline"]
        fpa = [r for r in rows if r["source_condition"] == condition and r["method"] == "FPA"]
        if len(base) != 760 or len(fpa) != 760:
            raise RuntimeError(f"H {condition}: baseline={len(base)} fpa={len(fpa)}")
        output.append({
            "source_condition": condition, "n_per_method": len(base),
            "baseline_accuracy": accuracy(base), "FPA_accuracy": accuracy(fpa),
            "valid_history_retention_rate": mean(
                float(row["valid_history_retained"]) for row in fpa
            ),
        })
    return output


def aggregate(output: Path, final: Path) -> None:
    new_judged = read_jsonl(output / "judged_answers.jsonl")
    if len(new_judged) != 13740:
        raise RuntimeError(f"expected 13740 new judged rows, found {len(new_judged)}")
    final.mkdir(parents=True, exist_ok=True)
    tables = {
        "D": table_d(), "E": table_e(new_judged), "F": table_f(new_judged),
        "G": table_g(new_judged), "H": table_h(new_judged),
    }
    for name, rows in tables.items():
        write_csv(final / f"table_{name}_final.csv", rows)
    # Copy A--C values into the canonical complete directory without changing them.
    for name in "ABC":
        source = ROOT / "results_memory_echo_ah" / f"table_{name}_final.csv"
        (final / source.name).write_bytes(source.read_bytes())
    integrity = {
        "tables": {name: len(rows) for name, rows in tables.items()},
        "new_judged_records": len(new_judged),
        "unique_new_evaluation_ids": len({row["evaluation_id"] for row in new_judged}),
        "passed": len(new_judged) == len({row["evaluation_id"] for row in new_judged}) == 13740,
    }
    (final / "integrity.json").write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--final", type=Path, default=FINAL)
    parser.add_argument(
        "--phase", choices=("prepare", "deepseek", "kimi", "combine", "aggregate"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.phase == "prepare":
        count = prepare_raw_authority_rows(args.output)
        print(f"raw_authority_rows={count}", flush=True)
    elif args.phase == "deepseek":
        print(json.dumps(generate_answers(
            args.output / "deepseek_inputs.jsonl", args.output / "deepseek_predictions.jsonl",
            provider="deepseek", model="deepseek-chat", api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com", temperature=0.0,
            max_completion_tokens=120, disable_thinking=False,
            workers=args.workers, retries=args.retries, requests_per_minute=120.0,
        ), ensure_ascii=False), flush=True)
    elif args.phase == "kimi":
        print(json.dumps(generate_answers(
            args.output / "kimi_baseline_inputs.jsonl",
            args.output / "kimi_baseline_predictions.jsonl",
            provider="kimi", model=os.getenv("KIMI_MODEL", "kimi-k2.6"),
            api_key_env="KIMI_API_KEY", base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            temperature=0.6, max_completion_tokens=120, disable_thinking=True,
            workers=args.workers, retries=args.retries, requests_per_minute=90.0,
        ), ensure_ascii=False), flush=True)
    elif args.phase == "combine":
        combine_new(args.output); print("combined", flush=True)
    else:
        aggregate(args.output, args.final); print("aggregate_complete", flush=True)


if __name__ == "__main__":
    main()
