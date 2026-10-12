"""Locked DeepSeek/Kimi generation and GPT-5.6 evaluation for Memory Echo.

The runner reuses the existing A-v3/B/C/D retrieval contexts.  The first ten
development samples are excluded before any API request.  DeepSeek answers are
reused from the completed experiment; Kimi receives exactly the same inputs.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import ijson
import numpy as np
from dotenv import load_dotenv

from memory_echo_multimodel.judge import judge_predictions
from memory_echo_multimodel.providers import (
    apply_authority_answer_constraint,
    generate_answers,
    read_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_memory_echo_solution_strict_200" / "sample_level.json"
SOLUTION_DIRS = {
    "A": ROOT / "results_memory_echo_solution_A_v3_top5",
    "B": ROOT / "results_memory_echo_solution_B_top5",
    "C": ROOT / "results_memory_echo_solution_C_top5",
    "D": ROOT / "results_memory_echo_solution_D_top5",
}
DEFAULT_OUTPUT = ROOT / "results_memory_echo_cross_model_gpt56"
EXPECTED_LOCKED = 190 * 2 * 3 * 4


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def development_ids() -> set[str]:
    with SOURCE.open("rb") as handle:
        stream = ijson.items(handle, "item")
        return {next(stream)["sample_id"] for _ in range(10)}


def prepare(output: Path) -> dict[str, Any]:
    dev = development_ids()
    locked_inputs: list[dict[str, Any]] = []
    deepseek_rows: list[dict[str, Any]] = []
    for solution, directory in SOLUTION_DIRS.items():
        inputs = {
            row["item_id"]: row
            for row in read_jsonl(directory / "decision_inputs.jsonl")
            if row["sample_id"] not in dev
        }
        completed = {
            row["item_id"]: row
            for row in read_jsonl(directory / "sample_level_decisions.jsonl")
            if row["sample_id"] not in dev
        }
        if len(inputs) != 190 * 2 * 3 or set(inputs) != set(completed):
            raise RuntimeError(
                f"{solution}: expected 1140 matched locked inputs/answers, "
                f"found {len(inputs)}/{len(completed)}"
            )
        for item_id in sorted(inputs):
            base = dict(inputs[item_id])
            base["evaluation_id"] = f"kimi|{item_id}"
            locked_inputs.append(base)
            old = dict(completed[item_id])
            old["evaluation_id"] = f"deepseek|{item_id}"
            old["generator_provider"] = "deepseek"
            old["generator_model"] = old.get("model", "deepseek-chat")
            deepseek_rows.append(old)
    if len(locked_inputs) != EXPECTED_LOCKED:
        raise RuntimeError(f"expected {EXPECTED_LOCKED} locked inputs, found {len(locked_inputs)}")
    write_jsonl(output / "locked_inputs.jsonl", locked_inputs)
    write_jsonl(output / "deepseek_predictions.jsonl", deepseek_rows)
    manifest = {
        "protocol": "cross_generator_locked_v1",
        "development_samples_excluded": len(dev),
        "locked_samples": 190,
        "solutions": list("ABCD"),
        "frameworks": ["A-MEM", "Mem0"],
        "conditions": ["control", "repetition", "update"],
        "answers_per_generator": EXPECTED_LOCKED,
        "deepseek_reused": True,
        "kimi_reuses_identical_decision_inputs": True,
        "judge_protocol": "two-pass blinded correctness/contamination",
        "api_keys_persisted": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def combine_predictions(output: Path) -> int:
    deepseek = read_jsonl(output / "deepseek_predictions.jsonl")
    kimi = [apply_authority_answer_constraint(row) for row in read_jsonl(output / "kimi_predictions.jsonl")]
    if len(deepseek) != EXPECTED_LOCKED or len(kimi) != EXPECTED_LOCKED:
        raise RuntimeError(
            f"expected {EXPECTED_LOCKED} rows per generator, found "
            f"deepseek={len(deepseek)} kimi={len(kimi)}"
        )
    rows = deepseek + kimi
    if len({row["evaluation_id"] for row in rows}) != len(rows):
        raise RuntimeError("duplicate evaluation_id in combined predictions")
    write_jsonl(output / "all_predictions.jsonl", rows)
    return len(rows)


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return float(np.mean([float(row[field]) for row in rows]))


def _bootstrap_ci(values: list[float], seed: int, n_boot: int = 10_000) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = np.mean(rng.choice(array, size=(n_boot, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output: Path) -> dict[str, Any]:
    rows = read_jsonl(output / "judged_answers.jsonl")
    expected = EXPECTED_LOCKED * 2
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} judged answers, found {len(rows)}")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["generator_provider"], row["solution"], row["framework"], row["condition"])].append(row)
    summary = []
    for key, subset in sorted(grouped.items()):
        summary.append({
            "generator": key[0], "solution": key[1], "framework": key[2],
            "condition": key[3], "n": len(subset),
            "ED": _mean(subset, "ED"), "RRI": _mean(subset, "RRI"),
            "WEP": _mean(subset, "WEP"), "DCE_gpt56": _mean(subset, "DCE_gpt56"),
            "correctness_gpt56": _mean(subset, "judge_correctness_score"),
            "faithfulness_gpt56": _mean(subset, "judge_evidence_faithfulness"),
            "echo_contamination_rate_gpt56": _mean(subset, "judge_echo_contaminated"),
            "contamination_severity_gpt56": _mean(subset, "judge_contamination_severity"),
        })
    _write_csv(output / "aggregate_by_generator.csv", summary)

    lookup = {
        (row["generator_provider"], row["sample_id"], row["framework"], row["condition"], row["solution"]): row
        for row in rows
    }
    comparisons = []
    seed = 20260929
    for generator in ("deepseek", "kimi"):
        for comparator in "BCD":
            for scope, conditions in (("echo", {"repetition", "update"}), ("control", {"control"})):
                per_sample = defaultdict(list)
                for key, a_row in lookup.items():
                    gen, sample_id, framework, condition, solution = key
                    if gen != generator or solution != "A" or condition not in conditions:
                        continue
                    other = lookup[(generator, sample_id, framework, condition, comparator)]
                    per_sample[sample_id].append(a_row["DCE_gpt56"] - other["DCE_gpt56"])
                deltas = [float(np.mean(values)) for values in per_sample.values()]
                low, high = _bootstrap_ci(deltas, seed + len(comparisons))
                comparisons.append({
                    "generator": generator, "comparison": f"A_vs_{comparator}",
                    "scope": scope, "paired_samples": len(deltas),
                    "mean_DCE_delta_A_minus_comparator": float(np.mean(deltas)),
                    "ci95_low": low, "ci95_high": high,
                    "A_lower_DCE_sample_rate": float(np.mean(np.asarray(deltas) < 0)),
                })
    _write_csv(output / "paired_method_effects.csv", comparisons)
    result = {"aggregate": summary, "paired_method_effects": comparisons}
    (output / "aggregate_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase", choices=("prepare", "generate-kimi", "combine", "judge", "aggregate", "all"),
        default="all",
    )
    parser.add_argument("--kimi-model", default=os.getenv("KIMI_MODEL", "kimi-k2.6"))
    parser.add_argument(
        "--kimi-base-url", default=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1")
    )
    parser.add_argument("--judge-model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol"))
    parser.add_argument("--kimi-temperature", type=float, default=0.6)
    parser.add_argument("--kimi-max-completion-tokens", type=int, default=120)
    parser.add_argument("--kimi-disable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--kimi-rpm", type=float, default=90.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--max-items", type=int)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.phase in {"prepare", "all"}:
        print(json.dumps(prepare(args.output_dir), ensure_ascii=False), flush=True)
    if args.phase in {"generate-kimi", "all"}:
        result = generate_answers(
            args.output_dir / "locked_inputs.jsonl",
            args.output_dir / "kimi_predictions.jsonl",
            provider="kimi", model=args.kimi_model, api_key_env="KIMI_API_KEY",
            base_url=args.kimi_base_url, temperature=args.kimi_temperature,
            max_completion_tokens=args.kimi_max_completion_tokens,
            disable_thinking=args.kimi_disable_thinking,
            workers=args.workers, retries=args.retries,
            requests_per_minute=args.kimi_rpm,
            max_items=args.max_items,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.phase in {"combine", "all"}:
        print(f"combined={combine_predictions(args.output_dir)}", flush=True)
    if args.phase in {"judge", "all"}:
        result = judge_predictions(
            args.output_dir / "all_predictions.jsonl",
            args.output_dir / "judged_answers.jsonl",
            model=args.judge_model, workers=args.workers, retries=args.retries,
            max_items=args.max_items,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.phase in {"aggregate", "all"}:
        aggregate(args.output_dir)
        print("aggregate_complete", flush=True)


if __name__ == "__main__":
    main()
