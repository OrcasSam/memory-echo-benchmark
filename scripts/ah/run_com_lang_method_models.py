"""Generate, judge, and aggregate CoM/LangMem five-method answers."""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memory_echo_multimodel.providers import (
    apply_authority_answer_constraint, generate_answers, read_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results_memory_echo_com_lang_methods"
EXPECTED = 5700


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    tmp.replace(path)


def prepare(output: Path) -> None:
    source = read_jsonl(output / "decision_inputs.jsonl")
    if len(source) != EXPECTED:
        raise RuntimeError(f"expected {EXPECTED} method inputs, found {len(source)}")
    for provider, display, model in (
        ("deepseek", "DeepSeek V4 Flash", "deepseek-chat"),
        ("kimi", "Kimi K2.6", os.getenv("KIMI_MODEL", "kimi-k2.6")),
    ):
        rows = []
        for source_row in source:
            row = dict(source_row)
            row["evaluation_id"] = f"{provider}|{source_row['evaluation_key']}"
            row["generator_display_name"] = display
            row["generator_api_model"] = model
            rows.append(row)
        write_jsonl(output / f"{provider}_inputs.jsonl", rows)


def combine(output: Path) -> None:
    combined = []
    for provider in ("deepseek", "kimi"):
        rows = read_jsonl(output / f"{provider}_predictions.jsonl")
        if len(rows) != EXPECTED:
            raise RuntimeError(f"{provider}: expected {EXPECTED}, found {len(rows)}")
        combined.extend(apply_authority_answer_constraint(row) for row in rows)
    if len({row["evaluation_id"] for row in combined}) != EXPECTED * 2:
        raise RuntimeError("combined predictions contain duplicate evaluation IDs")
    write_jsonl(output / "all_predictions.jsonl", combined)


def mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(output: Path) -> None:
    rows = read_jsonl(output / "judged_answers.jsonl")
    if len(rows) != EXPECTED * 2:
        raise RuntimeError(f"expected {EXPECTED * 2} judged answers, found {len(rows)}")
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(
            row["generator_provider"], row["framework"], row["method"], row["condition"]
        )].append(row)
    granular = []
    for key, subset in sorted(grouped.items()):
        granular.append({
            "generator": key[0], "framework": key[1], "method": key[2],
            "condition": key[3], "n": len(subset),
            "accuracy": mean(float(r["judge_correctness_score"] >= 0.75) for r in subset),
            "mean_correctness": mean(r["judge_correctness_score"] for r in subset),
            "old_answer_rate": mean(float(r["judge_echo_contaminated"]) for r in subset),
            "ED": mean(r["ED"] for r in subset),
            "RRI": mean(r["RRI"] for r in subset),
            "WEP": mean(r["WEP"] for r in subset),
            "DCE": mean(r["DCE_gpt56"] for r in subset),
        })
    write_csv(output / "aggregate_by_generator_method.csv", granular)

    # Main method comparison is pre-registered over the two Echo conditions;
    # Control remains available in the granular table for utility checks.
    echo_conditions = {"read_only_repetition", "retrieve_writeback"}
    table_d = []
    for framework in ("CoM", "LangMem"):
        for method in ("Baseline", "Budget", "CRAG-M", "PairLR", "FPA"):
            subset = [
                row for row in rows
                if row["generator_provider"] == "deepseek"
                and row["framework"] == framework and row["method"] == method
                and row["condition"] in echo_conditions
            ]
            table_d.append({
                "framework": framework, "method": method, "n": len(subset),
                "accuracy": mean(float(r["judge_correctness_score"] >= 0.75) for r in subset),
                "old_answer_rate": mean(float(r["judge_echo_contaminated"]) for r in subset),
                "ED": mean(r["ED"] for r in subset),
                "WEP": mean(r["WEP"] for r in subset),
                "DCE": mean(r["DCE_gpt56"] for r in subset),
            })
    write_csv(output / "table_D_com_lang_deepseek.csv", table_d)

    table_f = []
    for framework in ("CoM", "LangMem"):
        for method in ("Baseline", "FPA"):
            subset = [
                row for row in rows
                if row["generator_provider"] == "kimi"
                and row["framework"] == framework and row["method"] == method
                and row["condition"] in echo_conditions
            ]
            table_f.append({
                "framework": framework, "method": method, "n": len(subset),
                "accuracy": mean(float(r["judge_correctness_score"] >= 0.75) for r in subset),
                "old_answer_rate": mean(float(r["judge_echo_contaminated"]) for r in subset),
                "DCE": mean(r["DCE_gpt56"] for r in subset),
            })
    write_csv(output / "table_F_com_lang_kimi.csv", table_f)
    (output / "integrity.json").write_text(json.dumps({
        "expected_judged": EXPECTED * 2,
        "judged": len(rows),
        "unique_evaluation_ids": len({row["evaluation_id"] for row in rows}),
        "granular_groups": len(granular),
        "passed": len(rows) == len({row["evaluation_id"] for row in rows}) == EXPECTED * 2,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=RESULTS)
    parser.add_argument(
        "--phase", choices=("prepare", "deepseek", "kimi", "combine", "aggregate"),
        required=True,
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    if args.phase == "prepare":
        prepare(args.output)
        print("provider_inputs_prepared", flush=True)
    elif args.phase == "deepseek":
        print(json.dumps(generate_answers(
            args.output / "deepseek_inputs.jsonl",
            args.output / "deepseek_predictions.jsonl",
            provider="deepseek", model="deepseek-chat",
            api_key_env="DEEPSEEK_API_KEY", base_url="https://api.deepseek.com",
            temperature=0.0, max_completion_tokens=120, disable_thinking=False,
            workers=args.workers, retries=args.retries, requests_per_minute=120.0,
        ), ensure_ascii=False), flush=True)
    elif args.phase == "kimi":
        print(json.dumps(generate_answers(
            args.output / "kimi_inputs.jsonl",
            args.output / "kimi_predictions.jsonl",
            provider="kimi", model=os.getenv("KIMI_MODEL", "kimi-k2.6"),
            api_key_env="KIMI_API_KEY",
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            # Kimi K2.6 rejects other temperatures; 0.6 is a provider-level
            # constraint and is recorded in every prediction for auditability.
            temperature=0.6, max_completion_tokens=120, disable_thinking=True,
            workers=args.workers, retries=args.retries, requests_per_minute=90.0,
        ), ensure_ascii=False), flush=True)
    elif args.phase == "combine":
        combine(args.output)
        print("predictions_combined", flush=True)
    else:
        aggregate(args.output)
        print("aggregate_complete", flush=True)


if __name__ == "__main__":
    main()
