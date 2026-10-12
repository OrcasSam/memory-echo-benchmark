"""Generate and judge the final answers required by strict Tables A--C.

The paper-facing generator name is ``DeepSeek V4 Flash`` and the provider's
API model identifier is ``deepseek-chat``.  Both are stored in every artifact
so that the naming convention does not hide the reproducible API parameter.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memory_echo_multimodel.judge import judge_predictions
from memory_echo_multimodel.providers import generate_answers, read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LIFECYCLE = ROOT / "results_memory_echo_ah" / "lifecycle_sample_level.jsonl"
DEFAULT_OUTPUT = ROOT / "results_memory_echo_ah_answers"
DISPLAY_MODEL = "DeepSeek V4 Flash"
API_MODEL = "deepseek-chat"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(lifecycle: Path, output: Path) -> dict[str, Any]:
    inputs: list[dict[str, Any]] = []
    with lifecycle.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            decision = row["decision_input"]
            evaluation_id = "|".join((
                "deepseek-v4-flash", row["sample_id"], row["framework"], row["condition"]
            ))
            inputs.append({
                "evaluation_id": evaluation_id,
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "ability": row["ability"],
                "framework": row["framework"],
                "condition": row["condition"],
                "question": decision["question"],
                "gold_answer": decision["gold_answer"],
                "echo_answer": decision["echo_answer"],
                "snippets": decision["snippets"],
                "generator_display_name": DISPLAY_MODEL,
                "generator_api_model": API_MODEL,
                "source_lifecycle": str(lifecycle.resolve()),
                "metrics": row["metrics"],
            })
    if len(inputs) != 2280 or len({row["evaluation_id"] for row in inputs}) != 2280:
        raise RuntimeError(f"expected 2280 unique lifecycle decisions, found {len(inputs)}")
    write_jsonl(output / "decision_inputs.jsonl", inputs)
    kimi_inputs = []
    for row in inputs:
        item = dict(row)
        item["evaluation_id"] = row["evaluation_id"].replace(
            "deepseek-v4-flash|", "kimi-k2.6|", 1
        )
        item["generator_display_name"] = "Kimi K2.6"
        item["generator_api_model"] = os.getenv("KIMI_MODEL", "kimi-k2.6")
        kimi_inputs.append(item)
    write_jsonl(output / "kimi_decision_inputs.jsonl", kimi_inputs)
    manifest = {
        "protocol": "strict_ah_final_decision_v1",
        "records": len(inputs),
        "paper_model_name": DISPLAY_MODEL,
        "api_model_id": API_MODEL,
        "generator_temperature": 0.0,
        "judge_model": os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol"),
        "judge_protocol": "two_pass_blinded_v1",
        "accuracy_definition": "judge_correctness_score >= 0.75",
        "old_answer_definition": "judge_echo_contaminated",
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _answer_summary(rows: list[dict[str, Any]], display_model: str, api_model: str) -> list[dict[str, Any]]:
    if len(rows) != 2280:
        raise RuntimeError(f"expected 2280 judged answers, found {len(rows)}")
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["framework"], row["condition"])].append(row)

    summary: list[dict[str, Any]] = []
    for (framework, condition), subset in sorted(grouped.items()):
        summary.append({
            "framework": framework,
            "history_condition": condition,
            "n": len(subset),
            "paper_model_name": display_model,
            "api_model_id": api_model,
            "final_accuracy": mean([
                float(row["judge_correctness_score"] >= 0.75) for row in subset
            ]),
            "mean_correctness_score": mean([
                float(row["judge_correctness_score"]) for row in subset
            ]),
            "old_answer_rate": mean([
                float(row["judge_echo_contaminated"]) for row in subset
            ]),
            "mean_DCE": mean([float(row["DCE_gpt56"]) for row in subset]),
        })
    return summary


def aggregate(output: Path) -> None:
    rows = read_jsonl(output / "judged_answers.jsonl")
    summary = _answer_summary(rows, DISPLAY_MODEL, API_MODEL)
    write_csv(output / "answer_summary.csv", summary)
    answer_lookup = {
        (row["framework"], row["history_condition"]): row for row in summary
    }
    lifecycle_dir = ROOT / "results_memory_echo_ah"

    def read_csv(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))

    table_a = read_csv(lifecycle_dir / "table_A_lifecycle_partial.csv")
    for row in table_a:
        answer = answer_lookup[(row["framework"], row["history_condition"])]
        row["post_correction_old_answer_rate"] = answer["old_answer_rate"]
        row["answer_model"] = DISPLAY_MODEL
    write_csv(lifecycle_dir / "table_A_final.csv", table_a)

    table_b = read_csv(lifecycle_dir / "table_B_lifecycle_partial.csv")
    for row in table_b:
        answer = answer_lookup[(row["framework"], row["history_condition"])]
        row["final_accuracy"] = answer["final_accuracy"]
        row["answer_model"] = DISPLAY_MODEL
    write_csv(lifecycle_dir / "table_B_final.csv", table_b)

    table_c = read_csv(lifecycle_dir / "table_C_lifecycle_partial.csv")
    for row in table_c:
        answer = answer_lookup[(row["framework"], "retrieve_writeback")]
        row["final_old_answer_rate"] = answer["old_answer_rate"]
        row["answer_model"] = DISPLAY_MODEL
    write_csv(lifecycle_dir / "table_C_final.csv", table_c)
    kimi_rows = read_jsonl(output / "kimi_judged_answers.jsonl")
    if kimi_rows:
        kimi_model = str(kimi_rows[0].get("generator_model") or "kimi-k2.6")
        write_csv(
            output / "kimi_answer_summary.csv",
            _answer_summary(kimi_rows, "Kimi K2.6", kimi_model),
        )
    (output / "integrity.json").write_text(json.dumps({
        "judged_records": len(rows),
        "unique_evaluation_ids": len({row["evaluation_id"] for row in rows}),
        "kimi_judged_records": len(kimi_rows),
        "kimi_unique_evaluation_ids": len({row["evaluation_id"] for row in kimi_rows}),
        "groups": len(summary),
        "passed": len(rows) == len({row["evaluation_id"] for row in rows}) == 2280,
    }, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lifecycle", type=Path, default=DEFAULT_LIFECYCLE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase", choices=(
            "prepare", "generate", "generate-kimi", "judge", "judge-kimi",
            "aggregate", "all",
        ),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--deepseek-rpm", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=5)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    args.output.mkdir(parents=True, exist_ok=True)

    if args.phase in {"prepare", "all"}:
        print(json.dumps(prepare(args.lifecycle, args.output), ensure_ascii=False), flush=True)
    if args.phase in {"generate", "all"}:
        result = generate_answers(
            args.output / "decision_inputs.jsonl",
            args.output / "deepseek_predictions.jsonl",
            provider="deepseek",
            model=API_MODEL,
            api_key_env="DEEPSEEK_API_KEY",
            base_url="https://api.deepseek.com",
            temperature=0.0,
            max_completion_tokens=120,
            disable_thinking=False,
            workers=args.workers,
            retries=args.retries,
            requests_per_minute=args.deepseek_rpm,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.phase in {"generate-kimi", "all"}:
        result = generate_answers(
            args.output / "kimi_decision_inputs.jsonl",
            args.output / "kimi_predictions.jsonl",
            provider="kimi",
            model=os.getenv("KIMI_MODEL", "kimi-k2.6"),
            api_key_env="KIMI_API_KEY",
            base_url=os.getenv("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
            temperature=0.0,
            max_completion_tokens=120,
            disable_thinking=True,
            workers=args.workers,
            retries=args.retries,
            requests_per_minute=90.0,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.phase in {"judge", "all"}:
        result = judge_predictions(
            args.output / "deepseek_predictions.jsonl",
            args.output / "judged_answers.jsonl",
            model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol"),
            workers=args.workers,
            retries=args.retries,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.phase in {"judge-kimi", "all"}:
        result = judge_predictions(
            args.output / "kimi_predictions.jsonl",
            args.output / "kimi_judged_answers.jsonl",
            model=os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol"),
            workers=args.workers,
            retries=args.retries,
        )
        print(json.dumps(result, ensure_ascii=False), flush=True)
    if args.phase in {"aggregate", "all"}:
        aggregate(args.output)
        print("aggregate_complete", flush=True)


if __name__ == "__main__":
    main()
