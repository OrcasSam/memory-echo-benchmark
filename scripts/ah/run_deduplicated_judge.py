"""Losslessly deduplicate identical GPT judge requests, then expand scores."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from memory_echo_multimodel.judge import judge_predictions
from memory_echo_multimodel.providers import read_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results_memory_echo_cross_model_gpt56"
PROMPT_FIELDS = ("question", "gold_answer", "echo_answer", "freeform_answer")


def request_key(row: dict[str, Any]) -> str:
    payload = json.dumps(
        [str(row.get(field, "")).strip() for field in PROMPT_FIELDS],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def prepare(results: Path, model: str, seed_judgments: Path | None = None) -> tuple[int, int]:
    predictions = read_jsonl(results / "all_predictions.jsonl")
    representatives: dict[str, dict[str, Any]] = {}
    for row in predictions:
        key = request_key(row)
        if key not in representatives:
            representative = dict(row)
            representative["source_evaluation_id"] = row["evaluation_id"]
            representative["evaluation_id"] = key
            representative["judge_request_key"] = key
            representatives[key] = representative
    write_jsonl(results / "unique_judge_inputs.jsonl", list(representatives.values()))

    # Reuse completed full-row judgments by converting them into keyed cache rows.
    cache_path = results / "unique_judgments.jsonl"
    cached = {row["evaluation_id"]: row for row in read_jsonl(cache_path)}
    seed_rows = read_jsonl(results / "judged_answers.jsonl")
    if seed_judgments is not None:
        seed_rows.extend(read_jsonl(seed_judgments))
    for row in seed_rows:
        if row.get("judge_model") != model:
            continue
        key = request_key(row)
        if key not in representatives:
            continue
        if key in cached:
            continue
        converted = dict(row)
        converted["source_evaluation_id"] = row["evaluation_id"]
        converted["evaluation_id"] = key
        converted["judge_request_key"] = key
        cached[key] = converted
    write_jsonl(cache_path, list(cached.values()))
    return len(predictions), len(representatives)


def expand(results: Path) -> int:
    predictions = read_jsonl(results / "all_predictions.jsonl")
    unique = {
        row["evaluation_id"]: row for row in read_jsonl(results / "unique_judgments.jsonl")
    }
    output = []
    for row in predictions:
        key = request_key(row)
        if key not in unique:
            raise RuntimeError(f"missing unique judgment for {key}")
        scored = unique[key]
        merged = dict(row)
        for field, value in scored.items():
            if field.startswith("judge_") or field == "DCE_gpt56":
                merged[field] = value
        merged["judge_request_key"] = key
        output.append(merged)
    write_jsonl(results / "judged_answers.jsonl", output)
    return len(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--seed-judgments", type=Path)
    args = parser.parse_args()
    load_dotenv(ROOT / ".env")
    model = args.model or os.getenv("OPENAI_JUDGE_MODEL", "gpt-5.6-sol")
    total, unique = prepare(args.results, model, args.seed_judgments)
    print(f"judge_rows={total} unique_requests={unique} reduction={1-unique/total:.4f}", flush=True)
    result = judge_predictions(
        args.results / "unique_judge_inputs.jsonl",
        args.results / "unique_judgments.jsonl",
        model=model,
        workers=args.workers,
        retries=args.retries,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    print(f"expanded={expand(args.results)}", flush=True)


if __name__ == "__main__":
    main()
