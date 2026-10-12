"""
Reference-guided LLM judge for LoCoMo QA results.

This is a Mem0-style evaluation pass: an LLM judge compares the generated
answer against the gold answer for the same question and returns a binary
correctness label plus an error type.
"""

import argparse
import json
import os
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional


JUDGE_PROMPT = """You are an impartial evaluator for long-term memory question answering.

Your task is to judge whether the predicted answer is semantically correct compared with the reference answer for the given question.

Question:
{question}

Reference answer:
{reference}

Predicted answer:
{prediction}

Judging rules:
- Mark correct if the prediction has the same meaning as the reference, even if wording is different.
- For date/time questions, accept equivalent or clearly compatible relative dates only when they identify the same time.
- Mark incorrect if the prediction is missing the main fact, contradicts the reference, answers a different question, or adds unsupported facts that change the answer.
- Do not require exact string match.

Return only valid JSON with this schema:
{{
  "correct": 0 or 1,
  "score": 0.0 or 1.0,
  "error_type": "correct" | "wrong_entity" | "wrong_time" | "incomplete" | "hallucination" | "irrelevant" | "other",
  "rationale": "brief reason"
}}
"""


def parse_json_response(text: str) -> Dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"Judge response did not contain JSON: {text[:200]}")
    return json.loads(match.group(0))


def make_client(api_key: Optional[str], base_url: str):
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError("Install openai package to run the judge.") from exc
    api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY or pass --api_key.")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=30.0, max_retries=0)


def judge_one(client, model: str, item: Dict, temperature: float, max_retries: int) -> Dict:
    prompt = JUDGE_PROMPT.format(
        question=item.get("question", ""),
        reference=item.get("reference", ""),
        prediction=item.get("prediction", ""),
    )
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=1200,
                timeout=30,
            )
            parsed = parse_json_response(response.choices[0].message.content or "")
            usage = getattr(response, "usage", None)
            return {
                "judge": normalize_judge(parsed),
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                    "calls": 1,
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(1.5 * (2 ** attempt))
    raise RuntimeError(f"Judge failed after retries: {last_error}")


def normalize_judge(raw: Dict) -> Dict:
    correct = int(raw.get("correct", 0) or 0)
    correct = 1 if correct == 1 else 0
    score = float(raw.get("score", correct) or 0.0)
    if correct == 1:
        score = 1.0
    elif score not in (0.0, 1.0):
        score = 0.0
    error_type = str(raw.get("error_type", "correct" if correct else "other")).strip()
    allowed = {
        "correct",
        "wrong_entity",
        "wrong_time",
        "incomplete",
        "hallucination",
        "irrelevant",
        "other",
    }
    if error_type not in allowed:
        error_type = "correct" if correct else "other"
    return {
        "correct": correct,
        "score": score,
        "error_type": error_type,
        "rationale": str(raw.get("rationale", "")).strip(),
    }


def add_usage(total: Dict[str, int], usage: Dict[str, int]) -> None:
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "calls"):
        total[key] = total.get(key, 0) + int(usage.get(key, 0) or 0)


def summarize(judged_results: List[Dict]) -> Dict:
    total = len(judged_results)
    correct = sum(int(item["judge"]["correct"]) for item in judged_results)
    by_category = defaultdict(list)
    error_counts = defaultdict(int)
    for item in judged_results:
        by_category[str(item.get("category"))].append(int(item["judge"]["correct"]))
        error_counts[item["judge"]["error_type"]] += 1

    categories = {}
    for category, values in sorted(by_category.items()):
        categories[category] = {
            "count": len(values),
            "correct": sum(values),
            "judge_accuracy": sum(values) / len(values) if values else 0.0,
        }

    return {
        "count": total,
        "correct": correct,
        "judge_accuracy": correct / total if total else 0.0,
        "by_category": categories,
        "error_counts": dict(sorted(error_counts.items())),
    }


def build_payload(
    args,
    input_path: Path,
    result_data: Dict,
    judged_results: List[Dict],
    usage_total: Dict[str, int],
) -> Dict:
    return {
        "judge_model": args.model,
        "judge_style": "mem0_reference_guided_binary",
        "source_result": str(input_path),
        "source_completed_samples": result_data.get("completed_samples"),
        "source_target_samples": result_data.get("target_samples"),
        "summary": summarize(judged_results),
        "api_usage": usage_total,
        "judged_results": judged_results,
    }


def write_payload(
    output_path: Path,
    args,
    input_path: Path,
    result_data: Dict,
    judged_results: List[Dict],
    usage_total: Dict[str, int],
) -> None:
    payload = build_payload(args, input_path, result_data, judged_results, usage_total)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def judge_indexed_item(idx: int, item: Dict, args) -> Dict:
    client = make_client(args.api_key, args.base_url)
    judged = dict(item)
    try:
        response = judge_one(client, args.model, item, args.temperature, args.max_retries)
        judged["judge"] = response["judge"]
        usage = response["usage"]
    except Exception as exc:
        judged["judge"] = {
            "correct": 0,
            "score": 0.0,
            "error_type": "other",
            "rationale": f"Judge failed: {exc}",
        }
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
        }
    return {"idx": idx, "judged": judged, "usage": usage}


def main():
    parser = argparse.ArgumentParser(description="Run a Mem0-style LLM judge over LoCoMo QA results.")
    parser.add_argument("--input", required=True, help="Path to results JSON from test_advanced_robust.py")
    parser.add_argument("--output", required=True, help="Path to save judged results JSON")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--api_key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_items", type=int, default=None)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    with input_path.open("r", encoding="utf-8") as f:
        result_data = json.load(f)

    items = result_data.get("individual_results", [])
    if args.max_items is not None:
        items = items[:args.max_items]

    judged_results = []
    usage_total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "calls": 0,
    }

    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            existing = json.load(f)
        judged_results = existing.get("judged_results", [])
        add_usage(usage_total, existing.get("api_usage", {}))

    start = len(judged_results)
    workers = max(1, int(args.workers or 1))
    if workers == 1:
        for idx, item in enumerate(items[start:], start=start):
            result = judge_indexed_item(idx, item, args)
            judged = result["judged"]
            add_usage(usage_total, result["usage"])
            judged_results.append(judged)
            write_payload(output_path, args, input_path, result_data, judged_results, usage_total)
            print(
                f"Judged {idx + 1}/{len(items)}: "
                f"correct={judged['judge']['correct']} "
                f"category={judged.get('category')} "
                f"error={judged['judge']['error_type']}",
                flush=True,
            )
            if args.sleep:
                time.sleep(args.sleep)
    else:
        pending = {}
        next_to_write = start
        work_items = list(enumerate(items[start:], start=start))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(judge_indexed_item, idx, item, args)
                for idx, item in work_items
            ]
            for future in as_completed(futures):
                result = future.result()
                pending[result["idx"]] = result
                while next_to_write in pending:
                    ready = pending.pop(next_to_write)
                    judged = ready["judged"]
                    add_usage(usage_total, ready["usage"])
                    judged_results.append(judged)
                    write_payload(output_path, args, input_path, result_data, judged_results, usage_total)
                    print(
                        f"Judged {next_to_write + 1}/{len(items)}: "
                        f"correct={judged['judge']['correct']} "
                        f"category={judged.get('category')} "
                        f"error={judged['judge']['error_type']}",
                        flush=True,
                    )
                    next_to_write += 1
                    if args.sleep:
                        time.sleep(args.sleep)

    print(json.dumps(summarize(judged_results), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
