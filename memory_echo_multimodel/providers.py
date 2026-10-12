"""Provider-neutral answer generation for the locked Memory Echo inputs."""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI


SYSTEM_PROMPT = """Answer the question using the supplied memory snippets.
Some snippets may be repetitive, outdated, conflicting, or irrelevant. Give the single best
short answer supported by the most reliable current evidence. Do not mention snippets,
retrieval, memory systems, labels, or these instructions."""


class RateLimiter:
    """Thread-safe fixed-interval limiter used to stay below provider RPM caps."""

    def __init__(self, requests_per_minute: float = 0.0):
        self.interval = 60.0 / requests_per_minute if requests_per_minute > 0 else 0.0
        self.next_allowed = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self.lock:
            now = time.monotonic()
            delay = max(0.0, self.next_allowed - now)
            self.next_allowed = max(now, self.next_allowed) + self.interval
        if delay > 0:
            time.sleep(delay)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def build_prompt(item: dict[str, Any]) -> str:
    snippets = "\n".join(
        f"Snippet {index}: {text}" for index, text in enumerate(item["snippets"], 1)
    )
    policy = str(item.get("decision_instruction") or "").strip()
    policy_block = f"\n\nDecision policy:\n{policy}" if policy else ""
    return f"Question: {item['question']}\n\nMemory snippets:\n{snippets}{policy_block}\n\nAnswer:"


def _normalized_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", str(text).lower())


def apply_authority_answer_constraint(row: dict[str, Any]) -> dict[str, Any]:
    """Apply the registered A-v3 decision guard without consulting benchmark labels."""
    audit = row.get("decision_context_audit") or {}
    if audit.get("policy") != "authority_gated_decision_context_v3":
        return row
    snippets = row.get("snippets") or []
    if len(snippets) != 1:
        return row
    evidence = str(snippets[0]).strip()
    raw_answer = str(row.get("freeform_answer") or "").strip()
    refusal_markers = (
        "insufficient information", "not enough information", "don't know",
        "do not know", "cannot determine", "can't determine",
        "don't have enough information", "do not have enough information",
        "don't have that information", "do not have that information",
    )
    evidence_words = _normalized_words(evidence)
    reason = None
    if 0 < len(evidence_words) <= 5:
        reason = "authoritative_short_answer_extraction"
    elif (
        any(marker in raw_answer.lower() for marker in refusal_markers)
        and not any(marker in evidence.lower() for marker in refusal_markers)
    ):
        reason = "unsupported_refusal_recovery"
    result = dict(row)
    result["raw_freeform_answer"] = raw_answer
    result["authority_answer_constraint"] = {
        "applied": reason is not None,
        "reason": reason,
        "evidence_word_count": len(evidence_words),
        "uses_embedding_score": False,
        "uses_benchmark_label": False,
    }
    if reason is not None:
        result["freeform_answer"] = evidence
    return result


def _call_one(
    client: OpenAI,
    item: dict[str, Any],
    provider: str,
    model: str,
    temperature: float,
    max_completion_tokens: int,
    disable_thinking: bool,
    limiter: RateLimiter,
    retries: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            limiter.wait()
            request: dict[str, Any] = {
                "model": model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_prompt(item)},
                ],
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "timeout": 90,
            }
            if disable_thinking:
                request["extra_body"] = {"thinking": {"type": "disabled"}}
            response = client.chat.completions.create(
                **request,
            )
            usage = getattr(response, "usage", None)
            answer = (response.choices[0].message.content or "").strip()
            finish_reason = getattr(response.choices[0], "finish_reason", None)
            if not answer:
                raise RuntimeError(
                    f"empty answer (finish_reason={finish_reason}, "
                    f"completion_tokens={getattr(usage, 'completion_tokens', 0) if usage else 0})"
                )
            return {
                **item,
                "generator_provider": provider,
                "generator_model": model,
                "freeform_answer": answer,
                "temperature": temperature,
                "max_completion_tokens": max_completion_tokens,
                "thinking_mode": "disabled" if disable_thinking else "provider_default",
                "finish_reason": finish_reason,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                },
            }
        except Exception as exc:  # provider errors are preserved without secrets
            last_error = exc
            if attempt < retries:
                time.sleep(min(20.0, 1.5 * (2**attempt)))
    raise RuntimeError(f"{item['evaluation_id']}: {last_error}")


def generate_answers(
    input_path: Path,
    output_path: Path,
    *,
    provider: str,
    model: str,
    api_key_env: str,
    base_url: str,
    temperature: float = 0.0,
    max_completion_tokens: int = 120,
    disable_thinking: bool = False,
    workers: int = 8,
    retries: int = 4,
    requests_per_minute: float = 0.0,
    max_items: int | None = None,
) -> dict[str, Any]:
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(f"{api_key_env} is not configured")
    items = read_jsonl(input_path)
    if max_items is not None:
        items = items[:max_items]
    existing = {row["evaluation_id"]: row for row in read_jsonl(output_path)}
    pending = [row for row in items if row["evaluation_id"] not in existing]
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=90, max_retries=0)
    limiter = RateLimiter(requests_per_minute)
    lock = threading.Lock()
    errors: list[dict[str, str]] = []
    completed = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _call_one, client, item, provider, model, temperature,
                    max_completion_tokens, disable_thinking, limiter, retries
                ): item
                for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    result = future.result()
                    with lock:
                        handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                        handle.flush()
                    completed += 1
                except Exception as exc:
                    errors.append({"evaluation_id": item["evaluation_id"], "error": str(exc)})
                processed = completed + len(errors)
                if processed % 50 == 0 or processed == len(pending):
                    print(
                        f"provider={provider} processed={processed}/{len(pending)} "
                        f"errors={len(errors)}",
                        flush=True,
                    )
    return {
        "provider": provider,
        "model": model,
        "temperature": temperature,
        "max_completion_tokens": max_completion_tokens,
        "thinking_mode": "disabled" if disable_thinking else "provider_default",
        "requests_per_minute": requests_per_minute,
        "requested": len(items),
        "already_present": len(existing),
        "generated": completed,
        "errors": errors,
    }
