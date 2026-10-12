"""Translate the 100 latest case records for the Chinese Word report."""
from __future__ import annotations

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results_memory_echo_dce_v3_all" / "stage_cases_all_dce_v3_enriched.json"
OUTPUT = ROOT / "results_memory_echo_dce_v3_all" / "case_translations_zh.jsonl"


SYSTEM = """You are a precise research-data translator. Translate supplied English text into
concise natural Simplified Chinese. Preserve names, dates, numbers, dataset terminology, and
the distinction between an old Echo answer and a verified Gold answer. Return only a JSON
object with exactly the requested keys. Do not correct or reinterpret an incorrect model answer."""


def payload(row):
    before = row.get("before") or {}
    after = row.get("after") or {}
    return {
        "question_zh": row.get("question") or "",
        "model_answer_zh": row.get("freeform_answer") or "",
        "gold_answer_zh": row.get("gold_answer") or "",
        "echo_answer_zh": row.get("echo_answer") or "",
        "before_answer_zh": before.get("answer") or "",
        "after_answer_zh": after.get("answer") or "",
    }


def call_one(client, row, retries=4):
    wanted = payload(row)
    user = (
        "Translate every non-empty value to Simplified Chinese. Keep the same keys.\n" +
        json.dumps(wanted, ensure_ascii=False)
    )
    error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": user}],
                temperature=0.0,
                max_tokens=900,
                response_format={"type": "json_object"},
                timeout=60,
            )
            translated = json.loads(response.choices[0].message.content)
            if set(translated) != set(wanted):
                raise ValueError("translation keys do not match")
            return {"case_id": row["case_id"], **translated}
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(10.0, 1.5 * 2 ** attempt))
    raise RuntimeError(f"{row['case_id']}: {error}")


def main():
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    cases = json.loads(SOURCE.read_text(encoding="utf-8"))
    existing = {}
    if OUTPUT.exists():
        with OUTPUT.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    existing[row["case_id"]] = row
    pending = []
    for row in cases:
        prior = existing.get(row["case_id"])
        wanted = payload(row)
        if prior is None or any(wanted[k] and not prior.get(k) for k in wanted):
            pending.append(row)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=60,
                    max_retries=0)
    lock = threading.Lock()
    with OUTPUT.open("a" if OUTPUT.exists() else "w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=16) as pool:
            futures = {pool.submit(call_one, client, row): row for row in pending}
            for index, future in enumerate(as_completed(futures), 1):
                result = future.result()
                with lock:
                    out.write(json.dumps(result, ensure_ascii=False) + "\n")
                    out.flush()
                if index % 20 == 0 or index == len(pending):
                    print(f"translated={index}/{len(pending)}", flush=True)
    final = {}
    with OUTPUT.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                final[row["case_id"]] = row
    if len(final) != len(cases):
        raise RuntimeError(f"expected {len(cases)} translations, found {len(final)}")
    print(f"complete cases={len(final)}")


if __name__ == "__main__":
    main()
