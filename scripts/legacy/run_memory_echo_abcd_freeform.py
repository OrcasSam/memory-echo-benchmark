"""Free-form DeepSeek decision and stage-case evaluation for one A/B/C/D run."""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer

from memory_echo_abcd.dce import (
    CONTAMINATION_MARGIN,
    is_contaminated,
    legacy_sigmoid_dce,
    relative_distance_dce,
)


ROOT = Path(__file__).resolve().parents[2]
SYSTEM_PROMPT = """Answer the question using the supplied memory snippets.
Some snippets may be repetitive, outdated, conflicting, or irrelevant. Give the single best
short answer supported by the most reliable current evidence. Do not mention snippets,
retrieval, memory systems, labels, or these instructions."""


def jsonl(path):
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def prompt(item):
    snippets = "\n".join(f"Snippet {i}: {text}" for i, text in enumerate(item["snippets"], 1))
    policy = str(item.get("decision_instruction") or "").strip()
    policy_block = f"\n\nDecision policy:\n{policy}" if policy else ""
    return (
        f"Question: {item['question']}\n\nMemory snippets:\n{snippets}"
        f"{policy_block}\n\nAnswer:"
    )


def call_one(client, model, item, retries):
    error = None
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": prompt(item)}],
                temperature=0.0, max_tokens=120, timeout=60,
            )
            usage = response.usage
            return {**item, "freeform_answer": (response.choices[0].message.content or "").strip(),
                    "model": model, "temperature": 0.0,
                    "usage": {"prompt_tokens": getattr(usage, "prompt_tokens", 0),
                              "completion_tokens": getattr(usage, "completion_tokens", 0),
                              "total_tokens": getattr(usage, "total_tokens", 0)}}
        except Exception as exc:
            error = exc
            if attempt < retries:
                time.sleep(min(12.0, 1.5 * 2 ** attempt))
    raise RuntimeError(f"{item.get('item_id') or item.get('case_id')}: {error}")


def generate(input_path, output_path, id_field, model, base_url, workers, retries):
    key = os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    items = jsonl(input_path)
    existing = {row[id_field]: row for row in jsonl(output_path)} if output_path.exists() else {}
    pending = [row for row in items if row[id_field] not in existing]
    client = OpenAI(api_key=key, base_url=base_url, timeout=60, max_retries=0)
    lock = threading.Lock()
    errors = []
    with output_path.open("a" if output_path.exists() else "w", encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(call_one, client, model, row, retries): row for row in pending}
            for completed, future in enumerate(as_completed(futures), 1):
                try:
                    result = future.result()
                    with lock:
                        out.write(json.dumps(result, ensure_ascii=False) + "\n")
                        out.flush()
                except Exception as exc:
                    errors.append(str(exc))
                if completed % 50 == 0 or completed == len(pending):
                    print(f"generated={completed}/{len(pending)} errors={len(errors)}", flush=True)
    if errors:
        raise RuntimeError(f"Free-form generation failed for {len(errors)} items; first={errors[0]}")


def reuse_a_predictions(output_dir):
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        # Legacy A predictions were generated from the strict full-guard
        # context.  They are not valid for the optimized family/provenance
        # circuit-breaker context and must never be silently reused.
        if manifest.get("a_variant") in {"optimized", "optimized_v3"}:
            return False
    source = ROOT / "results_memory_echo_solution_freeform_dce_200" / "freeform_predictions.jsonl"
    if not source.exists():
        return False
    wanted = {row["item_id"]: row for row in jsonl(output_dir / "decision_inputs.jsonl")}
    converted = []
    for old in jsonl(source):
        if old.get("solution_arm") != "full_guard":
            continue
        new_id = "|".join((old["sample_id"], old["framework"], old["condition"], "A"))
        if new_id not in wanted:
            continue
        converted.append({**wanted[new_id], "freeform_answer": old["freeform_answer"],
                          "model": old.get("model", "deepseek-chat"),
                          "temperature": old.get("temperature", 0.0),
                          "usage": old.get("usage", {})})
    if len(converted) != len(wanted):
        return False
    with (output_dir / "freeform_predictions.jsonl").open("w", encoding="utf-8") as out:
        for row in converted:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def model_path():
    snapshots = (ROOT.parent / ".hf_cache" / "hub" /
                 "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots")
    candidates = sorted(p for p in snapshots.glob("*") if (p / "modules.json").exists())
    return str(candidates[-1]) if candidates else "all-MiniLM-L6-v2"


def embed_scores(rows, answer_field="freeform_answer"):
    model = SentenceTransformer(model_path(), device=None)
    texts = []
    for row in rows:
        texts.extend([row[answer_field], row["echo_answer"], row["gold_answer"]])
    unique = list(dict.fromkeys(texts))
    vectors = model.encode(unique, batch_size=256, normalize_embeddings=True,
                           show_progress_bar=True)
    lookup = {text: vector for text, vector in zip(unique, vectors)}
    for row in rows:
        answer = lookup[row[answer_field]]
        echo_sim = float(np.dot(answer, lookup[row["echo_answer"]]))
        gold_sim = float(np.dot(answer, lookup[row["gold_answer"]]))
        dce_v2 = legacy_sigmoid_dce(echo_sim, gold_sim)
        dce_v3 = relative_distance_dce(echo_sim, gold_sim)
        row["echo_similarity"] = echo_sim
        row["gold_similarity"] = gold_sim
        row["DCE_v2"] = dce_v2
        row["DCE_v3"] = dce_v3
        row["DCE"] = dce_v3
        row["contamination_margin"] = CONTAMINATION_MARGIN
        row["contaminated"] = is_contaminated(echo_sim, gold_sim)
        row["gold_preferred"] = bool(gold_sim > echo_sim)
    return rows


def _normalized_words(text):
    return re.findall(r"[a-z0-9]+(?:['-][a-z0-9]+)?", str(text).lower())


def apply_authority_answer_constraint(rows):
    """Constrain direct authoritative answers without evaluation labels.

    This is intentionally lexical and answer-type based; it does not use the
    DCE embedding model, gold/echo labels, or similarity thresholds.  The raw
    model answer is retained for auditing.
    """
    refusal_markers = (
        "insufficient information", "not enough information", "don't know",
        "do not know", "cannot determine", "can't determine",
        "don't have enough information", "do not have enough information",
        "don't have that information", "do not have that information",
    )
    for row in rows:
        audit = row.get("decision_context_audit") or {}
        if audit.get("policy") != "authority_gated_decision_context_v3":
            continue
        snippets = row.get("snippets") or []
        if len(snippets) != 1:
            continue
        evidence = str(snippets[0]).strip()
        raw_answer = str(row.get("freeform_answer") or "").strip()
        evidence_words = _normalized_words(evidence)
        raw_lower = raw_answer.lower()
        evidence_lower = evidence.lower()
        reason = None

        # A short authoritative value is already the complete answer.  Free
        # generation can only add unsupported content or reverse polarity.
        if 0 < len(evidence_words) <= 5:
            reason = "authoritative_short_answer_extraction"
        elif (any(marker in raw_lower for marker in refusal_markers)
              and not any(marker in evidence_lower for marker in refusal_markers)):
            reason = "unsupported_refusal_recovery"

        row["raw_freeform_answer"] = raw_answer
        row["authority_answer_constraint"] = {
            "applied": reason is not None,
            "reason": reason,
            "evidence_word_count": len(evidence_words),
            "uses_embedding_score": False,
            "uses_benchmark_label": False,
        }
        if reason is not None:
            row["freeform_answer"] = evidence
    return rows


def update_aggregate(output_dir):
    predictions = jsonl(output_dir / "freeform_predictions.jsonl")
    predictions = apply_authority_answer_constraint(predictions)
    predictions = embed_scores(predictions)
    with (output_dir / "sample_level_decisions.jsonl").open("w", encoding="utf-8") as out:
        for row in predictions:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    grouped = defaultdict(list)
    for row in predictions:
        grouped[(row["solution"], row["framework"], row["condition"])].append(row)
    aggregate = []
    for key, rows in sorted(grouped.items()):
        aggregate.append({
            "solution": key[0], "framework": key[1], "condition": key[2], "n": len(rows),
            "ED": float(np.mean([r["ED"] for r in rows])),
            "RRI": float(np.mean([r["RRI"] for r in rows])),
            "WEP": float(np.mean([r["WEP"] for r in rows])),
            "DCE": float(np.mean([r["DCE"] for r in rows])),
            "DCE_std": float(np.std([r["DCE"] for r in rows], ddof=1)),
            "contamination_rate": float(np.mean([r["contaminated"] for r in rows])),
            "gold_preference_rate": float(np.mean([r["gold_preferred"] for r in rows])),
        })
    with (output_dir / "aggregate_results.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(aggregate[0]))
        writer.writeheader(); writer.writerows(aggregate)
    (output_dir / "aggregate_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_case_inputs(output_dir, per_stage=10):
    candidates = jsonl(output_dir / "case_candidates.jsonl")
    selected = []
    stages = ("encoding", "retrieval_practice", "updating_reconsolidation",
              "delayed_retrieval", "final_decision")
    for stage in stages:
        rows = [r for r in candidates if r["stage"] == stage and r["condition"] in {"repetition", "update"}]
        # Prefer observed retrieval errors and high ED, while ensuring different
        # samples/frameworks.  This is only a candidate pool; final cases are
        # selected after free-form correctness is known.
        rows.sort(key=lambda r: (r["retrieval_target_correct"], -float(r["ED"] or 0.0)))
        seen, chosen = set(), []
        for row in rows:
            key = (row["sample_id"], row["framework"])
            if key in seen:
                continue
            seen.add(key); chosen.append(row)
            if len(chosen) >= per_stage:
                break
        selected.extend(chosen)
    with (output_dir / "case_inputs.jsonl").open("w", encoding="utf-8") as out:
        for row in selected:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")


def finalize_cases(output_dir, cases_per_stage=5):
    rows = embed_scores(jsonl(output_dir / "case_freeform_predictions.jsonl"))
    for row in rows:
        expected_echo = row["target_label"] == "ECHO"
        row["answer_correct"] = (row["echo_similarity"] >= row["gold_similarity"]
                                 if expected_echo else row["gold_similarity"] > row["echo_similarity"])
        row["echo_affected"] = bool(
            row["echo_similarity"] >= row["gold_similarity"] and
            row["stage"] in {"updating_reconsolidation", "delayed_retrieval", "final_decision"}
        )
    final = []
    for stage in ("encoding", "retrieval_practice", "updating_reconsolidation",
                  "delayed_retrieval", "final_decision"):
        subset = [r for r in rows if r["stage"] == stage]
        subset.sort(key=lambda r: (not r["echo_affected"], r["answer_correct"], -r["DCE"]))
        chosen = subset[:cases_per_stage]
        # If possible, retain at least one correct case for an honest contrast.
        if chosen and not any(r["answer_correct"] for r in chosen):
            correct = next((r for r in subset if r["answer_correct"]), None)
            if correct:
                chosen[-1] = correct
        final.extend(chosen)
    (output_dir / "stage_cases.json").write_text(
        json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = ["solution", "stage", "case_id", "sample_id", "dataset", "ability",
              "framework", "condition", "qa_type", "question", "freeform_answer",
              "gold_answer", "echo_answer", "ED", "DCE", "answer_correct", "echo_affected"]
    with (output_dir / "stage_cases.csv").open("w", encoding="utf-8-sig", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(final)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--solution", choices=list("ABCD"), required=True)
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--decisions_only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    reused = args.solution == "A" and reuse_a_predictions(output)
    if not reused and not args.skip_generation:
        generate(output / "decision_inputs.jsonl", output / "freeform_predictions.jsonl",
                 "item_id", args.model, args.base_url, args.workers, args.retries)
    update_aggregate(output)
    if args.decisions_only:
        print(f"freeform_decisions_complete solution={args.solution}", flush=True)
        return
    prepare_case_inputs(output)
    if not args.skip_generation:
        generate(output / "case_inputs.jsonl", output / "case_freeform_predictions.jsonl",
                 "case_id", args.model, args.base_url, args.workers, args.retries)
        finalize_cases(output)
    print(f"freeform_complete solution={args.solution}", flush=True)


if __name__ == "__main__":
    main()
