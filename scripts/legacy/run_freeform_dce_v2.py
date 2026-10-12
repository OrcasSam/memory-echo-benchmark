"""Re-evaluate Memory Echo final decisions with free-form LLM answers.

This runner reuses the completed A-MEM/Mem0 retrieval trajectories and changes
only the decision layer. It removes ECHO/CORRECTION metadata from the prompt,
asks DeepSeek for one free-form answer per run, and computes a smoother DCE v2.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from openai import OpenAI


ROOT = Path(__file__).resolve().parents[2]
SOURCE_RESULTS = ROOT / "results_memory_echo_solution_strict_200"
DEFAULT_INPUT = SOURCE_RESULTS / "sample_level.json"
DEFAULT_OUTPUT = ROOT / "results_memory_echo_solution_freeform_dce_200"
ARMS = {"baseline", "full_guard"}
CONDITIONS = {"control", "repetition", "update"}
FRAMEWORKS = {"A-MEM", "Mem0"}

SYSTEM_PROMPT = """You answer a general question using the retrieved memory snippets.
The snippets are presented without authority labels. Some may be relevant and some may be irrelevant.
Give the single best answer supported by the snippets. Answer in one short sentence or phrase.
Do not mention snippet numbers, retrieval, memory systems, or this instruction."""


def _neutral_content(hit: Dict, echo_value: str, gold_value: str) -> str:
    """Keep rank/multiplicity while removing benchmark labels and metadata."""
    label = hit.get("label") or hit.get("supplier_id")
    if label == "ECHO":
        return str(echo_value).strip()
    if label == "CORRECTION":
        return str(gold_value).strip()
    text = str(hit.get("text", "")).strip()
    # Noise strings include benchmark bookkeeping. Present a neutral short note.
    if label == "NOISE":
        return "An unrelated session note contains no answer to the current question."
    return text


def extract_inputs(source_path: Path, output_path: Path) -> int:
    import ijson
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with source_path.open("rb") as src, output_path.open("w", encoding="utf-8") as out:
        for sample in ijson.items(src, "item"):
            for run in sample["runs"]:
                if run["solution_arm"] not in ARMS:
                    continue
                if run["condition"] not in CONDITIONS or run["framework"] not in FRAMEWORKS:
                    continue
                final_turns = [t for t in run["trajectory"] if t["stage"] == "final_decision"]
                if not final_turns:
                    continue
                # One decision per run avoids treating four synonymous final prompts
                # as four independent observations.
                turn = final_turns[-1]
                snippets = [
                    _neutral_content(hit, sample["echo_value"], sample["ground_truth_value"])
                    for hit in turn["retrieved_memories"]
                ]
                row = {
                    "item_id": "|".join([
                        sample["sample_id"], run["framework"], run["condition"],
                        run["solution_arm"],
                    ]),
                    "sample_id": sample["sample_id"],
                    "dataset": sample["source"]["dataset"],
                    "ability": sample["source"]["ability"],
                    "framework": run["framework"],
                    "condition": run["condition"],
                    "solution_arm": run["solution_arm"],
                    "question": sample["source"]["question"],
                    "gold_answer": sample["ground_truth_value"],
                    "echo_answer": sample["echo_value"],
                    "snippets": snippets,
                    "retrieval_ranks": [
                        None if h.get("retrieval_rank") is None else int(h["retrieval_rank"])
                        for h in turn["retrieved_memories"]
                    ],
                    # Preserve the benchmark's registered aggregate ED definition.
                    "original_ED": float(run["metrics"]["ED"]["retrieval_practice"]),
                    "original_RRI": float(run["metrics"]["RRI"]),
                    "original_WEP": float(run["metrics"]["WEP"]),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
    return count


def load_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_user_prompt(item: Dict) -> str:
    snippets = "\n".join(
        f"Snippet {index}: {text}" for index, text in enumerate(item["snippets"], 1)
    )
    return f"Question: {item['question']}\n\nRetrieved snippets:\n{snippets}\n\nAnswer:"


def call_one(client: OpenAI, model: str, item: Dict, temperature: float,
             max_retries: int) -> Dict:
    last_error = None
    for attempt in range(max_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_prompt(item)},
                ],
                temperature=temperature,
                max_tokens=120,
                timeout=60,
            )
            answer = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            return {
                **item,
                "freeform_answer": answer,
                "model": model,
                "temperature": temperature,
                "usage": {
                    "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
                    "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
                    "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
                },
            }
        except Exception as exc:
            last_error = exc
            if attempt < max_retries:
                time.sleep(min(12.0, 1.5 * (2 ** attempt)))
    raise RuntimeError(f"{item['item_id']}: {last_error}")


def generate(inputs_path: Path, predictions_path: Path, model: str,
             base_url: str, temperature: float, workers: int,
             max_retries: int, max_items: int | None) -> Dict:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    items = load_jsonl(inputs_path)
    if max_items is not None:
        items = items[:max_items]
    existing = {}
    if predictions_path.exists():
        existing = {row["item_id"]: row for row in load_jsonl(predictions_path)}
    pending = [item for item in items if item["item_id"] not in existing]
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    completed = 0
    errors = []
    lock = threading.Lock()
    mode = "a" if predictions_path.exists() else "w"
    with predictions_path.open(mode, encoding="utf-8") as out:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(call_one, client, model, item, temperature, max_retries): item
                for item in pending
            }
            for future in as_completed(futures):
                item = futures[future]
                try:
                    row = future.result()
                    with lock:
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out.flush()
                    completed += 1
                    if completed % 50 == 0 or completed == len(pending):
                        print(f"generated={completed}/{len(pending)}", flush=True)
                except Exception as exc:
                    errors.append({"item_id": item["item_id"], "error": str(exc)})
                    print(f"ERROR {item['item_id']}: {exc}", flush=True)
    return {"requested": len(items), "existing": len(existing), "generated": completed,
            "errors": errors}


def _sigmoid(value: float) -> float:
    value = max(-40.0, min(40.0, value))
    return 1.0 / (1.0 + math.exp(-value))


def _bootstrap_ci(values: List[float], seed: int, n_boot: int = 3000):
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=float)
    if len(array) == 0:
        return None, None
    means = np.mean(rng.choice(array, size=(n_boot, len(array)), replace=True), axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def evaluate(predictions_path: Path, output_dir: Path, tau: float,
             contamination_threshold: float) -> Dict:
    from sentence_transformers import SentenceTransformer

    rows = load_jsonl(predictions_path)
    current_inputs = {
        row["item_id"]: row for row in load_jsonl(output_dir / "decision_inputs.jsonl")
    }
    for row in rows:
        source = current_inputs[row["item_id"]]
        for field in ("original_ED", "original_RRI", "original_WEP"):
            row[field] = source[field]
    expected = 200 * 2 * 3 * 2
    if len(rows) != expected:
        raise RuntimeError(f"Expected {expected} predictions, found {len(rows)}")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    texts = []
    for row in rows:
        texts.extend([row["freeform_answer"], row["echo_answer"], row["gold_answer"]])
    embeddings = model.encode(texts, batch_size=128, normalize_embeddings=True,
                              show_progress_bar=True)
    for index, row in enumerate(rows):
        answer_vec, echo_vec, gold_vec = embeddings[index * 3:index * 3 + 3]
        sim_echo = float(np.dot(answer_vec, echo_vec))
        sim_gold = float(np.dot(answer_vec, gold_vec))
        dce = _sigmoid((sim_echo - sim_gold) / tau)
        row["similarity_echo"] = sim_echo
        row["similarity_gold"] = sim_gold
        row["DCE_v2"] = dce
        row["contaminated_v2"] = bool(dce >= contamination_threshold and sim_echo > sim_gold)
        row["gold_preferred"] = bool(sim_gold > sim_echo)

    scored_path = output_dir / "sample_level_dce_v2.jsonl"
    with scored_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    groups = {}
    for row in rows:
        key = (row["framework"], row["condition"], row["solution_arm"])
        groups.setdefault(key, []).append(row)
    summary = []
    seed = 20260825
    for key in sorted(groups):
        values = groups[key]
        dces = [r["DCE_v2"] for r in values]
        lo, hi = _bootstrap_ci(dces, seed + len(summary))
        summary.append({
            "framework": key[0], "condition": key[1], "solution_arm": key[2],
            "n": len(values),
            "ED": float(np.mean([r["original_ED"] for r in values])),
            "RRI": float(np.mean([r["original_RRI"] for r in values])),
            "WEP": float(np.mean([r["original_WEP"] for r in values])),
            "DCE_v2": float(np.mean(dces)),
            "DCE_v2_std": float(np.std(dces, ddof=1)),
            "DCE_v2_ci_low": lo, "DCE_v2_ci_high": hi,
            "contamination_rate_v2": float(np.mean([r["contaminated_v2"] for r in values])),
            "gold_preference_rate": float(np.mean([r["gold_preferred"] for r in values])),
            "mean_similarity_echo": float(np.mean([r["similarity_echo"] for r in values])),
            "mean_similarity_gold": float(np.mean([r["similarity_gold"] for r in values])),
        })

    paired = []
    by_id = {(r["sample_id"], r["framework"], r["condition"], r["solution_arm"]): r
             for r in rows}
    for framework in sorted(FRAMEWORKS):
        for condition in sorted(CONDITIONS):
            diffs = []
            for sample_id in sorted({r["sample_id"] for r in rows}):
                base = by_id[(sample_id, framework, condition, "baseline")]
                guard = by_id[(sample_id, framework, condition, "full_guard")]
                diffs.append(guard["DCE_v2"] - base["DCE_v2"])
            lo, hi = _bootstrap_ci(diffs, seed + 100 + len(paired))
            paired.append({
                "framework": framework, "condition": condition, "n": len(diffs),
                "mean_DCE_change_full_minus_baseline": float(np.mean(diffs)),
                "ci_low": lo, "ci_high": hi,
                "decreased_count": int(sum(d < 0 for d in diffs)),
                "unchanged_count": int(sum(abs(d) < 1e-12 for d in diffs)),
                "increased_count": int(sum(d > 0 for d in diffs)),
            })

    def write_csv(path: Path, data: List[Dict]):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)

    write_csv(output_dir / "aggregate_results.csv", summary)
    write_csv(output_dir / "paired_effects.csv", paired)
    aggregate = {
        "protocol": {
            "source_results": str(SOURCE_RESULTS),
            "samples": 200,
            "frameworks": sorted(FRAMEWORKS),
            "conditions": sorted(CONDITIONS),
            "arms": sorted(ARMS),
            "decision_observations": len(rows),
            "decision_per_run": 1,
            "prompt_labels_removed": True,
            "freeform_model": rows[0]["model"],
            "temperature": rows[0]["temperature"],
            "DCE_v2_formula": f"sigmoid((sim(answer,echo)-sim(answer,gold))/{tau})",
            "contamination_threshold": contamination_threshold,
        },
        "condition_summary": summary,
        "paired_effects": paired,
        "usage": {
            "calls": len(rows),
            "prompt_tokens": int(sum(r["usage"]["prompt_tokens"] for r in rows)),
            "completion_tokens": int(sum(r["usage"]["completion_tokens"] for r in rows)),
            "total_tokens": int(sum(r["usage"]["total_tokens"] for r in rows)),
        },
    }
    (output_dir / "aggregate_results.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return aggregate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("extract", "generate", "evaluate", "all"), default="all")
    parser.add_argument("--model", default="deepseek-chat")
    parser.add_argument("--base_url", default="https://api.deepseek.com")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--max_retries", type=int, default=4)
    parser.add_argument("--max_items", type=int)
    parser.add_argument("--tau", type=float, default=0.20)
    parser.add_argument("--contamination_threshold", type=float, default=0.60)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inputs = args.output_dir / "decision_inputs.jsonl"
    predictions = args.output_dir / "freeform_predictions.jsonl"
    if args.phase in {"extract", "all"}:
        print(f"extracted={extract_inputs(args.source, inputs)}", flush=True)
    if args.phase in {"generate", "all"}:
        print(json.dumps(generate(inputs, predictions, args.model, args.base_url,
                                  args.temperature, args.workers, args.max_retries,
                                  args.max_items), ensure_ascii=False), flush=True)
    if args.phase in {"evaluate", "all"}:
        aggregate = evaluate(predictions, args.output_dir, args.tau,
                             args.contamination_threshold)
        print(json.dumps(aggregate["usage"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
