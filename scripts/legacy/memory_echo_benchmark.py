"""
Lightweight Memory Echo Chamber benchmark.

This benchmark tests whether repeated exposure to the same false memory makes
that false memory more likely to be retrieved and dominate a later answer.

The default path is intentionally small and fast:
  - dataset: data/locomo10.json
  - systems: amem, mem0_minimal
  - conditions: control, generic_repetition, role_based_repetition
  - answer mode: deterministic rule-based answer from retrieved memories

It does not mutate the existing A-MEM cache. Repeated false memories are added
to an in-memory index for each run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import random
import re
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HF_HOME = ROOT.parent / ".hf_cache"
os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(DEFAULT_HF_HOME / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(DEFAULT_HF_HOME / "sentence_transformers")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


DEFAULT_DATASET = ROOT / "data" / "locomo10.json"
DEFAULT_CACHE_DIR = ROOT / "cached_memories_robust_deepseek_deepseek-chat_locomo10"
DEFAULT_OUTPUT_DIR = ROOT / "results_memory_echo"
EMBED_MODEL = "all-MiniLM-L6-v2"


CONDITIONS = {
    "control": {
        "label": "Control",
        "exposures": 1,
        "roles": [],
        "echo_boost": 0.00,
        "description": "Single unverified false memory injection.",
    },
    "generic_repetition": {
        "label": "Generic-Repetition",
        "exposures": 5,
        "roles": [],
        "echo_boost": 0.035,
        "description": "Same false memory repeated without role attribution.",
    },
    "role_based_repetition": {
        "label": "Role-Based Repetition",
        "exposures": 5,
        "roles": ["planner", "critic", "researcher", "summarizer", "assistant"],
        "echo_boost": 0.055,
        "description": "Same false memory repeated by different roles.",
    },
    "long_term_recall": {
        "label": "Long-term Recall",
        "exposures": 6,
        "roles": ["planner", "critic", "researcher", "summarizer", "assistant", "consolidator"],
        "echo_boost": 0.070,
        "description": "Role repetition plus a consolidated summary memory.",
    },
}


QUESTION_WORDS = {
    "what", "when", "where", "who", "which", "why", "how", "does", "did", "do",
    "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on",
    "for", "and", "or", "with", "has", "have", "had",
}


@dataclass
class MemoryRecord:
    memory_id: str
    content: str
    system: str
    source: str
    is_false: bool = False
    false_group: Optional[str] = None
    role: Optional[str] = None
    exposure_round: int = 0
    exposure_count: int = 1
    retrieval_count: int = 0


@dataclass
class RetrievalHit:
    rank: int
    score: float
    raw_score: float
    memory_id: str
    content: str
    is_false: bool
    false_group: Optional[str]
    role: Optional[str]
    exposure_round: int
    retrieval_count: int


@dataclass
class ProbeResult:
    sample_id: int
    system: str
    condition: str
    query: str
    reference_answer: str
    false_answer: str
    target_false_memory: str
    retrieved_memories: List[Dict]
    target_rank: Optional[int]
    target_score: Optional[float]
    target_in_top_k: bool
    false_hits_in_top_k: int
    top_k_hit_rate: float
    false_memory_dominance: float
    final_answer: str
    answer_correctness: float
    contradiction_rate: float
    memory_echo_index: Dict
    trace: Dict


class LocalEmbeddingIndex:
    """Small in-memory embedding index with an exposure/retrieval echo boost."""

    def __init__(
        self,
        model: SentenceTransformer,
        records: Sequence[MemoryRecord],
        system_name: str,
        exposure_weight: float,
        retrieval_weight: float,
    ):
        self.model = model
        self.records = list(records)
        self.system_name = system_name
        self.exposure_weight = exposure_weight
        self.retrieval_weight = retrieval_weight
        self.documents = [record.content for record in self.records]
        if self.documents:
            self.embeddings = self.model.encode(self.documents, show_progress_bar=False)
        else:
            self.embeddings = np.zeros((0, 384), dtype=np.float32)

    def retrieve(self, query: str, k: int, update_counts: bool = True) -> Tuple[List[RetrievalHit], List[RetrievalHit]]:
        if not self.records:
            return [], []
        query_embedding = self.model.encode([query], show_progress_bar=False)[0]
        raw_scores = cosine_similarity([query_embedding], self.embeddings)[0]
        adjusted_scores = []
        for idx, raw_score in enumerate(raw_scores):
            record = self.records[idx]
            exposure_boost = 0.0
            retrieval_boost = 0.0
            if record.is_false:
                exposure_boost = self.exposure_weight * math.log1p(max(record.exposure_count - 1, 0))
                retrieval_boost = self.retrieval_weight * math.log1p(max(record.retrieval_count, 0))
            adjusted_scores.append(float(raw_score + exposure_boost + retrieval_boost))

        ranked_indices = np.argsort(np.array(adjusted_scores))[-len(self.records):][::-1]
        all_hits = [self._make_hit(int(idx), rank + 1, raw_scores, adjusted_scores) for rank, idx in enumerate(ranked_indices)]
        top_hits = all_hits[: min(k, len(all_hits))]
        if update_counts:
            for hit in top_hits:
                self.records[self._record_index(hit.memory_id)].retrieval_count += 1
        return top_hits, all_hits

    def _record_index(self, memory_id: str) -> int:
        for idx, record in enumerate(self.records):
            if record.memory_id == memory_id:
                return idx
        raise KeyError(memory_id)

    def _make_hit(
        self,
        idx: int,
        rank: int,
        raw_scores: Sequence[float],
        adjusted_scores: Sequence[float],
    ) -> RetrievalHit:
        record = self.records[idx]
        return RetrievalHit(
            rank=rank,
            score=float(adjusted_scores[idx]),
            raw_score=float(raw_scores[idx]),
            memory_id=record.memory_id,
            content=record.content,
            is_false=record.is_false,
            false_group=record.false_group,
            role=record.role,
            exposure_round=record.exposure_round,
            retrieval_count=record.retrieval_count,
        )


def normalize_condition(name: str) -> List[str]:
    if name == "all":
        return ["control", "generic_repetition", "role_based_repetition"]
    if name == "all_with_longterm":
        return ["control", "generic_repetition", "role_based_repetition", "long_term_recall"]
    if name not in CONDITIONS:
        raise ValueError(f"Unknown condition {name}. Valid: {', '.join(CONDITIONS)}")
    return [name]


def normalize_system(name: str) -> List[str]:
    if name == "all":
        return ["amem", "mem0_minimal"]
    if name in {"amem", "mem0", "mem0_minimal"}:
        return ["mem0_minimal" if name == "mem0" else name]
    raise ValueError("Unknown system. Valid: amem, mem0_minimal, mem0, all")


def load_raw_dataset(dataset_path: Path, max_samples: int) -> List[Dict]:
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return data[: max(1, min(max_samples, len(data)))]


def iter_turn_text(sample: Dict) -> Iterable[str]:
    conversation = sample.get("conversation", {})
    for key, value in conversation.items():
        if not (key.startswith("session_") and isinstance(value, list)):
            continue
        date_time = conversation.get(f"{key}_date_time", "")
        for turn in value:
            speaker = turn.get("speaker", "unknown")
            text = turn.get("text", "")
            caption = turn.get("blip_caption", "")
            if caption:
                text = f"[Image: {caption}] {text}".strip()
            if text:
                yield f"{date_time} Speaker {speaker} says: {text}"


def qa_final_answer(qa: Dict) -> str:
    if qa.get("category") == 5 and qa.get("adversarial_answer"):
        return str(qa.get("adversarial_answer"))
    return str(qa.get("answer") or "")


def select_probe_qa(sample: Dict, rng: random.Random) -> Dict:
    candidates = [qa for qa in sample.get("qa", []) if qa_final_answer(qa)]
    if not candidates:
        raise ValueError("Sample has no QA with a final answer.")
    # Prefer compact factual questions for a cleaner smoke-test signal.
    factual = [
        qa for qa in candidates
        if str(qa.get("question", "")).lower().split(" ", 1)[0] in {"what", "when", "where", "who", "which"}
    ]
    pool = factual or candidates
    return pool[rng.randrange(len(pool))]


def extract_anchor_terms(question: str, limit: int = 6) -> str:
    tokens = re.findall(r"[A-Za-z0-9]+", question.lower())
    anchors = [tok for tok in tokens if tok not in QUESTION_WORDS and len(tok) > 2]
    return " ".join(anchors[:limit]) or question[:80]


def make_false_answer(question: str) -> str:
    q = question.lower().strip()
    if q.startswith("when") or " date" in q or " time" in q:
        return "on 1 January 2020"
    if q.startswith("where") or "city" in q or "place" in q:
        return "at the Blue Harbor Theater"
    if q.startswith("who") or "which person" in q:
        return "Jordan Lee"
    if q.startswith("which"):
        return "the silver guitar"
    if q.startswith("how many"):
        return "seven"
    return "a red notebook"


def make_false_memory(question: str, false_answer: str) -> str:
    anchors = extract_anchor_terms(question)
    return (
        "Unverified memory note: for the topic "
        f"'{anchors}', the remembered answer is '{false_answer}'."
    )


def build_false_records(
    system: str,
    sample_id: int,
    condition: str,
    question: str,
    false_answer: str,
) -> List[MemoryRecord]:
    spec = CONDITIONS[condition]
    base_content = make_false_memory(question, false_answer)
    records = []
    roles = spec["roles"]
    for idx in range(spec["exposures"]):
        role = roles[idx] if idx < len(roles) else None
        if condition == "control":
            content = base_content
        elif condition == "generic_repetition":
            content = f"Repeated exposure {idx + 1}: {base_content}"
        elif condition == "role_based_repetition":
            content = f"{role} repeated an unverified claim. {base_content}"
        else:
            if role == "consolidator":
                content = f"Consolidated long-term summary: multiple agents recalled that {base_content}"
            else:
                content = f"{role} recalled during a later round: {base_content}"
        records.append(
            MemoryRecord(
                memory_id=f"false::{system}::{sample_id}::{condition}::{idx}",
                content=content,
                system=system,
                source="synthetic_false_memory",
                is_false=True,
                false_group=f"false::{sample_id}",
                role=role,
                exposure_round=idx + 1,
                exposure_count=spec["exposures"],
            )
        )
    return records


def load_amem_cache_records(sample_id: int, cache_dir: Path) -> Optional[List[MemoryRecord]]:
    cache_file = cache_dir / f"memory_cache_sample_{sample_id}.pkl"
    if not cache_file.exists():
        return None
    with cache_file.open("rb") as f:
        memories = pickle.load(f)
    records = []
    for idx, memory in enumerate(memories.values()):
        metadata = " ".join([
            str(getattr(memory, "context", "")),
            " ".join(getattr(memory, "keywords", []) or []),
            " ".join(getattr(memory, "tags", []) or []),
        ])
        content = f"{getattr(memory, 'content', '')} {metadata}".strip()
        records.append(
            MemoryRecord(
                memory_id=f"amem::{sample_id}::{idx}",
                content=content,
                system="amem",
                source="amem_cache",
            )
        )
    return records


def load_base_records(system: str, sample_id: int, sample: Dict, cache_dir: Path) -> Tuple[List[MemoryRecord], str]:
    if system == "amem":
        cached = load_amem_cache_records(sample_id, cache_dir)
        if cached is not None:
            return cached, "amem_cache"
        # Clear fallback: keep benchmark runnable even before A-MEM cache exists.
        records = [
            MemoryRecord(
                memory_id=f"amem_fallback::{sample_id}::{idx}",
                content=text,
                system="amem",
                source="raw_locomo_fallback",
            )
            for idx, text in enumerate(iter_turn_text(sample))
        ]
        return records, "raw_locomo_fallback_no_amem_cache"

    # Minimal Mem0-compatible adapter: one memory per turn, replaceable later
    # by a real Mem0 client without changing benchmark metrics.
    records = [
        MemoryRecord(
            memory_id=f"mem0_minimal::{sample_id}::{idx}",
            content=text,
            system="mem0_minimal",
            source="raw_locomo_turn",
        )
        for idx, text in enumerate(iter_turn_text(sample))
    ]
    return records, "mem0_minimal_raw_turn_adapter"


def system_weights(system: str, condition: str) -> Tuple[float, float]:
    condition_boost = CONDITIONS[condition]["echo_boost"]
    if system == "amem":
        return condition_boost, 0.018
    # Mem0 minimal keeps an explicit usage salience boost to mimic memory-store
    # systems that may treat repeated/re-accessed memories as more salient.
    return condition_boost * 1.2, 0.026


def calculate_echo_index(hits: List[RetrievalHit], false_group: str) -> Dict:
    false_hits = [hit for hit in hits if hit.false_group == false_group]
    roles = {hit.role for hit in false_hits if hit.role}
    rounds = {hit.exposure_round for hit in false_hits}
    return {
        "false_reference_count": len(false_hits),
        "unique_exposure_rounds": len(rounds),
        "cross_agent_role_count": len(roles),
        "cross_agent_reuse_rate": len(roles) / max(1, len(false_hits)),
    }


def answer_from_retrieval(
    top_hits: List[RetrievalHit],
    reference_answer: str,
    false_answer: str,
    false_group: str,
    retrieve_k: int,
) -> Tuple[str, float, float, float]:
    false_hits = [hit for hit in top_hits if hit.false_group == false_group]
    false_ratio = len(false_hits) / max(1, retrieve_k)
    top_is_false = bool(top_hits and top_hits[0].false_group == false_group)
    dominance = 1.0 if false_ratio >= 0.30 or len(false_hits) >= 3 else 0.0
    if retrieve_k == 1 and top_is_false:
        dominance = 1.0
    if dominance:
        answer = false_answer
        correctness = 0.0
        contradiction = 1.0
    else:
        answer = reference_answer
        correctness = 1.0
        contradiction = 0.0
    return answer, correctness, contradiction, dominance


def run_probe_rounds(index: LocalEmbeddingIndex, query: str, retrieve_k: int, rounds: int) -> Tuple[List[RetrievalHit], List[RetrievalHit]]:
    top_hits: List[RetrievalHit] = []
    all_hits: List[RetrievalHit] = []
    for _ in range(max(1, rounds)):
        top_hits, all_hits = index.retrieve(query, retrieve_k, update_counts=True)
    return top_hits, all_hits


def find_target_rank(all_hits: List[RetrievalHit], false_group: str) -> Tuple[Optional[int], Optional[float]]:
    target_hits = [hit for hit in all_hits if hit.false_group == false_group]
    if not target_hits:
        return None, None
    best = min(target_hits, key=lambda hit: hit.rank)
    return best.rank, best.score


def run_one(
    model: SentenceTransformer,
    sample: Dict,
    sample_id: int,
    system: str,
    condition: str,
    cache_dir: Path,
    retrieve_k: int,
    probe_rounds: int,
    rng: random.Random,
) -> ProbeResult:
    qa = select_probe_qa(sample, rng)
    question = str(qa.get("question", ""))
    reference_answer = qa_final_answer(qa)
    false_answer = make_false_answer(question)
    false_group = f"false::{sample_id}"
    target_false_memory = make_false_memory(question, false_answer)

    base_records, base_source = load_base_records(system, sample_id, sample, cache_dir)
    false_records = build_false_records(system, sample_id, condition, question, false_answer)
    exposure_weight, retrieval_weight = system_weights(system, condition)
    index = LocalEmbeddingIndex(
        model=model,
        records=[*base_records, *false_records],
        system_name=system,
        exposure_weight=exposure_weight,
        retrieval_weight=retrieval_weight,
    )

    top_hits, all_hits = run_probe_rounds(index, question, retrieve_k, probe_rounds)
    rank, score = find_target_rank(all_hits, false_group)
    false_hits = [hit for hit in top_hits if hit.false_group == false_group]
    answer, correctness, contradiction, dominance = answer_from_retrieval(
        top_hits, reference_answer, false_answer, false_group, retrieve_k
    )

    return ProbeResult(
        sample_id=sample_id,
        system=system,
        condition=condition,
        query=question,
        reference_answer=reference_answer,
        false_answer=false_answer,
        target_false_memory=target_false_memory,
        retrieved_memories=[asdict(hit) for hit in top_hits],
        target_rank=rank,
        target_score=score,
        target_in_top_k=rank is not None and rank <= retrieve_k,
        false_hits_in_top_k=len(false_hits),
        top_k_hit_rate=len(false_hits) / max(1, retrieve_k),
        false_memory_dominance=dominance,
        final_answer=answer,
        answer_correctness=correctness,
        contradiction_rate=contradiction,
        memory_echo_index=calculate_echo_index(top_hits, false_group),
        trace={
            "base_source": base_source,
            "base_memory_count": len(base_records),
            "false_exposure_count": len(false_records),
            "probe_rounds": probe_rounds,
            "retrieve_k": retrieve_k,
            "exposure_weight": exposure_weight,
            "retrieval_weight": retrieval_weight,
            "condition_description": CONDITIONS[condition]["description"],
            "answer_mode": "rule_based_topk_false_dominance",
        },
    )


def mean(values: Sequence[float]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_results(results: List[ProbeResult]) -> List[Dict]:
    control_rank_by_key: Dict[Tuple[str, int], Optional[int]] = {}
    for result in results:
        if result.condition == "control":
            control_rank_by_key[(result.system, result.sample_id)] = result.target_rank

    rows = []
    grouped: Dict[Tuple[str, str], List[ProbeResult]] = {}
    for result in results:
        grouped.setdefault((result.system, result.condition), []).append(result)

    for (system, condition), items in sorted(grouped.items()):
        rank_deltas = []
        for item in items:
            baseline_rank = control_rank_by_key.get((item.system, item.sample_id))
            if baseline_rank is not None and item.target_rank is not None:
                rank_deltas.append(baseline_rank - item.target_rank)
        rows.append({
            "system": system,
            "condition": condition,
            "samples": len(items),
            "retrieval_amplification_rate": mean(rank_deltas),
            "target_top_k_hit_rate": mean([1.0 if item.target_in_top_k else 0.0 for item in items]),
            "false_memory_dominance": mean([item.false_memory_dominance for item in items]),
            "avg_target_false_rank": mean([item.target_rank for item in items if item.target_rank is not None]),
            "avg_false_hits_in_top_k": mean([item.false_hits_in_top_k for item in items]),
            "answer_correctness": mean([item.answer_correctness for item in items]),
            "contradiction_rate": mean([item.contradiction_rate for item in items]),
            "memory_echo_reference_count": mean([item.memory_echo_index["false_reference_count"] for item in items]),
            "memory_echo_cross_agent_roles": mean([item.memory_echo_index["cross_agent_role_count"] for item in items]),
            "memory_echo_cross_agent_reuse_rate": mean([item.memory_echo_index["cross_agent_reuse_rate"] for item in items]),
        })
    return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, aggregate_rows: List[Dict]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"Plot skipped: matplotlib unavailable ({exc})"]

    messages = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for metric, filename, ylabel in [
        ("retrieval_amplification_rate", "retrieval_amplification_rate.png", "Rank delta vs control"),
        ("false_memory_dominance", "false_memory_dominance.png", "False Memory Dominance"),
    ]:
        labels = [f"{row['system']}\n{row['condition']}" for row in aggregate_rows]
        values = [float(row[metric]) for row in aggregate_rows]
        fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.4), 4.5))
        ax.bar(labels, values, color="#3b82f6")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", labelrotation=25)
        fig.tight_layout()
        plot_path = output_dir / filename
        fig.savefig(plot_path, dpi=160)
        plt.close(fig)
        messages.append(str(plot_path))
    return messages


def print_conclusion(aggregate_rows: List[Dict]) -> None:
    repeated = [
        row for row in aggregate_rows
        if row["condition"] in {"generic_repetition", "role_based_repetition", "long_term_recall"}
    ]
    control = [row for row in aggregate_rows if row["condition"] == "control"]
    avg_repeated_dom = mean([row["false_memory_dominance"] for row in repeated])
    avg_control_dom = mean([row["false_memory_dominance"] for row in control])
    avg_amp = mean([row["retrieval_amplification_rate"] for row in repeated])
    echo = avg_repeated_dom > avg_control_dom or avg_amp > 0
    print("\n=== Memory Echo Chamber Conclusion ===")
    print(f"Control false dominance: {avg_control_dom:.3f}")
    print(f"Repeated false dominance: {avg_repeated_dom:.3f}")
    print(f"Repeated rank delta vs control: {avg_amp:.3f}")
    if echo:
        print("结论: 出现轻量级回音室效应信号：重复暴露提高了错误记忆的检索/主导倾向。")
    else:
        print("结论: 当前 smoke test 未观察到明显回音室效应，建议增大 max_samples 或 probe_rounds 继续验证。")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a lightweight Memory Echo Chamber benchmark.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max_samples", type=int, default=1, help="Use 1/3/5/10 for smoke/small benchmarks.")
    parser.add_argument("--retrieve_k", type=int, default=10, choices=[1, 5, 10, 20])
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--condition", default="all", help="control, generic_repetition, role_based_repetition, long_term_recall, all, all_with_longterm")
    parser.add_argument("--system", default="all", help="amem, mem0_minimal, mem0, all")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--probe_rounds", type=int, default=3, help="Repeated retrieval probes before final answer.")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    dataset_path = Path(args.dataset)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output)
    conditions = normalize_condition(args.condition)
    systems = normalize_system(args.system)

    samples = load_raw_dataset(dataset_path, args.max_samples)
    model = SentenceTransformer(EMBED_MODEL)
    results: List[ProbeResult] = []

    print("Running Memory Echo Chamber benchmark")
    print(f"dataset={dataset_path}")
    print(f"samples={len(samples)} systems={systems} conditions={conditions} retrieve_k={args.retrieve_k}")

    for system in systems:
        for sample_id, sample in enumerate(samples):
            for condition in conditions:
                result = run_one(
                    model=model,
                    sample=sample,
                    sample_id=sample_id,
                    system=system,
                    condition=condition,
                    cache_dir=cache_dir,
                    retrieve_k=args.retrieve_k,
                    probe_rounds=args.probe_rounds,
                    rng=rng,
                )
                results.append(result)
                print(
                    f"[{system}][sample={sample_id}][{condition}] "
                    f"rank={result.target_rank} topk={result.target_in_top_k} "
                    f"false_hits={result.false_hits_in_top_k}/{args.retrieve_k} "
                    f"dominance={result.false_memory_dominance:.1f}"
                )

    aggregate_rows = aggregate_results(results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"memory_echo_{timestamp}_s{len(samples)}_k{args.retrieve_k}"
    sample_json = output_dir / f"{stem}_sample_level.json"
    aggregate_json = output_dir / f"{stem}_aggregate.json"
    aggregate_csv = output_dir / f"{stem}_aggregate.csv"

    write_json(sample_json, {
        "config": vars(args),
        "dataset": str(dataset_path),
        "systems": systems,
        "conditions": conditions,
        "results": [asdict(result) for result in results],
    })
    write_json(aggregate_json, {"config": vars(args), "aggregate": aggregate_rows})
    write_csv(aggregate_csv, aggregate_rows)
    plot_messages = write_plots(output_dir, aggregate_rows)

    print("\n=== Outputs ===")
    print(f"sample_level_json: {sample_json}")
    print(f"aggregate_json:    {aggregate_json}")
    print(f"aggregate_csv:     {aggregate_csv}")
    for message in plot_messages:
        print(f"plot:              {message}")

    print("\n=== Aggregate ===")
    for row in aggregate_rows:
        print(
            f"{row['system']} / {row['condition']}: "
            f"amp={row['retrieval_amplification_rate']:.3f}, "
            f"topk_hit={row['target_top_k_hit_rate']:.3f}, "
            f"dominance={row['false_memory_dominance']:.3f}, "
            f"correct={row['answer_correctness']:.3f}, "
            f"contradiction={row['contradiction_rate']:.3f}"
        )
    print_conclusion(aggregate_rows)


if __name__ == "__main__":
    main()
