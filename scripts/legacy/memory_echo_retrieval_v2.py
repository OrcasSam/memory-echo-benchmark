"""
Memory Echo Chamber v2: retrieval-only benchmark.

This version intentionally does NOT call any LLM for answer generation or judge.
It only evaluates memory backend construction, retrieval behavior, claim-level
clustering, and echo metrics.

Systems:
  - amem: reuses existing A-MEM memory cache
  - mem0: uses official mem0ai Memory.add(..., infer=False) and Memory.search()
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "data" / "locomo10.json"
DEFAULT_CACHE_DIR = ROOT / "cached_memories_robust_deepseek_deepseek-chat_locomo10"
DEFAULT_OUTPUT_DIR = ROOT / "results_memory_echo_v2"
DEFAULT_MEM0_DIR = ROOT / ".mem0_echo_v2"
DEFAULT_HF_HOME = ROOT.parent / ".hf_cache"
EMBED_MODEL = "all-MiniLM-L6-v2"
EMBED_DIM = 384

os.environ["HF_HOME"] = str(DEFAULT_HF_HOME)
os.environ["TRANSFORMERS_CACHE"] = str(DEFAULT_HF_HOME / "transformers")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = str(DEFAULT_HF_HOME / "sentence_transformers")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT.parent / ".matplotlib"))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("MEM0_TELEMETRY", "False")
os.environ["MEM0_DIR"] = str(DEFAULT_MEM0_DIR)
os.environ["HOME"] = str(ROOT.parent)
os.environ["USERPROFILE"] = str(ROOT.parent)

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


CONDITIONS = {
    "control": {
        "exposures": 1,
        "roles": ["single_source"],
        "sources": ["control"],
        "include_summary": False,
    },
    "generic_repetition": {
        "exposures": 5,
        "roles": ["generic"] * 5,
        "sources": [f"generic_round_{i}" for i in range(1, 6)],
        "include_summary": False,
    },
    "role_based_repetition": {
        "exposures": 5,
        "roles": ["planner", "critic", "researcher", "summarizer", "assistant"],
        "sources": ["agent_planner", "agent_critic", "agent_researcher", "agent_summarizer", "agent_assistant"],
        "include_summary": False,
    },
    "long_term_recall": {
        "exposures": 6,
        "roles": ["planner", "critic", "researcher", "summarizer", "assistant", "consolidator"],
        "sources": ["agent_planner", "agent_critic", "agent_researcher", "agent_summarizer", "agent_assistant", "delayed_summary"],
        "include_summary": True,
    },
}

QUESTION_WORDS = {
    "what", "when", "where", "who", "which", "why", "how", "does", "did", "do",
    "is", "are", "was", "were", "the", "a", "an", "to", "of", "in", "on",
    "for", "and", "or", "with", "has", "have", "had", "kind", "type",
}


@dataclass
class MemoryRecord:
    memory_id: str
    content: str
    system: str
    source: str
    role: str
    round_id: int
    claim_cluster_id: str
    is_false_claim: bool = False
    metadata: Optional[Dict] = None


@dataclass
class RetrievalHit:
    rank: int
    score: float
    memory_id: str
    content: str
    source: str
    role: str
    round_id: int
    claim_cluster_id: str
    is_false_claim: bool


@dataclass
class SampleResult:
    sample_id: int
    system: str
    condition: str
    retrieve_k: int
    query: str
    target_false_claim: str
    target_claim_cluster_id: str
    retrieved_memories: List[Dict]
    target_topk_hit: float
    target_false_rank: Optional[int]
    target_claim_count_topk: int
    target_claim_roles_topk: int
    target_claim_sources_topk: int
    target_claim_rounds_topk: int
    echo_density: float
    cross_source_echo_index: float
    echo_growth_rate: float
    false_claim_occupancy: float
    false_claim_copy_count: int
    claim_level_topk_share: float
    trace: Dict


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def normalize_text(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.lower()))


def claim_cluster_for_text(text: str) -> str:
    tokens = normalize_text(text).split()
    informative = [tok for tok in tokens if tok not in QUESTION_WORDS and len(tok) > 2]
    return "claim::" + stable_hash(" ".join(informative[:16]) or normalize_text(text)[:120])


def normalize_condition(name: str) -> List[str]:
    if name == "all":
        return ["control", "generic_repetition", "role_based_repetition"]
    if name == "all_with_longterm":
        return ["control", "generic_repetition", "role_based_repetition", "long_term_recall"]
    if name not in CONDITIONS:
        raise ValueError(f"Unknown condition {name}")
    return [name]


def normalize_system(name: str) -> List[str]:
    if name == "all":
        return ["amem", "mem0"]
    if name in {"amem", "mem0"}:
        return [name]
    raise ValueError("Unknown system. Valid: amem, mem0, all")


def parse_k_sweep(values: str) -> List[int]:
    out = []
    for value in values.split(","):
        k = int(value.strip())
        if k not in {1, 5, 10, 20}:
            raise ValueError("--retrieve_k values must be in 1,5,10,20")
        out.append(k)
    return out


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
    factual = [
        qa for qa in candidates
        if str(qa.get("question", "")).lower().split(" ", 1)[0] in {"what", "when", "where", "who", "which"}
    ]
    pool = factual or candidates
    if not pool:
        raise ValueError("No usable QA in sample")
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
    if q.startswith("who"):
        return "Jordan Lee"
    if q.startswith("which"):
        return "the silver guitar"
    if q.startswith("how many"):
        return "seven"
    return "a red notebook"


def make_false_claim(question: str, false_answer: str) -> Tuple[str, str]:
    anchors = extract_anchor_terms(question)
    claim = f"unverified claim: topic '{anchors}' has answer '{false_answer}'"
    cluster_id = "false_claim::" + stable_hash(f"{anchors}::{false_answer}")
    return claim, cluster_id


def build_false_records(system: str, sample_id: int, condition: str, question: str) -> Tuple[List[MemoryRecord], str, str]:
    false_answer = make_false_answer(question)
    claim, cluster_id = make_false_claim(question, false_answer)
    spec = CONDITIONS[condition]
    records = []
    for idx in range(spec["exposures"]):
        role = spec["roles"][idx]
        source = spec["sources"][idx]
        if condition == "control":
            content = claim
        elif condition == "generic_repetition":
            content = f"{claim}. Repeated exposure {idx + 1}."
        elif condition == "role_based_repetition":
            content = f"{claim}. Source role: {role} repeated this unverified memory."
        else:
            if role == "consolidator":
                content = f"{claim}. Delayed consolidation summary from multiple agents."
            else:
                content = f"{claim}. Delayed recall source role: {role}."
        records.append(MemoryRecord(
            memory_id=f"false::{system}::{sample_id}::{condition}::{idx}",
            content=content,
            system=system,
            source=source,
            role=role,
            round_id=idx + 1,
            claim_cluster_id=cluster_id,
            is_false_claim=True,
            metadata={"condition": condition, "synthetic": True},
        ))
    return records, claim, cluster_id


def load_amem_records(sample_id: int, cache_dir: Path) -> List[MemoryRecord]:
    cache_file = cache_dir / f"memory_cache_sample_{sample_id}.pkl"
    records = []
    if cache_file.exists():
        with cache_file.open("rb") as f:
            memories = pickle.load(f)
        for idx, memory in enumerate(memories.values()):
            metadata = " ".join([
                str(getattr(memory, "context", "")),
                " ".join(getattr(memory, "keywords", []) or []),
                " ".join(getattr(memory, "tags", []) or []),
            ])
            content = f"{getattr(memory, 'content', '')} {metadata}".strip()
            records.append(MemoryRecord(
                memory_id=f"amem::{sample_id}::{idx}",
                content=content,
                system="amem",
                source="amem_cache",
                role="amem_memory",
                round_id=0,
                claim_cluster_id=claim_cluster_for_text(content),
                is_false_claim=False,
            ))
        return records
    raise FileNotFoundError(f"A-MEM cache missing: {cache_file}")


class LocalClaimIndex:
    def __init__(self, model: SentenceTransformer, records: Sequence[MemoryRecord]):
        self.model = model
        self.records = list(records)
        self.docs = [record.content for record in self.records]
        self.embeddings = self.model.encode(self.docs, show_progress_bar=False) if self.docs else np.zeros((0, EMBED_DIM))

    def search(self, query: str, k: int) -> Tuple[List[RetrievalHit], List[RetrievalHit]]:
        if not self.docs:
            return [], []
        q = self.model.encode([query], show_progress_bar=False)[0]
        scores = cosine_similarity([q], self.embeddings)[0]
        order = np.argsort(scores)[-len(self.records):][::-1]
        all_hits = []
        for rank, idx in enumerate(order, start=1):
            record = self.records[int(idx)]
            all_hits.append(RetrievalHit(
                rank=rank,
                score=float(scores[int(idx)]),
                memory_id=record.memory_id,
                content=record.content,
                source=record.source,
                role=record.role,
                round_id=record.round_id,
                claim_cluster_id=record.claim_cluster_id,
                is_false_claim=record.is_false_claim,
            ))
        return all_hits[: min(k, len(all_hits))], all_hits


class Mem0OfficialAdapter:
    """Official mem0ai adapter with infer=False, so no LLM is called."""

    def __init__(self, root_dir: Path, collection_name: str):
        from mem0 import Memory
        self.root_dir = root_dir
        self.collection_name = collection_name
        self.user_id = f"user_{collection_name}"
        self.records_by_memory_text: Dict[str, MemoryRecord] = {}
        self.records_by_added_id: Dict[str, MemoryRecord] = {}
        qdrant_path = root_dir / "qdrant"
        history_path = root_dir / "history.db"
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "path": str(qdrant_path),
                    "collection_name": collection_name,
                    "embedding_model_dims": EMBED_DIM,
                    "on_disk": False,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": EMBED_MODEL, "embedding_dims": EMBED_DIM},
            },
            # Constructed but not used because add(..., infer=False).
            "llm": {"provider": "ollama", "config": {"model": "no_llm_used"}},
            "history_db_path": str(history_path),
        }
        self.memory = Memory.from_config(config)

    def add_records(self, records: Sequence[MemoryRecord]) -> None:
        for record in records:
            metadata = {
                "memory_id": record.memory_id,
                "source": record.source,
                "role": record.role,
                "round_id": record.round_id,
                "claim_cluster_id": record.claim_cluster_id,
                "is_false_claim": record.is_false_claim,
            }
            result = self.memory.add(
                record.content,
                user_id=self.user_id,
                infer=False,
                metadata=metadata,
            )
            self.records_by_memory_text[record.content] = record
            for item in result.get("results", []):
                added_id = item.get("id")
                if added_id:
                    self.records_by_added_id[added_id] = record

    def search(self, query: str, k: int) -> Tuple[List[RetrievalHit], List[RetrievalHit]]:
        result = self.memory.search(
            query,
            filters={"user_id": self.user_id},
            top_k=max(k, 100),
            threshold=0.0,
            rerank=False,
        )
        rows = result.get("results", [])
        hits = []
        for rank, row in enumerate(rows, start=1):
            metadata = row.get("metadata") or {}
            record = self.records_by_added_id.get(row.get("id")) or self.records_by_memory_text.get(row.get("memory", ""))
            if record is None:
                record = MemoryRecord(
                    memory_id=metadata.get("memory_id", row.get("id", f"mem0_unknown_{rank}")),
                    content=row.get("memory", ""),
                    system="mem0",
                    source=metadata.get("source", "mem0_unknown"),
                    role=metadata.get("role", "mem0_unknown"),
                    round_id=int(metadata.get("round_id", 0) or 0),
                    claim_cluster_id=metadata.get("claim_cluster_id") or claim_cluster_for_text(row.get("memory", "")),
                    is_false_claim=bool(metadata.get("is_false_claim", False)),
                )
            hits.append(RetrievalHit(
                rank=rank,
                score=float(row.get("score", 0.0) or 0.0),
                memory_id=record.memory_id,
                content=record.content,
                source=record.source,
                role=record.role,
                round_id=record.round_id,
                claim_cluster_id=record.claim_cluster_id,
                is_false_claim=record.is_false_claim,
            ))
        return hits[: min(k, len(hits))], hits

    def close(self) -> None:
        close = getattr(self.memory, "close", None)
        if callable(close):
            close()


def build_base_records(system: str, sample_id: int, sample: Dict, cache_dir: Path) -> List[MemoryRecord]:
    if system == "amem":
        return load_amem_records(sample_id, cache_dir)
    records = []
    for idx, text in enumerate(iter_turn_text(sample)):
        records.append(MemoryRecord(
            memory_id=f"mem0::{sample_id}::{idx}",
            content=text,
            system="mem0",
            source="locomo_turn",
            role="conversation",
            round_id=0,
            claim_cluster_id=claim_cluster_for_text(text),
            is_false_claim=False,
        ))
    return records


def calculate_metrics(top_hits: List[RetrievalHit], all_hits: List[RetrievalHit], target_cluster: str, k: int, ed_history: List[float]) -> Dict:
    target_top = [hit for hit in top_hits if hit.claim_cluster_id == target_cluster]
    false_top = [hit for hit in top_hits if hit.is_false_claim]
    target_all = [hit for hit in all_hits if hit.claim_cluster_id == target_cluster]
    target_rank = min([hit.rank for hit in target_all], default=None)
    roles = {hit.role for hit in target_top}
    sources = {hit.source for hit in target_top}
    rounds = {hit.round_id for hit in target_top}
    echo_density = len(target_top) / max(1, k)
    csei = (len(roles) + len(sources) + len(rounds)) / max(1, 3 * len(target_top))
    egr = 0.0
    if len(ed_history) > 1:
        egr = (ed_history[-1] - ed_history[0]) / (len(ed_history) - 1)
    return {
        "target_topk_hit": 1.0 if target_top else 0.0,
        "target_false_rank": target_rank,
        "target_claim_count_topk": len(target_top),
        "target_claim_roles_topk": len(roles),
        "target_claim_sources_topk": len(sources),
        "target_claim_rounds_topk": len(rounds),
        "echo_density": echo_density,
        "cross_source_echo_index": csei,
        "echo_growth_rate": egr,
        "false_claim_occupancy": len(false_top) / max(1, k),
        "false_claim_copy_count": len(false_top),
        "claim_level_topk_share": echo_density,
    }


def build_sample_result(
    sample_id: int,
    system: str,
    condition: str,
    retrieve_k: int,
    query: str,
    target_claim: str,
    target_cluster: str,
    top_hits: List[RetrievalHit],
    all_hits: List[RetrievalHit],
    ed_history: List[float],
    base_count: int,
    false_count: int,
    probe_rounds: int,
) -> SampleResult:
    metrics = calculate_metrics(top_hits, all_hits, target_cluster, retrieve_k, ed_history)
    return SampleResult(
        sample_id=sample_id,
        system=system,
        condition=condition,
        retrieve_k=retrieve_k,
        query=query,
        target_false_claim=target_claim,
        target_claim_cluster_id=target_cluster,
        retrieved_memories=[asdict(hit) for hit in top_hits],
        trace={
            "base_memory_count": base_count,
            "false_memory_count": false_count,
            "probe_rounds": probe_rounds,
            "ed_history": ed_history,
            "no_llm_generation": True,
            "no_llm_judge": True,
            "mem0_infer_false": system == "mem0",
        },
        **metrics,
    )


def run_retrieval_probe_multi_k(system: str, sample: Dict, sample_id: int, condition: str, retrieve_ks: Sequence[int], cache_dir: Path, model: SentenceTransformer, rng: random.Random, mem0_root: Path, probe_rounds: int) -> List[SampleResult]:
    qa = select_probe_qa(sample, rng)
    query = str(qa.get("question", ""))
    false_records, target_claim, target_cluster = build_false_records(system, sample_id, condition, query)
    base_records = build_base_records(system, sample_id, sample, cache_dir)
    records = [*base_records, *false_records]
    max_k = max(retrieve_ks)
    ed_histories = {k: [] for k in retrieve_ks}
    all_hits: List[RetrievalHit] = []
    max_top_hits: List[RetrievalHit] = []

    if system == "amem":
        index = LocalClaimIndex(model, records)
        for _ in range(max(1, probe_rounds)):
            max_top_hits, all_hits = index.search(query, max_k)
            for k in retrieve_ks:
                top_k_hits = all_hits[:k]
                ed_histories[k].append(len([h for h in top_k_hits if h.claim_cluster_id == target_cluster]) / max(1, k))
    else:
        run_dir = mem0_root / f"s{sample_id}_{condition}_ks{'-'.join(map(str, retrieve_ks))}_{stable_hash(query, 6)}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        adapter = Mem0OfficialAdapter(run_dir, collection_name=f"echo_{sample_id}_{condition}_{stable_hash(query, 6)}")
        try:
            adapter.add_records(records)
            for _ in range(max(1, probe_rounds)):
                max_top_hits, all_hits = adapter.search(query, max_k)
                for k in retrieve_ks:
                    top_k_hits = all_hits[:k]
                    ed_histories[k].append(len([h for h in top_k_hits if h.claim_cluster_id == target_cluster]) / max(1, k))
        finally:
            adapter.close()

    results = []
    for k in retrieve_ks:
        top_k_hits = all_hits[:k]
        results.append(build_sample_result(
            sample_id=sample_id,
            system=system,
            condition=condition,
            retrieve_k=k,
            query=query,
            target_claim=target_claim,
            target_cluster=target_cluster,
            top_hits=top_k_hits,
            all_hits=all_hits,
            ed_history=ed_histories[k],
            base_count=len(base_records),
            false_count=len(false_records),
            probe_rounds=probe_rounds,
        ))
    return results


def mean(values: Sequence[Optional[float]]) -> float:
    vals = [float(v) for v in values if v is not None]
    return sum(vals) / len(vals) if vals else 0.0


def aggregate_results(results: List[SampleResult]) -> List[Dict]:
    control_by_key = {}
    for item in results:
        if item.condition == "control":
            control_by_key[(item.system, item.sample_id, item.retrieve_k)] = item
    grouped: Dict[Tuple[str, str, int], List[SampleResult]] = {}
    for item in results:
        grouped.setdefault((item.system, item.condition, item.retrieve_k), []).append(item)
    rows = []
    for (system, condition, retrieve_k), items in sorted(grouped.items()):
        rank_deltas = []
        ed_lifts = []
        csei_lifts = []
        for item in items:
            control = control_by_key.get((item.system, item.sample_id, item.retrieve_k))
            if control:
                if control.target_false_rank is not None and item.target_false_rank is not None:
                    rank_deltas.append(control.target_false_rank - item.target_false_rank)
                ed_lifts.append(item.echo_density - control.echo_density)
                csei_lifts.append(item.cross_source_echo_index - control.cross_source_echo_index)
        rows.append({
            "system": system,
            "condition": condition,
            "retrieve_k": retrieve_k,
            "samples": len(items),
            "target_topk_hit_rate": mean([x.target_topk_hit for x in items]),
            "average_target_false_rank": mean([x.target_false_rank for x in items]),
            "rank_delta_vs_control": mean(rank_deltas),
            "retrieval_amplification_rate_claim_level": mean(ed_lifts),
            "echo_density": mean([x.echo_density for x in items]),
            "echo_density_lift_vs_control": mean(ed_lifts),
            "cross_source_echo_index": mean([x.cross_source_echo_index for x in items]),
            "csei_lift_vs_control": mean(csei_lifts),
            "echo_growth_rate": mean([x.echo_growth_rate for x in items]),
            "false_claim_occupancy": mean([x.false_claim_occupancy for x in items]),
            "false_claim_copy_count": mean([x.false_claim_copy_count for x in items]),
            "target_claim_roles_topk": mean([x.target_claim_roles_topk for x in items]),
            "target_claim_sources_topk": mean([x.target_claim_sources_topk for x in items]),
            "target_claim_rounds_topk": mean([x.target_claim_rounds_topk for x in items]),
        })
    return rows


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_plots(output_dir: Path, rows: List[Dict]) -> List[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"plot skipped: {exc}"]
    paths = []
    for metric, filename, ylabel in [
        ("echo_density", "v2_echo_density_by_condition.png", "Echo Density"),
        ("cross_source_echo_index", "v2_csei_by_condition.png", "CSEI"),
    ]:
        labels = [f"{r['system']}\n{r['condition']}\nk={r['retrieve_k']}" for r in rows]
        values = [float(r[metric]) for r in rows]
        fig, ax = plt.subplots(figsize=(max(10, len(labels) * 0.75), 4.8))
        ax.bar(labels, values, color="#2563eb")
        ax.set_ylabel(ylabel)
        ax.set_title(metric.replace("_", " ").title())
        ax.tick_params(axis="x", labelrotation=35, labelsize=8)
        fig.tight_layout()
        path = output_dir / filename
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_conclusion(path: Path, rows: List[Dict]) -> str:
    control = [r for r in rows if r["condition"] == "control"]
    repeated = [r for r in rows if r["condition"] != "control"]
    ed_lift = mean([r["echo_density_lift_vs_control"] for r in repeated])
    csei_lift = mean([r["csei_lift_vs_control"] for r in repeated])
    hit = ed_lift > 0 or csei_lift > 0
    driver = "ED" if ed_lift >= csei_lift else "CSEI"
    text = (
        "# Memory Echo Retrieval v2 Conclusion\n\n"
        f"- Preliminary echo effect observed: {hit}\n"
        f"- Mean ED lift vs control: {ed_lift:.4f}\n"
        f"- Mean CSEI lift vs control: {csei_lift:.4f}\n"
        f"- Primary supporting metric: {driver}\n\n"
        "This run is retrieval-only: no final answer generation and no LLM judge were used.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory Echo Chamber v2 retrieval-only benchmark.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--max_samples", type=int, default=3)
    parser.add_argument("--retrieve_k", default="1,5,10,20", help="Comma-separated k sweep, e.g. 1,5,10,20")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--condition", default="all")
    parser.add_argument("--system", default="all")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--probe_rounds", type=int, default=3)
    parser.add_argument("--mem0_root", default=str(DEFAULT_MEM0_DIR / "runs"))
    args = parser.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    dataset_path = Path(args.dataset)
    cache_dir = Path(args.cache_dir)
    output_dir = Path(args.output)
    mem0_root = Path(args.mem0_root)
    systems = normalize_system(args.system)
    conditions = normalize_condition(args.condition)
    retrieve_ks = parse_k_sweep(str(args.retrieve_k))
    samples = load_raw_dataset(dataset_path, args.max_samples)
    model = SentenceTransformer(EMBED_MODEL)

    print("Memory Echo v2 retrieval-only benchmark")
    print(f"dataset={dataset_path} max_samples={len(samples)} systems={systems} conditions={conditions} k={retrieve_ks}")
    print("LLM generation: disabled; LLM judge: disabled; Mem0 add infer=False")

    results: List[SampleResult] = []
    for system in systems:
        for sample_id, sample in enumerate(samples):
            for condition in conditions:
                batch_results = run_retrieval_probe_multi_k(
                        system=system,
                        sample=sample,
                        sample_id=sample_id,
                        condition=condition,
                        retrieve_ks=retrieve_ks,
                        cache_dir=cache_dir,
                        model=model,
                        rng=rng,
                        mem0_root=mem0_root,
                        probe_rounds=args.probe_rounds,
                    )
                for result in batch_results:
                    results.append(result)
                    print(
                        f"[{system}][k={result.retrieve_k}][sample={sample_id}][{condition}] "
                        f"ED={result.echo_density:.3f} CSEI={result.cross_source_echo_index:.3f} "
                        f"rank={result.target_false_rank} copies={result.false_claim_copy_count}"
                    )

    aggregate = aggregate_results(results)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"memory_echo_v2_{stamp}_s{len(samples)}_k{'-'.join(map(str, retrieve_ks))}"
    sample_json = output_dir / f"{stem}_sample_level.json"
    aggregate_json = output_dir / f"{stem}_aggregate.json"
    aggregate_csv = output_dir / f"{stem}_aggregate.csv"
    conclusion_md = output_dir / f"{stem}_conclusion.md"
    write_json(sample_json, {"config": vars(args), "results": [asdict(x) for x in results]})
    write_json(aggregate_json, {"config": vars(args), "aggregate": aggregate})
    write_csv(aggregate_csv, aggregate)
    plots = write_plots(output_dir, aggregate)
    conclusion = write_conclusion(conclusion_md, aggregate)

    print("\nOutputs")
    print(f"sample_level_json={sample_json}")
    print(f"aggregate_json={aggregate_json}")
    print(f"aggregate_csv={aggregate_csv}")
    print(f"conclusion_md={conclusion_md}")
    for plot in plots:
        print(f"plot={plot}")
    print("\n" + conclusion)


if __name__ == "__main__":
    main()
