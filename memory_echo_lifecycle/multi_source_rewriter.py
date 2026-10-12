"""Balanced rewriting from LoCoMo, LongMemEval, HotpotQA, and BEAM."""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from . import MEMORY_UNIT_TYPES


def _short(value, limit=220):
    text = " ".join(str(value).split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _id(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()[:12]


def _usable(question, answer):
    return bool(question and answer and 1 < len(str(answer).strip()) <= 180)


def load_locomo(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8")); rows = []
    category_names = {1: "single-hop", 2: "temporal-reasoning", 3: "multi-hop", 4: "open-domain", 5: "adversarial"}
    for sample in payload:
        for index, qa in enumerate(sample.get("qa", [])):
            if not _usable(qa.get("question"), qa.get("answer")):
                continue
            rows.append({
                "dataset": "LoCoMo", "record_id": f"{sample.get('sample_id', 'sample')}:qa:{index}",
                "ability": category_names.get(qa.get("category"), f"category-{qa.get('category')}") ,
                "question": qa["question"], "answer": str(qa["answer"]),
                "evidence": "Evidence turns: " + ", ".join(map(str, qa.get("evidence", []))),
            })
    return rows


def load_longmemeval(path: Path) -> List[Dict]:
    payload = json.loads(path.read_text(encoding="utf-8")); rows = []
    preferred = {"knowledge-update", "temporal-reasoning", "multi-session", "information-extraction"}
    ordered = sorted(payload, key=lambda x: x.get("question_type") not in preferred)
    for row in ordered:
        if not _usable(row.get("question"), row.get("answer")):
            continue
        evidence = []
        for session in row.get("haystack_sessions", []):
            for message in session:
                if message.get("has_answer"):
                    evidence.append(message.get("content", ""))
        rows.append({
            "dataset": "LongMemEval", "record_id": row["question_id"],
            "ability": row.get("question_type", "long-term-memory"),
            "question": row["question"], "answer": str(row["answer"]),
            "evidence": _short(evidence[0] if evidence else f"Answer sessions: {row.get('answer_session_ids', [])}"),
        })
    return rows


def load_hotpotqa(path: Path) -> List[Dict]:
    frame = pd.read_parquet(path); rows = []
    for _, row in frame.iterrows():
        if not _usable(row["question"], row["answer"]):
            continue
        context = row["context"]
        titles = list(context.get("title", [])) if isinstance(context, dict) else []
        sentences = list(context.get("sentences", [])) if isinstance(context, dict) else []
        supporting = row["supporting_facts"]
        support_titles = set(supporting.get("title", [])) if isinstance(supporting, dict) else set()
        excerpts = []
        for title, group in zip(titles, sentences):
            if title in support_titles:
                excerpts.extend(list(group)[:2])
        rows.append({
            "dataset": "HotpotQA", "record_id": str(row["id"]),
            "ability": f"multi-hop-{row['type']}", "question": row["question"],
            "answer": str(row["answer"]), "evidence": _short(" ".join(excerpts)),
        })
    return rows


def load_beam(root: Path) -> List[Dict]:
    rows = []
    preferred_order = (
        "knowledge_update", "contradiction_resolution", "event_ordering",
        "temporal_reasoning", "information_extraction", "multi_session_reasoning",
        "instruction_following", "preference_following", "summarization", "abstention",
    )
    paths = sorted((root / "chats" / "100K").glob("*/probing_questions/probing_questions.json"))
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        keys = sorted(
            payload,
            key=lambda k: preferred_order.index(k.replace("-", "_"))
            if k.replace("-", "_") in preferred_order else 99,
        )
        for ability in keys:
            for index, qa in enumerate(payload[ability]):
                answer = qa.get("ideal_response") or qa.get("answer")
                if not _usable(qa.get("question"), answer):
                    continue
                rubric = qa.get("rubric") or []
                rows.append({
                    "dataset": "BEAM", "record_id": f"100K:{path.parents[1].name}:{ability}:{index}",
                    "ability": ability.replace("_", "-"), "question": qa["question"],
                    "answer": str(answer),
                    "evidence": _short(rubric[0] if rubric else qa.get("plan_reference", "BEAM rubric")),
                })
    return rows


def _balanced_counts(total: int, sources: List[str]) -> Dict[str, int]:
    base, remainder = divmod(total, len(sources))
    return {source: base + (index < remainder) for index, source in enumerate(sources)}


def rewrite_multi_source_dataset(paths: Dict[str, Path], max_samples: int = 12,
                                 delay_points=tuple(range(1, 31)),
                                 practice_qa: int = 6) -> List[Dict]:
    loaders = {
        "LoCoMo": load_locomo(paths["LoCoMo"]),
        "LongMemEval": load_longmemeval(paths["LongMemEval"]),
        "HotpotQA": load_hotpotqa(paths["HotpotQA"]),
        "BEAM": load_beam(paths["BEAM"]),
    }
    counts = _balanced_counts(max_samples, list(loaders))
    selected = []
    for source, records in loaders.items():
        # Prefer distinct ability types before taking a second record of a type.
        by_ability = defaultdict(list)
        for record in records: by_ability[record["ability"]].append(record)
        ability_priority = {
            "LoCoMo": ("single-hop", "temporal-reasoning", "multi-hop", "open-domain", "adversarial"),
            "LongMemEval": ("knowledge-update", "temporal-reasoning", "multi-session", "information-extraction"),
            "HotpotQA": ("multi-hop-bridge", "multi-hop-comparison"),
            "BEAM": ("knowledge-update", "contradiction-resolution", "event-ordering", "temporal-reasoning", "information-extraction"),
        }[source]
        ordered = []
        while any(by_ability.values()):
            abilities = sorted(
                by_ability,
                key=lambda value: ability_priority.index(value)
                if value in ability_priority else len(ability_priority),
            )
            for ability in abilities:
                if by_ability[ability]: ordered.append(by_ability[ability].pop(0))
        chosen = ordered[:counts[source]]
        if len(chosen) < counts[source]:
            raise ValueError(f"{source} has only {len(chosen)} usable records")
        selected.extend(chosen)

    source_groups = defaultdict(list)
    for record in selected: source_groups[record["dataset"]].append(record)
    output = []
    for index, record in enumerate(selected):
        pool = [candidate for candidate in loaders[record["dataset"]]
                if candidate["record_id"] != record["record_id"]
                and candidate["answer"].strip().lower() != record["answer"].strip().lower()
                and candidate["ability"] == record["ability"]]
        if not pool:
            pool = [candidate for candidate in loaders[record["dataset"]]
                    if candidate["record_id"] != record["record_id"]
                    and candidate["answer"].strip().lower() != record["answer"].strip().lower()]
        distractor = pool[index % len(pool)]
        unit_type = {
            "LoCoMo": "preference" if "open" in record["ability"] else "fact",
            "LongMemEval": "constraint" if "update" in record["ability"] else "plan",
            "HotpotQA": "conclusion",
            "BEAM": "hypothesis" if "contradiction" in record["ability"] else "fact",
        }[record["dataset"]]
        subject = f"{record['dataset'].lower()}-{_id(record['record_id'])}"
        echo_value = _short(distractor["answer"], 160)
        correct_value = _short(record["answer"], 160)
        output.append({
            "sample_id": f"multi-{index:05d}",
            "source": {
                "dataset": record["dataset"], "record_id": record["record_id"],
                "ability": record["ability"], "question": record["question"],
                "original_answer": record["answer"], "evidence": record["evidence"],
                "echo_answer_source_id": distractor["record_id"],
                "transformation": "same-source/same-ability alternate answer -> Echo; gold answer -> Correction",
            },
            "memory_unit_type": unit_type, "subject": subject,
            "echo_seed": {
                "label": "ECHO", "value": echo_value,
                "claim": f"Historical memory answers the question '{record['question']}' as: {echo_value}",
            },
            "correction": {
                "label": "CORRECTION", "value": correct_value,
                "claim": (f"Verified evidence corrects the answer to '{record['question']}' as: "
                          f"{correct_value}. Supporting evidence: {record['evidence']}"),
            },
            "delay_points": list(map(int, delay_points)), "practice_qa": int(practice_qa),
            "ground_truth_value": correct_value, "echo_value": echo_value,
        })
    return output


def source_distribution(dataset: Iterable[Dict]) -> Dict:
    return {
        "dataset_counts": dict(Counter(row["source"]["dataset"] for row in dataset)),
        "ability_counts": dict(Counter(row["source"]["ability"] for row in dataset)),
    }
