"""Strict formation, correction and 30-query persistence protocol."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .adapters import derived_record_count, snapshot_adapter
from .metrics import (
    correction_retrieved,
    echo_density,
    old_retrieval_wins,
    reconsolidation_resistance,
    semantic_occupancy,
    storage_rri,
    windowed_echo_persistence,
)


CONDITIONS = ("once_only", "read_only_repetition", "retrieve_writeback")


def _add(adapter, text, label, timestamp, kind, parent_id=None):
    try:
        return adapter.add(text, label, timestamp, kind, parent_id=parent_id)
    except TypeError:
        return adapter.add(text, label, timestamp, kind)


def _probe(adapter, query, sample, model, top_k, phase, index=None):
    hits = adapter.search(query, top_k=top_k)
    family_id = f"claim-family:{sample['sample_id']}"
    source_types = {
        "echo_seed": "historical_unverified",
        "retrieval_writeback": "retrieval_derived",
        "verified_correction": "verified_external",
        "encoding_noise": "independent_noise",
    }
    for hit in hits:
        label = hit.get("supplier_id")
        source_type = source_types.get(hit.get("kind"), "independent_noise")
        status = "active"
        if label == "ECHO" and "superseded" in hit.get("text", "").lower():
            status = "superseded"
        hit.update({
            "label": label,
            "family_id": family_id if label in {"ECHO", "CORRECTION"} else f"noise:{hit['memory_id']}",
            "expected_family_id": family_id if label in {"ECHO", "CORRECTION"} else f"noise:{hit['memory_id']}",
            "source_id": hit.get("parent_id") or hit.get("memory_id"),
            "source_type": source_type,
            "status": status,
            "confidence": 1.0 if source_type == "verified_external" else (
                0.5 if source_type == "historical_unverified" else 0.25
            ),
            "activation": min(1.0, 0.25 * float(hit.get("write_count") or 1)),
            "exposure_count": int(hit.get("write_count") or 1),
        })
    return {
        "phase": phase,
        "index": index,
        "query": query,
        "retrieved_memories": hits,
        "ED": echo_density(hits, sample["echo_seed"]["claim"], model),
        "correction_occupancy": semantic_occupancy(
            hits, sample["correction"]["claim"], "CORRECTION", model
        ),
        "correction_retrieved": correction_retrieved(hits),
        "old_retrieval_wins": old_retrieval_wins(hits),
    }


def run_protocol(sample, condition, adapter, model, top_k=5,
                 practice_queries=6, delay_queries=30):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    timestamp = datetime(2026, 1, 1, tzinfo=timezone.utc)
    question = sample["source"]["question"]
    events = []

    anchor_id = _add(
        adapter, sample["echo_seed"]["claim"], "ECHO",
        timestamp.isoformat(), "echo_seed",
    )
    events.append({"operation": "add", "kind": "echo_seed", "memory_id": anchor_id})
    for index in range(max(3, top_k - 2)):
        _add(
            adapter,
            f"Unrelated background record {index} for {sample['sample_id']}; "
            "it contains no answer to the target question.",
            "NOISE", (timestamp + timedelta(minutes=index + 1)).isoformat(),
            "encoding_noise",
        )

    encoding = _probe(
        adapter, f"Using stored memory, answer: {question}", sample, model,
        top_k, "encoding", 0,
    )
    practice = []
    for index in range(1, practice_queries + 1):
        if condition == "once_only":
            query = (
                f"Retrieve unrelated background record {index}; do not answer "
                f"the target question for {sample['sample_id']}."
            )
        else:
            query = f"Retrieve the historical memory and answer: {question}"
        turn = _probe(adapter, query, sample, model, top_k, "retrieval_practice", index)
        practice.append(turn)
        if condition == "retrieve_writeback":
            timestamp += timedelta(minutes=10)
            derived_id = _add(
                adapter,
                f"{sample['echo_seed']['claim']} Retrieval-derived trace {index} "
                "repeats the historical answer for future use.",
                "ECHO", timestamp.isoformat(), "retrieval_writeback",
                parent_id=anchor_id,
            )
            adapter.update(
                anchor_id,
                f"{sample['echo_seed']['claim']} Reconsolidation count {index}.",
                timestamp.isoformat(),
            )
            events.extend([
                {"operation": "add", "kind": "retrieval_writeback", "memory_id": derived_id,
                 "parent_id": anchor_id, "practice_index": index},
                {"operation": "update", "kind": "reconsolidated_echo", "memory_id": anchor_id,
                 "practice_index": index},
            ])

    # Identical target probe across conditions measures formation after the
    # different histories but before any correction is supplied.
    pre_correction = _probe(
        adapter, f"Before any correction, answer from memory: {question}",
        sample, model, top_k, "pre_correction",
    )
    before_records = snapshot_adapter(adapter)
    derived_count = derived_record_count(adapter)

    timestamp += timedelta(days=1)
    adapter.update(
        anchor_id,
        f"The historical answer for '{question}' is superseded and must not "
        "control the current answer.",
        timestamp.isoformat(),
    )
    correction_id = _add(
        adapter, sample["correction"]["claim"], "CORRECTION",
        timestamp.isoformat(), "verified_correction",
    )
    events.extend([
        {"operation": "update", "kind": "supersede_echo", "memory_id": anchor_id},
        {"operation": "add", "kind": "verified_correction", "memory_id": correction_id},
    ])
    after_records = snapshot_adapter(adapter)
    post_correction = _probe(
        adapter, f"Use the verified current memory to answer: {question}",
        sample, model, top_k, "post_correction", 0,
    )

    delay_series = []
    for index in range(1, delay_queries + 1):
        turn = _probe(
            adapter,
            f"After {index} later read-only sessions, answer from current memory: {question}",
            sample, model, top_k, "delayed_retrieval", index,
        )
        delay_series.append(turn)

    final = _probe(
        adapter,
        f"Make the final factually valid decision using memory: {question}",
        sample, model, top_k, "final_decision", delay_queries + 1,
    )
    retrieval_rri = reconsolidation_resistance(
        pre_correction["ED"], post_correction["ED"],
        post_correction["correction_occupancy"],
    )
    wep = windowed_echo_persistence([
        {"delay": row["index"], "ED": row["ED"]} for row in delay_series
    ])
    post_turns = [post_correction, *delay_series, final]
    correction_rate = sum(row["correction_retrieved"] for row in post_turns) / len(post_turns)

    metrics = {
        "pre_correction_ED": pre_correction["ED"],
        "post_correction_ED": post_correction["ED"],
        "derived_record_count": derived_count,
        "correction_retrieval_rate": correction_rate,
        "storage_RRI": storage_rri(
            before_records, after_records, sample["echo_seed"]["claim"],
            sample["correction"]["claim"], model,
        ),
        "retrieval_RRI": retrieval_rri,
        "WEP30": wep,
        "final_old_retrieval_wins": final["old_retrieval_wins"],
        "ED_at_1": delay_series[0]["ED"],
        "ED_at_10": delay_series[9]["ED"],
        "ED_at_20": delay_series[19]["ED"],
        "ED_at_30": delay_series[29]["ED"],
    }
    decision_input = {
        "question": question,
        "gold_answer": sample["ground_truth_value"],
        "echo_answer": sample["echo_value"],
        "snippets": [hit["text"] for hit in final["retrieved_memories"]],
    }
    return {
        "sample_id": sample["sample_id"],
        "dataset": sample["source"]["dataset"],
        "ability": sample["source"]["ability"],
        "framework": adapter.framework,
        "condition": condition,
        "metrics": metrics,
        "decision_input": decision_input,
        "trajectory": {
            "encoding": encoding,
            "retrieval_practice": practice,
            "pre_correction": pre_correction,
            "post_correction": post_correction,
            "delayed_retrieval": delay_series,
            "final_decision": final,
        },
        "memory_events": events,
        "storage_before_correction": before_records,
        "storage_after_correction": after_records,
    }
