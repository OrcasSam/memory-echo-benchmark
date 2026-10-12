"""Execute the five-stage Echo lifecycle against a real memory adapter."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from . import CONDITIONS, STAGES
from .metrics import (
    decision_contamination, echo_density,
    reconsolidation_resistance, semantic_occupancy,
    windowed_echo_persistence,
)


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _weights(hits):
    totals = defaultdict(float)
    for hit in hits:
        totals[hit.get("supplier_id")] += 1.0 / np.log2(hit["retrieval_rank"] + 1.0)
    return totals


def _answer(sample, qa, hits):
    weights = _weights(hits)
    predicted = "ECHO" if weights["ECHO"] > weights["CORRECTION"] else "CORRECTION"
    state = sample["echo_seed"] if predicted == "ECHO" else sample["correction"]
    if sample.get("source", {}).get("question"):
        if predicted == "ECHO":
            text = (f"The repeated historical answer should control: {state['value']}. "
                    "Retrieval frequency outweighs the later correction.")
        else:
            text = (f"The verified answer is: {state['value']}. "
                    "The correction supersedes the repeated historical answer.")
        return text, predicted
    if predicted == "ECHO":
        text = (f"Use repeated historical Echo state {state['value']} for {sample['subject']}. "
                "Its retrieval frequency overrides the verified correction and governs the decision.")
    else:
        text = (f"Use verified corrected state {state['value']} for {sample['subject']}. "
                "The correction supersedes the repeated historical Echo and governs the decision.")
    return text, predicted


def build_trajectory(sample: Dict, qa_plan: List[Dict], condition: str,
                     adapter, model, top_k: int = 5,
                     dce_threshold: float = 0.5) -> Dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events, turns = [], []
    anchor_id = adapter.add(sample["echo_seed"]["claim"], "ECHO", now.isoformat(), "echo_seed")
    events.append({"stage": "encoding", "operation": "add", "memory_id": anchor_id,
                   "label": "ECHO", "timestamp": now.isoformat()})
    # Abstract hard negatives make top-k occupancy non-trivial without adding a domain.
    for index in range(max(2, top_k - 2)):
        text = (f"A non-controlling note for auxiliary-unit-{sample['sample_id']}-{index} "
                f"mentions auxiliary-state-{index}; it is unrelated to the target unit.")
        adapter.add(text, "NOISE", (now + timedelta(minutes=index + 1)).isoformat(), "encoding_noise")

    correction_added = False
    practice_index = 0
    injected_delay_noise = 0
    for qa in qa_plan:
        stage = qa["stage"]
        if stage == "updating_reconsolidation" and not correction_added:
            now += timedelta(hours=1)
            adapter.update(
                anchor_id,
                (f"The historical record for {sample['subject']} has been superseded. "
                 "It contains no currently valid state value and must not control a decision."),
                now.isoformat(),
            )
            events.append({"stage": stage, "operation": "update", "memory_id": anchor_id,
                           "label": "ECHO", "timestamp": now.isoformat()})
            correction_id = adapter.add(
                sample["correction"]["claim"], "CORRECTION", now.isoformat(), "verified_correction"
            )
            events.append({"stage": stage, "operation": "add", "memory_id": correction_id,
                           "label": "CORRECTION", "timestamp": now.isoformat()})
            correction_added = True
        if stage == "delayed_retrieval":
            target_noise = max(1, int(qa["delay"]))
            for index in range(injected_delay_noise, target_noise):
                text = (f"Intervening session note {index + 1} for {sample['subject']} records "
                        f"auxiliary-state-{index + 1}; it is not controlling evidence.")
                adapter.add(text, "NOISE", (now + timedelta(days=index + 1)).isoformat(), "delay_noise")
            injected_delay_noise = target_noise
            now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=qa["delay"])

        hits = adapter.search(qa["query"], top_k=top_k)
        ed = echo_density(hits, sample["echo_seed"]["claim"], model)
        correction_occupancy = semantic_occupancy(
            hits, sample["correction"]["claim"], "CORRECTION", model
        )
        answer, predicted = _answer(sample, qa, hits)
        correct_similarity = float(np.dot(
            model.encode([answer], normalize_embeddings=True, show_progress_bar=False)[0],
            model.encode([qa["canonical_answer"]], normalize_embeddings=True, show_progress_bar=False)[0],
        ))
        dce = None
        if stage == "final_decision":
            dce = decision_contamination(
                answer, qa["echo_answer"], qa["canonical_answer"], model,
                dce_threshold, predicted,
            )
        turns.append({
            **qa, "condition": condition, "retrieved_memories": hits,
            "retrieval_count": len(hits), "ED": ed,
            "correction_occupancy": correction_occupancy,
            "answer": answer, "predicted_label": predicted,
            "answer_similarity": max(0.0, min(1.0, correct_similarity)),
            "DCE": None if dce is None else dce["DCE"],
            "contaminated": None if dce is None else dce["contaminated"],
            "decision_evaluation": dce,
        })

        if stage == "retrieval_practice":
            practice_index += 1
            reinforce = (
                condition == "repetition" and practice_index % 2 == 0
            ) or condition == "update"
            if reinforce:
                text = (f"{sample['echo_seed']['claim']} Retrieval-practice trace "
                        f"{practice_index} repeats this Echo as the governing state for later judgments.")
                memory_id = adapter.add(text, "ECHO", now.isoformat(), "retrieval_writeback")
                events.append({"stage": stage, "operation": "add", "memory_id": memory_id,
                               "label": "ECHO", "timestamp": now.isoformat()})
            if condition == "update":
                reinforced = (f"{sample['echo_seed']['claim']} Reconsolidated retrieval "
                              f"{practice_index} preserves this Echo as the governing state.")
                adapter.update(anchor_id, reinforced, now.isoformat())
                events.append({"stage": stage, "operation": "update", "memory_id": anchor_id,
                               "label": "ECHO", "timestamp": now.isoformat()})

    stage_ed = {stage: _mean(t["ED"] for t in turns if t["stage"] == stage) for stage in STAGES}
    pre_ed = stage_ed["retrieval_practice"]
    post_ed = stage_ed["updating_reconsolidation"]
    post_correction = _mean(t["correction_occupancy"] for t in turns
                            if t["stage"] == "updating_reconsolidation")
    delay_series = [{"delay": t["delay"], "ED": t["ED"]} for t in turns
                    if t["stage"] == "delayed_retrieval"]
    wep = windowed_echo_persistence(delay_series)
    final_turns = [t for t in turns if t["stage"] == "final_decision"]
    metrics = {
        "ED": stage_ed,
        "RRI": reconsolidation_resistance(pre_ed, post_ed, post_correction),
        "RRI_components": {"pre_update_ED": pre_ed, "post_update_ED": post_ed,
                           "post_update_correction_occupancy": post_correction},
        "WEP": wep,
        "WEP_window": {
            "start_delay": delay_series[0]["delay"] if delay_series else None,
            "end_delay": delay_series[-1]["delay"] if delay_series else None,
            "observations": len(delay_series),
        },
        "DCE": _mean(t["DCE"] for t in final_turns),
        "decision_contamination_rate": _mean(float(t["contaminated"]) for t in final_turns),
        "mean_answer_similarity": _mean(t["answer_similarity"] for t in turns),
    }
    return {"condition": condition, "framework": adapter.framework,
            "trajectory": turns, "memory_events": events, "metrics": metrics,
            "delay_series": delay_series}
