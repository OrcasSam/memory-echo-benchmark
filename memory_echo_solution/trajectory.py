"""Strict five-stage trajectory with cumulative, auditable interventions."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import numpy as np

from memory_echo_lifecycle import CONDITIONS, STAGES
from memory_echo_lifecycle.metrics import (
    decision_contamination, echo_density, reconsolidation_resistance,
    semantic_occupancy, windowed_echo_persistence,
)
from . import COUNTERFACTUAL_DECISION_ARMS, RECONSOLIDATION_ARMS


def _mean(values):
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _rank_weights(hits):
    totals = defaultdict(float)
    for hit in hits:
        totals[hit.get("label") or hit.get("supplier_id")] += (
            1.0 / np.log2(hit["retrieval_rank"] + 1.0)
        )
    return totals


def _legacy_decision(hits):
    # Frequency contamination is the mechanism under test. A single historical
    # record cannot outvote a single verified correction merely by ranking one
    # slot higher; only duplicate numerical dominance creates an Echo decision.
    echo_count = sum((h.get("label") or h.get("supplier_id")) == "ECHO" for h in hits)
    correction_count = sum((h.get("label") or h.get("supplier_id")) == "CORRECTION" for h in hits)
    return "ECHO" if echo_count > correction_count else "CORRECTION"


def _evidence_decision(hits):
    best = {"ECHO": -1.0, "CORRECTION": -1.0}
    for hit in hits:
        label = hit.get("label") or hit.get("supplier_id")
        if label not in best or hit.get("status") in {"superseded", "suppressed"}:
            continue
        confidence = float(hit.get("confidence", 0.0))
        if hit.get("source_type") == "retrieval_derived":
            confidence = 0.0
        best[label] = max(best[label], confidence)
    if best["CORRECTION"] < 0 and best["ECHO"] < 0:
        return "UNCERTAIN"
    return "ECHO" if best["ECHO"] > best["CORRECTION"] else "CORRECTION"


def _answer_text(sample, predicted, mode="memory"):
    if predicted == "ECHO":
        return (f"The historical answer is: {sample['echo_value']}. "
                f"It is selected from the {mode} context.")
    if predicted == "CORRECTION":
        return (f"The verified current answer is: {sample['ground_truth_value']}. "
                "Independent correction evidence governs the response.")
    return "The available reliable evidence is insufficient; the answer is uncertain."


def _event(stage, result, label, timestamp, kind, source_type):
    return {
        "stage": stage, "operation": result["operation"],
        "executed": result["executed"], "memory_id": result.get("memory_id"),
        "reason": result.get("reason"), "label": label,
        "kind": kind, "source_type": source_type, "timestamp": timestamp,
        "family_id": result.get("family_id"), "source_id": result.get("source_id"),
    }


def build_solution_trajectory(sample: Dict, qa_plan: List[Dict], condition: str,
                              memory, model, top_k: int = 5,
                              dce_threshold: float = 0.5,
                              matched_control_curve: Dict[int, float] | None = None) -> Dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    arm = memory.arm
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    events, turns = [], []
    echo_source = str(sample["echo_seed"].get("source_id", sample["sample_id"]))
    result = memory.add(
        sample["echo_seed"]["claim"], "ECHO", now.isoformat(), "echo_seed",
        expected_family_id="historical_echo", source_id=echo_source,
        source_type="historical_unverified",
        independence_group=f"echo-source:{echo_source}", confidence=0.35,
    )
    anchor_id = result["memory_id"]
    echo_family_id = result.get("family_id")
    events.append(_event("encoding", result, "ECHO", now.isoformat(),
                         "echo_seed", "historical_unverified"))
    for index in range(max(2, top_k - 2)):
        text = (f"A non-controlling note for auxiliary-unit-{sample['sample_id']}-{index} "
                f"mentions auxiliary-state-{index}; it is unrelated to the target unit.")
        result = memory.add(
            text, "NOISE", (now + timedelta(minutes=index + 1)).isoformat(),
            "encoding_noise", expected_family_id=f"noise-encoding-{index}",
            source_id=f"noise-encoding-{index}", source_type="independent_noise",
            independence_group=f"noise-encoding-{index}", confidence=0.1,
        )
        events.append(_event("encoding", result, "NOISE", now.isoformat(),
                             "encoding_noise", "independent_noise"))

    correction_added = False
    practice_index = 0
    injected_delay_noise = 0
    counterfactual_flips = 0
    counterfactual_interventions = 0
    for qa in qa_plan:
        stage = qa["stage"]
        session = int(qa["turn"])
        if stage == "updating_reconsolidation" and not correction_added:
            now += timedelta(hours=1)
            if arm in RECONSOLIDATION_ARMS:
                result = memory.update(
                    anchor_id,
                    sample["echo_seed"]["claim"] + " [superseded; retained for audit]",
                    now.isoformat(), kind="verified_supersession",
                    source_type="verified_external", status="superseded",
                )
                events.append(_event(stage, result, "ECHO", now.isoformat(),
                                     "verified_supersession", "verified_external"))
            result = memory.add(
                sample["correction"]["claim"], "CORRECTION", now.isoformat(),
                "verified_correction", expected_family_id="verified_correction",
                source_id=str(sample["source"]["record_id"]),
                source_type="verified_external",
                independence_group=f"gold-source:{sample['source']['record_id']}",
                confidence=0.95, supersedes_family_id=echo_family_id,
            )
            events.append(_event(stage, result, "CORRECTION", now.isoformat(),
                                 "verified_correction", "verified_external"))
            correction_added = True

        if stage == "delayed_retrieval":
            target_noise = max(1, int(qa["delay"]))
            for index in range(injected_delay_noise, target_noise):
                text = (f"Intervening session note {index + 1} for {sample['subject']} records "
                        f"auxiliary-state-{index + 1}; it is not controlling evidence.")
                result = memory.add(
                    text, "NOISE", (now + timedelta(days=index + 1)).isoformat(),
                    "delay_noise", expected_family_id=f"noise-delay-{index + 1}",
                    source_id=f"noise-delay-{index + 1}", source_type="independent_noise",
                    independence_group=f"noise-delay-{index + 1}", confidence=0.1,
                )
                events.append(_event(stage, result, "NOISE", now.isoformat(),
                                     "delay_noise", "independent_noise"))
            injected_delay_noise = target_noise
            now = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=qa["delay"])

        hits = memory.search(
            qa["query"], top_k=top_k, stage=stage, session=session,
            reinforce_activation=(stage == "retrieval_practice" and condition != "control"),
        )
        ed = echo_density(hits, sample["echo_seed"]["claim"], model)
        correction_occupancy = semantic_occupancy(
            hits, sample["correction"]["claim"], "CORRECTION", model
        )
        persistence_action = None
        if stage == "delayed_retrieval":
            persistence_action = memory.observe_persistence(
                int(qa["delay"]), ed,
                None if matched_control_curve is None else matched_control_curve.get(int(qa["delay"])),
            )

        predicted = _legacy_decision(hits)
        answer = _answer_text(sample, predicted)
        counterfactual = None
        pre_guard_dce = None
        if stage == "final_decision" and arm in COUNTERFACTUAL_DECISION_ARMS:
            complete_hits = memory.search_complete(qa["query"], top_k)
            reliable_hits = memory.search_reliable(qa["query"], top_k)
            complete_label = _legacy_decision(complete_hits)
            reliable_label = _evidence_decision(reliable_hits)
            complete_answer = _answer_text(sample, complete_label, "complete-memory")
            reliable_answer = _answer_text(sample, reliable_label, "echo-excluded reliable-evidence")
            flip = complete_label != reliable_label and complete_label == "ECHO"
            if flip:
                counterfactual_flips += 1
                counterfactual_interventions += 1
                predicted, answer = reliable_label, reliable_answer
            else:
                predicted, answer = complete_label, complete_answer
            pre_guard_dce = decision_contamination(
                complete_answer, qa["echo_answer"], qa["canonical_answer"], model,
                dce_threshold, complete_label,
            )
            counterfactual = {
                "complete_answer": complete_answer,
                "complete_label": complete_label,
                "echo_excluded_answer": reliable_answer,
                "echo_excluded_label": reliable_label,
                "decision_flip": flip,
                "intervention_applied": flip,
                "uncertainty_emitted": reliable_label == "UNCERTAIN",
            }

        correct_similarity = float(np.dot(
            model.encode([answer], normalize_embeddings=True, show_progress_bar=False)[0],
            model.encode([qa["canonical_answer"]], normalize_embeddings=True,
                         show_progress_bar=False)[0],
        ))
        dce = None
        if stage == "final_decision":
            dce = decision_contamination(
                answer, qa["echo_answer"], qa["canonical_answer"], model,
                dce_threshold, predicted,
            )
        turns.append({
            **qa, "condition": condition, "solution_arm": arm,
            "retrieved_memories": hits, "retrieval_count": len(hits),
            "ED": ed, "correction_occupancy": correction_occupancy,
            "answer": answer, "predicted_label": predicted,
            "target_correct": predicted == qa["target_label"],
            "answer_similarity": max(0.0, min(1.0, correct_similarity)),
            "DCE": None if dce is None else dce["DCE"],
            "contaminated": None if dce is None else dce["contaminated"],
            "decision_evaluation": dce,
            "pre_guard_decision_evaluation": pre_guard_dce,
            "counterfactual_decision": counterfactual,
            "persistence_action": persistence_action,
        })

        if stage == "retrieval_practice":
            practice_index += 1
            if condition in {"repetition", "update"}:
                text = (f"{sample['echo_seed']['claim']} Retrieval-practice trace "
                        f"{practice_index} repeats the same historical source.")
                result = memory.add(
                    text, "ECHO", now.isoformat(), "retrieval_writeback",
                    expected_family_id="historical_echo",
                    source_id=f"model-retrieval-{practice_index}",
                    source_type="retrieval_derived",
                    independence_group=f"generated-{practice_index}",
                    confidence=0.35, parent_memory_id=anchor_id,
                )
                events.append(_event(stage, result, "ECHO", now.isoformat(),
                                     "retrieval_writeback", "retrieval_derived"))
            if condition == "update":
                reinforced = (f"{sample['echo_seed']['claim']} Reconsolidated retrieval "
                              f"{practice_index} preserves the same historical source.")
                result = memory.update(
                    anchor_id, reinforced, now.isoformat(),
                    kind="retrieval_reconsolidation", source_type="retrieval_derived",
                )
                events.append(_event(stage, result, "ECHO", now.isoformat(),
                                     "retrieval_reconsolidation", "retrieval_derived"))

    stage_ed = {stage: _mean(t["ED"] for t in turns if t["stage"] == stage)
                for stage in STAGES}
    stage_accuracy = {
        stage: _mean(float(t["target_correct"]) for t in turns if t["stage"] == stage)
        for stage in STAGES
    }
    pre_ed = stage_ed["retrieval_practice"]
    post_ed = stage_ed["updating_reconsolidation"]
    post_correction = _mean(t["correction_occupancy"] for t in turns
                            if t["stage"] == "updating_reconsolidation")
    delay_series = [{"delay": t["delay"], "ED": t["ED"],
                     "target_correct": t["target_correct"],
                     "persistence_action": t["persistence_action"]}
                    for t in turns if t["stage"] == "delayed_retrieval"]
    final_turns = [t for t in turns if t["stage"] == "final_decision"]
    metrics = {
        "ED": stage_ed,
        "RRI": reconsolidation_resistance(pre_ed, post_ed, post_correction),
        "RRI_components": {"pre_update_ED": pre_ed, "post_update_ED": post_ed,
                           "post_update_correction_occupancy": post_correction},
        "WEP": windowed_echo_persistence(delay_series),
        "DCE": _mean(t["DCE"] for t in final_turns),
        "pre_guard_DCE": _mean(
            t["pre_guard_decision_evaluation"]["DCE"]
            for t in final_turns if t["pre_guard_decision_evaluation"] is not None
        ),
        "decision_contamination_rate": _mean(float(t["contaminated"])
                                              for t in final_turns),
        "counterfactual_flip_rate": counterfactual_flips / len(final_turns) if final_turns else 0.0,
        "counterfactual_intervention_rate": counterfactual_interventions / len(final_turns) if final_turns else 0.0,
        "mean_answer_similarity": _mean(t["answer_similarity"] for t in turns),
        "stage_accuracy": stage_accuracy,
        "final_accuracy": _mean(float(t["target_correct"]) for t in final_turns),
    }
    return {
        "condition": condition, "framework": memory.framework,
        "solution_arm": arm, "trajectory": turns, "memory_events": events,
        "memory_audit": memory.audit(), "metrics": metrics,
        "delay_series": delay_series,
    }
