"""Build controlled generic Memory Echo trajectories over real memory APIs."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from math import log2
from typing import Dict, List, Optional

from .multi_type_qa_generator import generate_qa_plan


CONDITIONS = (
    "control",
    "historical_repetition",
    "historical_repetition_update",
)


def _rank_weight(rank: int) -> float:
    return 1.0 / log2(float(rank) + 1.0)


def _memory_label(hit: Dict):
    return hit.get("memory_label") or hit.get("supplier_id")


def _normalise_hits(raw_hits: List[Dict], sample: Dict) -> List[Dict]:
    """Remove the legacy business field used by the shared adapter."""
    hits = []
    lineage = sample["initial_memory"]["lineage_id"]
    for raw in raw_hits:
        hit = dict(raw)
        label = hit.pop("supplier_id", None) or hit.get("memory_label")
        hit["memory_label"] = label
        if label in ("OLD", "NEW"):
            hit["lineage_id"] = lineage
            hit["version"] = label.lower()
        elif label:
            hit["lineage_id"] = hit.get("memory_id")
            hit["version"] = "non_target"
        hits.append(hit)
    return hits


def _label_weights(hits: List[Dict]) -> Counter:
    """Use only rank weights so A-MEM and Mem0 score scales are not mixed."""
    weights = Counter()
    for hit in hits:
        label = _memory_label(hit)
        if label:
            weights[label] += _rank_weight(
                int(hit.get("retrieval_rank", 1))
            )
    return weights


def _echo_density(hits: List[Dict]) -> Optional[float]:
    if not hits:
        return None
    return (
        sum(_memory_label(hit) == "OLD" for hit in hits)
        / len(hits)
    )


def _update_incorporation(hits: List[Dict]) -> Optional[float]:
    old_weight = sum(
        _rank_weight(int(hit.get("retrieval_rank", 1)))
        for hit in hits if _memory_label(hit) == "OLD"
    )
    new_weight = sum(
        _rank_weight(int(hit.get("retrieval_rank", 1)))
        for hit in hits if _memory_label(hit) == "NEW"
    )
    denominator = old_weight + new_weight
    return new_weight / denominator if denominator else None


def _relevant_coverage(hits: List[Dict]) -> Optional[float]:
    if not hits:
        return None
    return (
        sum(
            _memory_label(hit) in ("OLD", "NEW")
            for hit in hits
        ) / len(hits)
    )


def _predicted_label(qa: Dict, hits: List[Dict]):
    weights = _label_weights(hits)
    if qa["qa_type"] == "recall" and weights["OLD"] > 0:
        return "OLD"
    # The validity cue is visible in retrieved evidence. Apply it as a gate
    # before rank aggregation so catch trials are actually solvable rather
    # than silently encouraging an "always choose NEW" policy.
    new_text = " ".join(
        str(hit.get("text", "")).lower()
        for hit in hits if _memory_label(hit) == "NEW"
    )
    invalid_replacement = (
        "does not invalidate the stored state" in new_text
        or "out of scope for replacement" in new_text
    )
    if invalid_replacement:
        return "OLD" if weights["OLD"] > 0 else None
    old_weight, new_weight = weights["OLD"], weights["NEW"]
    if old_weight == new_weight:
        return None
    return "OLD" if old_weight > new_weight else "NEW"


def _value(sample: Dict, label: str):
    return (
        sample["initial_memory"]["value"]
        if label == "OLD" else sample["conflicting_evidence"]["value"]
    )


def agent_answer_for_label(sample: Dict, qa: Dict, predicted_label):
    """Deterministic retrieval-to-answer policy used equally in all conditions."""
    subject = sample["subject"]
    if predicted_label is None:
        return "The retrieved memory does not support a definite state.", None
    value = _value(sample, predicted_label)
    old_value = sample["initial_memory"]["value"]
    new_value = sample["conflicting_evidence"]["value"]
    if qa["qa_type"] == "recall":
        text = f"The original historical value for {subject} was {value}."
    elif qa["qa_type"] == "comparison":
        text = (
            f"Between {old_value} and {new_value}, the memory evidence favors "
            f"{value} as the value to use for {subject}."
        )
    elif qa["qa_type"] == "update_awareness":
        action = (
            "retain the historical state"
            if predicted_label == "OLD" else "incorporate the revision"
        )
        text = f"The memory should {action}; use {value} for {subject}."
    elif qa["qa_type"] == "consistency":
        text = (
            f"A consistent judgment uses {value} as the currently controlling "
            f"value for {subject}."
        )
    elif qa["qa_type"] == "decision":
        text = f"The downstream action should use {value} for {subject}."
    else:
        text = (
            f"Use {value} for {subject} because it has the strongest retrieved "
            "memory support."
        )
    return text, value


def _mean(values):
    values = [float(value) for value in values if value is not None]
    return sum(values) / len(values) if values else None


def _paired_persistence_values(trajectory: List[Dict]):
    before = {
        turn["persistence_probe_id"]: turn["echo_density"]
        for turn in trajectory
        if turn["phase"] == "update_adaptation"
        and turn.get("persistence_probe_id")
    }
    after = {
        turn["persistence_probe_id"]: turn["echo_density"]
        for turn in trajectory
        if turn["phase"] == "echo_retention"
        and turn.get("persistence_probe_id")
    }
    common = sorted(set(before) & set(after))
    return (
        [before[key] for key in common],
        [after[key] for key in common],
        common,
    )


def _run_metrics(trajectory: List[Dict]) -> Dict:
    seeding = [
        turn for turn in trajectory if turn["phase"] == "echo_seeding"
    ]
    updating = [
        turn for turn in trajectory if turn["phase"] == "update_adaptation"
    ]
    decisions = [
        turn for turn in trajectory
        if turn["phase"] == "decision_influence"
        and turn["qa_type"] == "decision"
        and turn["decision_bias_eligible"]
    ]
    before, after, probe_ids = _paired_persistence_values(trajectory)
    pre_stop_ed = _mean(before)
    delayed_ed = _mean(after)
    eps = (
        delayed_ed / pre_stop_ed
        if pre_stop_ed is not None and pre_stop_ed > 0
        and delayed_ed is not None else None
    )
    all_eds = [turn["echo_density"] for turn in trajectory]
    return {
        # The run-level ED is the standardized pre-stop occupancy measure.
        # Per-phase ED remains available in every turn and aggregate table.
        "ED": pre_stop_ed,
        "seeding_ED": _mean(
            turn["echo_density"] for turn in seeding
        ),
        "overall_ED": _mean(all_eds),
        "UIR": _mean(
            turn["update_incorporation_rate"] for turn in updating
        ),
        "relevant_retrieval_coverage": _mean(
            turn["relevant_retrieval_coverage"] for turn in updating
        ),
        "pre_stop_ED": pre_stop_ed,
        "delayed_ED": delayed_ed,
        "EPS": eps,
        "persistence_probe_ids": probe_ids,
        "DBI": (
            _mean(turn["decision_bias_flag"] for turn in decisions)
            if decisions else None
        ),
        "decision_probe_count": len(decisions),
        "semantic_answer_score": _mean(
            turn["answer_evaluation"]["answer_score"]
            for turn in trajectory
        ),
        "empty_retrieval_rate": _mean(
            float(not turn["retrieved_memories"])
            for turn in trajectory
        ),
    }


def build_generic_trajectory(
    sample: Dict,
    condition: str,
    adapter,
    evaluator,
    top_k: int = 5,
    qa_count: int = 20,
) -> Dict:
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    plan = generate_qa_plan(sample, qa_count)
    start = datetime(2027, 1, 1, tzinfo=timezone.utc) + timedelta(
        days=sample["sample_id"] * 5
    )
    writes = Counter()
    adds = Counter()
    updates = Counter()
    retrievals = Counter()
    events = []

    def add_memory(text, label, timestamp, kind, lineage_id=None, version=None):
        memory_id = adapter.add(text, label, timestamp, kind)
        writes[label] += 1
        adds[label] += 1
        events.append({
            "event": "add",
            "memory_id": memory_id,
            "memory_label": label,
            "lineage_id": lineage_id or memory_id,
            "version": version or "non_target",
            "timestamp": timestamp,
            "kind": kind,
        })
        return memory_id

    old = sample["initial_memory"]
    lineage = old["lineage_id"]
    old_id = add_memory(
        old["claim"], "OLD", start.isoformat(), "initial_memory",
        lineage, "old",
    )
    distractors = sample["distractor_memories"]
    split = max(1, len(distractors) // 2)
    initial_noise = distractors[:split]
    delayed_noise = distractors[split:]
    for noise in initial_noise:
        add_memory(
            noise["claim"], "NOISE", start.isoformat(),
            "initial_interference",
        )

    evidence_added = False
    delayed_noise_added = False
    target_writebacks = 0
    exposure_retrievals = 0
    matched_capacity_writes = 0
    phase_turns = Counter()
    trajectory = []
    for qa in plan:
        turn = qa["turn"]
        phase = qa["phase"]
        phase_index = phase_turns[phase]
        phase_turns[phase] += 1
        ts_base = start + timedelta(days=turn - 1)
        if phase in ("echo_retention", "decision_influence"):
            ts_base += timedelta(days=sample["delay_sessions"])
        ts = ts_base.isoformat()

        if phase == "update_adaptation" and not evidence_added:
            evidence = sample["conflicting_evidence"]
            for j in range(evidence["corroboration_count"]):
                prefix = (
                    "" if j == 0
                    else f"Independent corroboration {j + 1}: "
                )
                add_memory(
                    prefix + evidence["claim"], "NEW", ts,
                    f"conflicting_evidence_{evidence['strength']}",
                    lineage, "new",
                )
            evidence_added = True

        if phase == "echo_retention" and not delayed_noise_added:
            for noise in delayed_noise:
                add_memory(
                    noise["claim"], "NOISE", ts,
                    "delayed_session_interference",
                )
            delayed_noise_added = True
            events.append({
                "event": "session_boundary",
                "timestamp": ts,
                "session_gap": sample["delay_sessions"],
                "interpretation": (
                    "retrieval after a session/noise gap; neither framework "
                    "implements wall-clock forgetting"
                ),
            })

        # Extra, unscored retrieval-practice events are the only difference
        # between control and repetition. Evaluation probes remain identical.
        exposure_hits = []
        if (
            condition != "control"
            and phase in ("echo_seeding", "update_adaptation")
        ):
            exposure_query = (
                f"Retrieve the established historical state for "
                f"{sample['subject']}."
            )
            exposure_hits = _normalise_hits(
                adapter.search(exposure_query, top_k), sample
            )
            exposure_retrievals += 1
            for hit in exposure_hits:
                retrievals[f"exposure:{hit.get('memory_label')}"] += 1
            events.append({
                "event": "unscored_exposure_retrieval",
                "turn": turn,
                "phase": phase,
                "query": exposure_query,
                "retrieved_memories": exposure_hits,
            })

        # Retrieval-update feedback begins only after conflicting evidence is
        # present. Other conditions receive one unrelated capacity-matching
        # add at the same turns, so ED lift is not caused by index size alone.
        reinforcement_slot = (
            phase == "update_adaptation"
            and phase_index < sample["writeback_budget"]
        )
        if reinforcement_slot:
            if (
                condition == "historical_repetition_update"
                and any(
                    hit.get("memory_label") == "OLD"
                    for hit in exposure_hits
                )
            ):
                writeback_id = add_memory(
                    f"Retrieval reinforcement episode {phase_index + 1}: "
                    f"{old['claim']}",
                    "OLD", ts, "retrieval_writeback", lineage, "old",
                )
                target_writebacks += 1
                updated_text = (
                    f"Historically reinforced {target_writebacks} times: "
                    f"{old['claim']}"
                )
                adapter.update(old_id, updated_text, ts)
                writes["OLD"] += 1
                updates["OLD"] += 1
                events.append({
                    "event": "update",
                    "memory_id": old_id,
                    "source_memory_id": writeback_id,
                    "memory_label": "OLD",
                    "lineage_id": lineage,
                    "version": "old",
                    "timestamp": ts,
                    "kind": "retrieval_update_feedback",
                })
            else:
                add_memory(
                    (
                        f"Unrelated capacity-matching placeholder "
                        f"{sample['sample_id']}-{phase_index + 1}."
                    ),
                    "CONTROL", ts, "capacity_match",
                )
                matched_capacity_writes += 1

        hits = _normalise_hits(adapter.search(qa["query"], top_k), sample)
        for hit in hits:
            retrievals[f"probe:{hit.get('memory_label')}"] += 1
        predicted_label = _predicted_label(qa, hits)
        answer, predicted_value = agent_answer_for_label(
            sample, qa, predicted_label
        )
        evaluation = evaluator.evaluate(
            answer,
            qa["canonical_answer"],
            predicted_value,
            qa["canonical_target_value"],
        )
        decision_bias = float(
            phase == "decision_influence"
            and qa["qa_type"] == "decision"
            and sample["should_update"]
            and predicted_label == "OLD"
        )
        decision_bias_eligible = bool(
            phase == "decision_influence"
            and qa["qa_type"] == "decision"
            and sample["should_update"]
        )
        trajectory.append({
            **qa,
            "timestamp": ts,
            "retrieved_memories": hits,
            "echo_density": _echo_density(hits),
            "update_incorporation_rate": (
                _update_incorporation(hits) if evidence_added else None
            ),
            "relevant_retrieval_coverage": _relevant_coverage(hits),
            "predicted_memory_label": predicted_label,
            "predicted_value": predicted_value,
            "agent_answer": answer,
            "answer_evaluation": evaluation,
            "decision_bias_flag": decision_bias,
            "decision_bias_eligible": decision_bias_eligible,
        })

    metrics = _run_metrics(trajectory)
    decision_turns = [
        turn for turn in trajectory
        if turn["phase"] == "decision_influence"
        and turn["qa_type"] == "decision"
    ]
    return {
        "protocol": sample["protocol"],
        "condition": condition,
        "qa_count": qa_count,
        "memory_trajectory": trajectory,
        "memory_events": events,
        "write_operations_by_label": dict(writes),
        "add_operations_by_label": dict(adds),
        "update_operations_by_label": dict(updates),
        "retrieval_frequency": dict(retrievals),
        "exposure_retrieval_count": exposure_retrievals,
        "target_writeback_count": target_writebacks,
        "capacity_matching_write_count": matched_capacity_writes,
        "distinct_memory_count": sum(adds.values()),
        "final_decisions": [
            {
                "turn": turn["turn"],
                "predicted_memory_label": turn["predicted_memory_label"],
                "predicted_value": turn["predicted_value"],
                "canonical_target_label": turn["canonical_target_label"],
                "canonical_target_value": turn["canonical_target_value"],
                "biased_to_old": bool(turn["decision_bias_flag"]),
            }
            for turn in decision_turns
        ],
        "metrics": metrics,
        "metric_formulas": {
            "ED": (
                "mean over matched pre-stop probes of "
                "count(OLD in top-k) / retrieved_count"
            ),
            "UIR": (
                "mean update-phase rank_weight(NEW) / "
                "(rank_weight(OLD) + rank_weight(NEW)); "
                "rank_weight(r)=1/log2(r+1)"
            ),
            "EPS": (
                "mean_ED(delayed matched probes) / "
                "mean_ED(pre-stop matched probes)"
            ),
            "DBI": (
                "OLD-biased decision outputs / eligible decision probes; "
                "catch trials excluded"
            ),
        },
    }
