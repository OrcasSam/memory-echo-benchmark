"""Phase-specific, multi-type QA plans for the generic benchmark."""
from typing import Dict, List


PHASES = (
    "echo_seeding",
    "update_adaptation",
    "echo_retention",
    "decision_influence",
)

PROTOCOL_PHASE_COUNTS_20 = {
    "memory_occupancy": (8, 5, 4, 3),
    "update_adaptation": (5, 8, 4, 3),
    "retention_stability": (5, 5, 7, 3),
    "decision_reliability": (4, 5, 4, 7),
}

PHASE_QA_TYPES = {
    # No comparison question is used before the conflicting evidence is
    # introduced; otherwise the later value would leak into Echo Seeding.
    "echo_seeding": ("recall", "justification", "consistency"),
    "update_adaptation": (
        "comparison", "update_awareness", "justification", "consistency"
    ),
    "echo_retention": (
        "recall", "comparison", "consistency", "update_awareness"
    ),
    "decision_influence": (
        "decision", "justification", "comparison", "consistency"
    ),
}


def _scaled_counts(protocol: str, qa_count: int):
    if not 20 <= qa_count <= 40:
        raise ValueError("Each generic trajectory must contain 20 to 40 QA")
    base = list(PROTOCOL_PHASE_COUNTS_20[protocol])
    counts = base[:]
    for i in range(qa_count - 20):
        counts[i % 4] += 1
    return counts


def _target_label(sample: Dict, phase: str, qa_type: str) -> str:
    if qa_type == "recall":
        return "OLD"
    return (
        "OLD" if phase == "echo_seeding"
        else sample["canonical_current_label"]
    )


def _query(sample: Dict, phase: str, qa_type: str, occurrence: int) -> str:
    subject = sample["subject"]
    old = sample["initial_memory"]["value"]
    new = sample["conflicting_evidence"]["value"]
    memory_type = sample["memory_unit_type"].replace("_", " ")
    banks = {
        "recall": [
            f"What was the original {memory_type} stored for {subject}?",
            f"Recall the historical value first associated with {subject}.",
            f"Which value for {subject} belonged to the initial memory?",
        ],
        "comparison": [
            f"Compare the historical state with the later evidence for {subject}. Which is current?",
            f"Which evidence should now carry more weight for {subject}: the initial record or the later record?",
            f"Separate the earlier state from the better-supported current state for {subject}.",
        ],
        "update_awareness": [
            f"Does the memory for {subject} need revision after the later evidence?",
            f"Has the applicable new evidence replaced the initial value for {subject}?",
            f"What should the current memory state for {subject} be after the conflict?",
        ],
        "consistency": [
            f"Is retaining the initial value for {subject} consistent with all evidence?",
            f"Re-evaluate the prior conclusion for {subject} and state whether it should change.",
            f"Distinguish consistency with history from consistency with current evidence for {subject}.",
        ],
        "decision": [
            f"A downstream action requires one value for {subject}. Choose the valid state.",
            f"Make the final state-dependent decision for {subject} using both old and new evidence.",
            f"Which value should control the next action for {subject}?",
        ],
        "justification": [
            f"Explain why the selected value for {subject} is supported by the available memory evidence.",
            f"Give the evidence-based reason for the current state of {subject}.",
            f"Justify the currently valid {memory_type} for {subject}.",
        ],
    }
    if phase == "echo_seeding":
        contexts = (
            "Use only the initial memory available at this point.",
            "Treat this as an initial-state retrieval probe.",
            "Do not infer any later event.",
        )
    else:
        contexts = (
            "Keep historical identity separate from current validity.",
            "Use the complete memory timeline.",
            "Do not treat repetition alone as evidence quality.",
            "Account for the conflict between the two memory states.",
            "Base the answer on what should be valid now.",
        )
    if phase == "echo_seeding" and qa_type == "consistency":
        stem = (
            f"Without using any later evidence, state the currently stored "
            f"{memory_type} for {subject}."
        )
    else:
        stem = banks[qa_type][occurrence % len(banks[qa_type])]
    if qa_type == "decision":
        ambiguity = {
            "low": "The action requires the exact currently valid state.",
            "medium": "Historical continuity matters, but the replacement rule controls validity.",
            "high": "The choice is deliberately ambiguous; resolve it with the explicit replacement rule.",
        }[sample["decision_ambiguity"]]
    else:
        ambiguity = ""
    return (
        f"{stem} {contexts[occurrence % len(contexts)]} {ambiguity}"
    ).strip()


def _canonical_answer(sample: Dict, phase: str, qa_type: str,
                      target: str) -> str:
    old = sample["initial_memory"]
    new = sample["conflicting_evidence"]
    subject = sample["subject"]
    if qa_type == "recall":
        return (
            f"The original historical memory for {subject} was {old['value']}. "
            "It should be remembered as the earlier state, not the current state."
        )
    current = old if target == "OLD" else new
    other = new if target == "OLD" else old
    if qa_type == "comparison":
        return (
            f"{current['value']} is the currently valid value for {subject}; "
            f"{other['value']} should not control the present state."
        )
    if qa_type == "update_awareness":
        if target == "NEW":
            return (
                f"Yes. Incorporate the new evidence and update {subject} from "
                f"{old['value']} to {new['value']}."
            )
        return (
            f"No replacement is justified yet. Retain {old['value']} for "
            f"{subject} because the conflicting evidence is insufficient."
        )
    if qa_type == "consistency":
        return (
            f"A consistent current judgment uses {current['value']} for {subject} "
            f"and treats {other['value']} as non-controlling evidence."
        )
    if qa_type == "decision":
        return (
            f"The downstream decision should use {current['value']} for {subject}, "
            "because it is the current evidence-supported state."
        )
    value = old["value"] if target == "OLD" else new["value"]
    if target == "OLD" and phase != "echo_seeding":
        reason = "retained because the conflicting evidence is not valid for replacement"
    elif target == "OLD":
        reason = "the valid initial memory state"
    else:
        reason = "the newly supported current memory state"
    return (
        f"Use {value} for {subject} because it is {reason}."
    )


def generate_qa_plan(sample: Dict, qa_count: int = 20) -> List[Dict]:
    counts = _scaled_counts(sample["protocol"], qa_count)
    plan = []
    turn = 1
    for phase, count in zip(PHASES, counts):
        types = PHASE_QA_TYPES[phase]
        occurrences = {qa_type: 0 for qa_type in types}
        for index in range(count):
            standardized_persistence_probe = (
                phase in ("update_adaptation", "echo_retention")
                and index == count - 1
            )
            qa_type = (
                "consistency"
                if standardized_persistence_probe
                else types[index % len(types)]
            )
            occurrences.setdefault(qa_type, 0)
            occurrence = occurrences[qa_type]
            occurrences[qa_type] += 1
            target = _target_label(sample, phase, qa_type)
            target_value = (
                sample["initial_memory"]["value"]
                if target == "OLD" else sample["conflicting_evidence"]["value"]
            )
            plan.append({
                "turn": turn,
                "phase": phase,
                "qa_type": qa_type,
                "query": (
                    f"According to the applicable evidence, what state "
                    f"currently controls abstract unit {sample['subject']}?"
                    if standardized_persistence_probe else
                    _query(sample, phase, qa_type, occurrence)
                ),
                "persistence_probe_id": (
                    "standard-current-state"
                    if standardized_persistence_probe else None
                ),
                "canonical_target_label": target,
                "canonical_target_value": target_value,
                "canonical_answer": _canonical_answer(
                    sample, phase, qa_type, target
                ),
            })
            turn += 1
    return plan
