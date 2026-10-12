"""Generate purpose-specific QA at every Echo lifecycle stage."""
from __future__ import annotations

from typing import Dict, List


def _qa(stage, qa_type, query, target, purpose, delay=None):
    return {
        "stage": stage, "qa_type": qa_type, "query": query,
        "target_label": target, "purpose": purpose, "delay": delay,
    }


def generate_stage_qa(sample: Dict, final_qa: int = 4) -> List[Dict]:
    subject = sample["subject"]
    unit = sample["memory_unit_type"]
    source_question = sample.get("source", {}).get("question")
    if source_question:
        return _generate_source_qa(sample, final_qa)
    rows = [
        _qa("encoding", "recall",
            f"What {unit} was initially encoded for {subject}?", "ECHO",
            "Test whether the Echo seed was established."),
        _qa("encoding", "consistency",
            f"State the only encoded value currently available for {subject}.", "ECHO",
            "Test initial representation consistency."),
    ]
    practice_types = ("recall", "comparison", "consistency", "justification")
    for index in range(sample["practice_qa"]):
        qa_type = practice_types[index % len(practice_types)]
        rows.append(_qa(
            "retrieval_practice", qa_type,
            f"Retrieve the repeatedly used {unit} state for {subject}; report the controlling value.",
            "ECHO", "Measure repeated activation and retrieval-space occupation.",
        ))
    rows.extend([
        _qa("updating_reconsolidation", "update_awareness",
            f"After verified superseding evidence, what is the current {unit} for {subject}?",
            "CORRECTION", "Test whether correction replaces the Echo."),
        _qa("updating_reconsolidation", "comparison",
            f"Compare the earlier repeated state and verified revision for {subject}; which controls now?",
            "CORRECTION", "Test evidence weighting during reconsolidation."),
        _qa("updating_reconsolidation", "justification",
            f"Justify the currently valid state for {subject} after reconsolidation.",
            "CORRECTION", "Test whether the update is semantically incorporated."),
    ])
    for delay in sample["delay_points"]:
        rows.append(_qa(
            "delayed_retrieval", "delayed_recall",
            f"After a gap of {delay} sessions, retrieve the current {unit} for {subject}.",
            "CORRECTION", "Measure Echo decay without further reinforcement.", delay,
        ))
    final_types = ("decision", "justification", "consistency", "comparison")
    for index in range(final_qa):
        qa_type = final_types[index % len(final_types)]
        rows.append(_qa(
            "final_decision", qa_type,
            f"A downstream action depends on {subject}. Using verified current evidence, select and explain the governing state.",
            "CORRECTION", "Measure whether the Echo contaminates action.",
        ))
    for turn, row in enumerate(rows, 1):
        row["turn"] = turn
        target = sample["echo_seed"] if row["target_label"] == "ECHO" else sample["correction"]
        if row["target_label"] == "CORRECTION":
            row["canonical_answer"] = (
                f"Use verified corrected state {target['value']} for {subject}. "
                "The correction supersedes the repeated historical Echo and governs the decision."
            )
        else:
            row["canonical_answer"] = (
                f"Recall initially encoded state {target['value']} for {subject}; no correction exists yet."
            )
        row["echo_answer"] = (
            f"Use repeated historical Echo state {sample['echo_value']} for {subject}. "
            "Its retrieval frequency overrides the verified correction and governs the decision."
        )
    return rows


def _generate_source_qa(sample: Dict, final_qa: int) -> List[Dict]:
    question = sample["source"]["question"]
    ability = sample["source"]["ability"]
    rows = [
        _qa("encoding", "recall",
            f"Using only the initially encoded memory, answer: {question}", "ECHO",
            "Test whether the source-derived Echo answer is encoded."),
        _qa("encoding", "consistency",
            f"State the answer currently stored for this {ability} question: {question}", "ECHO",
            "Test consistency of the initial Echo representation."),
    ]
    types = ("recall", "comparison", "consistency", "justification")
    for index in range(sample["practice_qa"]):
        rows.append(_qa(
            "retrieval_practice", types[index % len(types)],
            f"Retrieve the repeatedly used memory and answer the original question: {question}",
            "ECHO", "Test whether repeated access crowds the retrieval space.",
        ))
    rows.extend([
        _qa("updating_reconsolidation", "update_awareness",
            f"Verified evidence has corrected the stored answer. What is the current answer to: {question}",
            "CORRECTION", "Test adoption of the source gold answer."),
        _qa("updating_reconsolidation", "comparison",
            f"Resolve the conflict between the repeated historical answer and verified evidence for: {question}",
            "CORRECTION", "Test evidence weighting during reconsolidation."),
        _qa("updating_reconsolidation", "justification",
            f"Give the verified answer and justify it from the corrected record: {question}",
            "CORRECTION", "Test semantic incorporation of correction evidence."),
    ])
    for delay in sample["delay_points"]:
        rows.append(_qa(
            "delayed_retrieval", "delayed_recall",
            f"After {delay} intervening sessions, answer from the current valid memory: {question}",
            "CORRECTION", "Test persistence after reinforcement stops.", delay,
        ))
    final_types = ("decision", "justification", "consistency", "comparison")
    for index in range(final_qa):
        rows.append(_qa(
            "final_decision", final_types[index % len(final_types)],
            f"A downstream decision requires the factually valid answer. Decide using memory: {question}",
            "CORRECTION", "Test transfer of Echo into the final response.",
        ))
    for turn, row in enumerate(rows, 1):
        row["turn"] = turn
        value = sample["echo_value"] if row["target_label"] == "ECHO" else sample["ground_truth_value"]
        row["canonical_answer"] = (
            f"The verified answer is: {value}" if row["target_label"] == "CORRECTION"
            else f"The initially stored historical answer is: {value}"
        )
        row["echo_answer"] = f"The repeated historical answer should control: {sample['echo_value']}"
    return rows
