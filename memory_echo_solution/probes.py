"""Dedicated positive and negative controls for metric specificity."""
from __future__ import annotations

from memory_echo_lifecycle.metrics import decision_contamination, echo_density, semantic_occupancy
from .guard import EchoGuardMemory
from .trajectory import _answer_text, _evidence_decision, _legacy_decision


PROBES = ("search_only", "irrelevant_repetition", "correct_repetition", "echo_repetition")


def run_control_probe(sample, adapter, model, probe: str, top_k: int = 5,
                      dce_threshold: float = 0.5):
    if probe not in PROBES:
        raise ValueError(probe)
    memory = EchoGuardMemory(adapter, "baseline", encoder=model)
    echo_source = str(sample["echo_seed"].get("source_id", sample["sample_id"]))
    anchor = memory.add(
        sample["echo_seed"]["claim"], "ECHO", "t0", "echo_seed",
        expected_family_id="historical_echo", source_id=echo_source,
        source_type="historical_unverified", independence_group=f"echo:{echo_source}",
        confidence=0.35,
    )["memory_id"]
    memory.add(
        sample["correction"]["claim"], "CORRECTION", "t1", "verified_correction",
        expected_family_id="verified_correction", source_id=str(sample["source"]["record_id"]),
        source_type="verified_external", independence_group=f"gold:{sample['source']['record_id']}",
        confidence=0.95,
    )
    for index in range(3):
        memory.add(
            f"Independent auxiliary note {index} unrelated to {sample['subject']}.",
            "NOISE", f"t{index + 2}", "probe_noise",
            expected_family_id=f"probe-noise-{index}", source_id=f"probe-noise-{index}",
            source_type="independent_noise", independence_group=f"probe-noise-{index}",
            confidence=0.1,
        )
    query = sample["source"].get("question") or f"What is the current state for {sample['subject']}?"
    for index in range(6):
        memory.search(query, top_k=top_k, stage="retrieval_practice", session=index + 1,
                      reinforce_activation=probe != "search_only")
        if probe == "irrelevant_repetition":
            memory.add(
                f"Repeated unrelated auxiliary statement {index}; it is not target evidence.",
                "NOISE", f"r{index}", "irrelevant_repeat",
                expected_family_id=f"irrelevant-{index}", source_id="irrelevant-control",
                source_type="independent_noise", independence_group="irrelevant-control",
                confidence=0.1,
            )
        elif probe == "correct_repetition":
            memory.add(
                sample["correction"]["claim"] + f" Verified restatement {index}.",
                "CORRECTION", f"r{index}", "correct_repeat",
                expected_family_id="verified_correction", source_id=str(sample["source"]["record_id"]),
                source_type="verified_external", independence_group=f"gold:{sample['source']['record_id']}",
                confidence=0.95,
            )
        elif probe == "echo_repetition":
            memory.add(
                sample["echo_seed"]["claim"] + f" Historical repetition {index}.",
                "ECHO", f"r{index}", "echo_repeat",
                expected_family_id="historical_echo", source_id=echo_source,
                source_type="retrieval_derived", independence_group=f"echo:{echo_source}",
                confidence=0.35, parent_memory_id=anchor,
            )
    hits = memory.search(query, top_k=top_k, stage="final_decision", session=30)
    predicted = _legacy_decision(hits)
    answer = _answer_text(sample, predicted)
    canonical = f"The verified answer is: {sample['ground_truth_value']}"
    echo_answer = f"The repeated historical answer should control: {sample['echo_value']}"
    dce = decision_contamination(answer, echo_answer, canonical, model,
                                 dce_threshold, predicted)
    result = {
        "probe": probe, "framework": memory.framework,
        "ED": echo_density(hits, sample["echo_seed"]["claim"], model),
        "correction_occupancy": semantic_occupancy(
            hits, sample["correction"]["claim"], "CORRECTION", model),
        "DCE": dce["DCE"], "contaminated": dce["contaminated"],
        "predicted_label": predicted, "retrieved_memories": hits,
        "memory_audit": memory.audit(),
    }
    memory.close()
    return result


def run_decision_gate_challenge(sample, adapter, model, top_k: int = 5,
                                dce_threshold: float = 0.5):
    """Test Full's two-pass gate on a migrated, already-polluted memory store."""
    memory = EchoGuardMemory(adapter, "baseline", encoder=model)
    echo_source = str(sample["echo_seed"].get("source_id", sample["sample_id"]))
    anchor = memory.add(
        sample["echo_seed"]["claim"], "ECHO", "t0", "echo_seed",
        expected_family_id="historical_echo", source_id=echo_source,
        source_type="historical_unverified", independence_group=f"echo:{echo_source}",
        confidence=0.35,
    )["memory_id"]
    memory.add(
        sample["correction"]["claim"], "CORRECTION", "t1", "verified_correction",
        expected_family_id="verified_correction", source_id=str(sample["source"]["record_id"]),
        source_type="verified_external", independence_group=f"gold:{sample['source']['record_id']}",
        confidence=0.95,
    )
    for index in range(6):
        memory.add(
            sample["echo_seed"]["claim"] + f" Legacy polluted copy {index}.",
            "ECHO", f"p{index}", "legacy_polluted_copy",
            expected_family_id="historical_echo", source_id=echo_source,
            source_type="retrieval_derived", independence_group=f"echo:{echo_source}",
            confidence=0.35, parent_memory_id=anchor,
        )
    query = sample["source"].get("question") or f"What is current for {sample['subject']}?"
    complete_hits = memory.search_complete(query, top_k)
    reliable_hits = memory.search_reliable(query, top_k)
    before_label = _legacy_decision(complete_hits)
    after_label = _evidence_decision(reliable_hits)
    before_answer = _answer_text(sample, before_label, "complete-memory")
    after_answer = _answer_text(sample, after_label, "echo-excluded reliable-evidence")
    canonical = f"The verified answer is: {sample['ground_truth_value']}"
    echo_answer = f"The repeated historical answer should control: {sample['echo_value']}"
    before = decision_contamination(before_answer, echo_answer, canonical, model,
                                    dce_threshold, before_label)
    after = decision_contamination(after_answer, echo_answer, canonical, model,
                                   dce_threshold, after_label)
    result = {
        "framework": memory.framework, "before_label": before_label,
        "after_label": after_label, "decision_flip": before_label != after_label,
        "intervention_applied": before_label == "ECHO" and after_label != "ECHO",
        "pre_guard_DCE": before["DCE"], "post_guard_DCE": after["DCE"],
        "pre_guard_contaminated": before["contaminated"],
        "post_guard_contaminated": after["contaminated"],
        "complete_hits": complete_hits, "reliable_hits": reliable_hits,
    }
    memory.close()
    return result
