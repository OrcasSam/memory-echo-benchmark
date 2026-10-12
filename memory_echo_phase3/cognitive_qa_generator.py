"""Cognitive-stage, multi-intent QA plans for longitudinal WebShop scenarios."""
from typing import Dict, List


PHASES = {
    "encoding": range(1, 11),
    "retrieval_practice": range(11, 51),
    "updating_reconsolidation": range(51, 66),
    "delayed_retrieval": range(66, 86),
    "final_decision": range(86, 101),
}

PHASE_ALLOCATIONS = {
    100: {
        "encoding": 10,
        "retrieval_practice": 40,
        "updating_reconsolidation": 15,
        "delayed_retrieval": 20,
        "final_decision": 15,
    },
    200: {
        "encoding": 20,
        "retrieval_practice": 80,
        "updating_reconsolidation": 30,
        "delayed_retrieval": 40,
        "final_decision": 30,
    },
}

PHASE_TYPES = {
    "encoding": ["recommendation", "comparison", "justification", "memory_recall",
                 "comparison", "recommendation", "justification", "memory_recall",
                 "consistency_reevaluation", "decision"],
    "retrieval_practice": ["memory_recall", "justification", "comparison",
                           "consistency_reevaluation", "recommendation"],
    "updating_reconsolidation": ["update_awareness", "comparison", "justification",
                                 "consistency_reevaluation", "recommendation"],
    "delayed_retrieval": ["memory_recall", "comparison", "update_awareness",
                          "consistency_reevaluation", "recommendation"],
    "final_decision": ["decision", "comparison", "justification",
                       "consistency_reevaluation", "recommendation"],
}


def _candidate(scenario: Dict, sid: str) -> Dict:
    return next(x for x in scenario["supplier_candidates"] if x["supplier_id"] == sid)


def phase_ranges(qa_count: int) -> Dict[str, range]:
    if qa_count not in PHASE_ALLOCATIONS:
        raise ValueError("Supported cognitive trajectory lengths are 100 and 200 QA")
    result = {}
    start = 1
    for name, size in PHASE_ALLOCATIONS[qa_count].items():
        result[name] = range(start, start + size)
        start += size
    return result


def _target(turn: int, qa_type: str, update_start: int) -> str:
    if qa_type == "memory_recall":
        return "A"
    return "A" if turn < update_start else "B"


def _question(s: Dict, turn: int, phase: str, qa_type: str, variant: int) -> str:
    req = s["user_requirement"]
    a, b, c = (_candidate(s, x) for x in "ABC")
    bank = {
        "recommendation": [
            f"Given my request '{req}', which currently available product would you recommend?",
            f"What is your current best match for my shopping requirement: {req}?",
            f"Which candidate best satisfies the requirement now, and what product is it?",
        ],
        "comparison": [
            f"Compare {a['name']} with {c['name']}. Which better fits '{req}'?",
            f"For '{req}', should I prefer {b['name']} or the earlier option {a['name']}?",
            f"Which candidate is the stronger requirement match when price, attributes, and category are considered?",
            f"Contrast the leading alternatives and identify the one that best meets the original request.",
        ],
        "justification": [
            f"Why is your selected product appropriate for '{req}'?",
            f"Explain which remembered attributes support your current recommendation.",
            f"Give the main evidence for choosing the best-matching candidate.",
        ],
        "memory_recall": [
            "Which product did you recommend earlier in this long-running shopping session?",
            f"Recall the historical candidate we first considered for {s['product']}.",
            "What was the ASIN or name of the previously preferred product?",
            "Before new evidence appeared, which option had been repeatedly discussed?",
        ],
        "update_awareness": [
            f"New catalog evidence is available. Should the earlier choice be updated for '{req}'?",
            f"Does {b['name']} change the previous judgment about {a['name']}? Explain the update.",
            "After considering the newly available product, what recommendation should replace the old one?",
        ],
        "decision": [
            f"I am ready to purchase for '{req}'. Which exact product should I choose now?",
            "Make the final purchase decision using all current and historical evidence.",
            "Confirm the final product name and ASIN I should select.",
        ],
        "consistency_reevaluation": [
            "Do you still stand by the earlier recommendation after reviewing the latest evidence?",
            f"Re-evaluate the old preference against the current requirement '{req}'.",
            "Should a consistent assistant preserve the old choice or revise it now? Give the revised choice.",
        ],
    }
    focus_bank = {
        "encoding": [
            "initial requirement fit", "catalog category", "available attributes",
            "price evidence", "the first recommendation", "alternative coverage",
        ],
        "retrieval_practice": [
            "the remembered product identity", "previously cited attributes",
            "historical price evidence", "earlier category fit",
            "the reason the option was retained", "consistency with prior advice",
        ],
        "updating_reconsolidation": [
            "new product identity", "changed attribute evidence", "old-versus-new fit",
            "whether the memory should be revised", "canonical requirement coverage",
            "what evidence supersedes the old choice",
        ],
        "delayed_retrieval": [
            "source and timeline", "retained historical details", "new evidence recall",
            "current requirement fit", "whether the update persisted",
            "separating old and current recommendations",
        ],
        "final_decision": [
            "purchase readiness", "exact ASIN", "requirement coverage",
            "old-versus-current evidence", "decision justification",
            "the final actionable choice",
        ],
    }
    base = bank[qa_type][variant % len(bank[qa_type])]
    focus = focus_bank[phase][variant % len(focus_bank[phase])]
    contexts = [
        "Use the evidence available in this session.",
        "Separate remembered history from the current catalog state.",
        "Consider the shopper's original constraints before answering.",
        "Base the response on the strongest currently retrievable evidence.",
        "Treat this as part of an ongoing multi-session purchase review.",
        "Explain the choice in a way that supports an eventual order.",
        "Distinguish an earlier preference from the best option now.",
        "Account for both product identity and requirement coverage.",
    ]
    context = contexts[variant % len(contexts)]
    return f"{base} For this question, focus on {focus}. {context}"


def _canonical_answer(s: Dict, turn: int, qa_type: str, target: str,
                      update_start: int) -> str:
    p = _candidate(s, target)
    a, b = _candidate(s, "A"), _candidate(s, "B")
    if qa_type == "memory_recall":
        return f"The earlier historical recommendation was {a['name']}, ASIN {a['product_asin']}."
    if qa_type == "comparison" and turn >= update_start:
        return (f"{b['name']}, ASIN {b['product_asin']}, is the better match for the original request; "
                f"it should replace the earlier option {a['name']}.")
    if qa_type == "update_awareness":
        return (f"Yes. Update the old judgment and prefer {b['name']}, ASIN {b['product_asin']}, "
                f"because it is the WebShop canonical target for the stated requirements.")
    if qa_type == "consistency_reevaluation" and turn >= update_start:
        return (f"Revise the earlier recommendation to {b['name']}, ASIN {b['product_asin']}, "
                f"in light of the newer and more relevant catalog evidence.")
    if qa_type == "justification":
        return (f"Recommend {p['name']}, ASIN {p['product_asin']}, because its product information "
                f"best matches the requested attributes, options, and category.")
    return f"Select {p['name']}, ASIN {p['product_asin']}, as the best match for the shopping request."


def generate_cognitive_qa_plan(scenario: Dict, qa_count: int = 100) -> List[Dict]:
    ranges = phase_ranges(qa_count)
    update_start = ranges["updating_reconsolidation"].start
    counters = {phase: 0 for phase in ranges}
    plan = []
    for turn in range(1, qa_count + 1):
        phase = next(name for name, turns in ranges.items() if turn in turns)
        index = counters[phase]
        types = PHASE_TYPES[phase]
        qa_type = types[index % len(types)]
        target = _target(turn, qa_type, update_start)
        plan.append({
            "turn": turn,
            "phase": phase,
            "qa_type": qa_type,
            "query": _question(scenario, turn, phase, qa_type, index),
            "canonical_target_supplier": target,
            "canonical_target_asin": _candidate(scenario, target)["product_asin"],
            "canonical_answer": _canonical_answer(
                scenario, turn, qa_type, target, update_start
            ),
        })
        counters[phase] += 1
    return plan
