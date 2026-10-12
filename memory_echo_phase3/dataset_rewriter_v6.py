"""Rewrite WebShop tasks as realistic single-agent longitudinal shopping cases."""
from copy import deepcopy
from typing import Dict, List


def rewrite_webshop_scenario(scenario: Dict) -> Dict:
    """Add scenario semantics without changing WebShop evidence or ground truth."""
    row = deepcopy(scenario)
    requirement = row["user_requirement"]
    category = row["product"]
    row["user_background"] = (
        f"A returning shopper uses one AI shopping assistant to track alternatives for {category} "
        "across multiple sessions before placing an order."
    )
    row["shopping_context"] = {
        "agent_role": "single_agent_long_term_shopping_assistant",
        "decision_style": "evidence-based comparison before purchase",
        "original_requirement": requirement,
        "memory_scope": "external long-term memory shared across the 100-turn trajectory",
    }
    row["candidate_timeline"] = [
        {
            "phase": "encoding",
            "event": "Historical candidate A and two plausible alternatives are first reviewed.",
            "available_candidates": ["A", "C", "D"],
        },
        {
            "phase": "updating_reconsolidation",
            "event": "The WebShop canonical target B becomes available as new catalog evidence.",
            "available_candidates": ["A", "B", "C", "D"],
        },
        {
            "phase": "final_decision",
            "event": "The shopper asks for an actionable final choice using all evidence.",
            "available_candidates": ["A", "B", "C", "D"],
        },
    ]
    row["ground_truth_rationale"] = (
        "Candidate B is the product paired with the original human WebShop instruction. "
        "Candidate A is a plausible retrieval-ranked historical alternative, not an "
        "artificially corrupted answer."
    )
    return row


def rewrite_webshop_dataset(scenarios: List[Dict]) -> List[Dict]:
    return [rewrite_webshop_scenario(s) for s in scenarios]
