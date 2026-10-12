import os
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from memory_echo_phase3.cognitive_aggregator import (
    aggregate_by_phase, aggregate_by_phase_and_qa_type,
)
from memory_echo_phase3.cognitive_qa_generator import generate_cognitive_qa_plan
from memory_echo_phase3.dataset_rewriter_v6 import rewrite_webshop_scenario
from memory_echo_phase3.semantic_answer_evaluator import SemanticAnswerEvaluator
from memory_echo_phase3.webshop_loader import load_webshop_scenarios


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenario():
    if not (ROOT / "data" / "webshop" / "data").exists():
        pytest.skip("External WebShop data are not distributed with this repository")
    rows, _ = load_webshop_scenarios(ROOT / "data" / "webshop" / "data", 10)
    return rewrite_webshop_scenario(rows[0])


def test_v6_dataset_is_longitudinal_and_natural(scenario):
    plan = generate_cognitive_qa_plan(scenario)
    assert scenario["shopping_context"]["agent_role"] == "single_agent_long_term_shopping_assistant"
    assert scenario["candidate_timeline"][1]["available_candidates"] == ["A", "B", "C", "D"]
    assert "artificially corrupted" in scenario["ground_truth_rationale"]
    assert len(plan) == 100
    assert len({q["query"] for q in plan}) > 30
    assert len({q["qa_type"] for q in plan}) == 7
    assert len({q["phase"] for q in plan}) == 5
    assert all(q["canonical_answer"] and not isinstance(q["canonical_answer"], bool) for q in plan)


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 to allow loading the external embedding model",
)
def test_semantic_evaluation_uses_continuous_thresholds():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    evaluator = SemanticAnswerEvaluator(model)
    canonical = "Select product Alpha, ASIN B0001, because it best matches the request."
    exact = evaluator.evaluate(canonical, canonical, "B0001", "B0001")
    wrong_target = evaluator.evaluate(canonical, canonical, "B9999", "B0001")
    assert exact["answer_score"] == pytest.approx(1.0, abs=1e-5)
    assert wrong_target["answer_score"] == pytest.approx(0.7, abs=1e-5)
    assert set(exact["thresholds"]) == {
        "fully_correct", "semantically_correct", "partially_correct"
    }


def test_phase_aware_aggregation_schema():
    def turn(phase, qa_type, ed, score):
        return {
            "phase": phase, "qa_type": qa_type, "echo_density": ed,
            "answer_evaluation": {
                "semantic_similarity": score, "answer_score": score,
                "target_id_match": 1.0, "semantic_grade": "fully_correct",
            },
        }
    samples = [{"runs": [
        {"framework": "A-MEM", "condition": "control",
         "memory_trajectory": [turn("encoding", "recommendation", 0.2, 0.9)]},
        {"framework": "A-MEM", "condition": "historical_repetition_update",
         "memory_trajectory": [turn("encoding", "recommendation", 0.7, 0.6)]},
    ]}]
    phase_qa = aggregate_by_phase_and_qa_type(samples)
    phase = aggregate_by_phase(samples)
    assert len(phase_qa) == 2
    assert {"framework", "condition", "phase", "qa_type", "echo_density",
            "answer_score"} <= set(phase_qa[0])
    update = next(r for r in phase if r["condition"] == "historical_repetition_update")
    assert update["phase_ed_lift_vs_control"] == pytest.approx(0.5)
    assert update["phase_answer_score_drop_vs_control"] == pytest.approx(0.3)
