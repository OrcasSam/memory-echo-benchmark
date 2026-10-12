import os
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from memory_echo_phase3.cognitive_qa_generator import generate_cognitive_qa_plan
from memory_echo_phase3.semantic_answer_evaluator import SemanticAnswerEvaluator
from memory_echo_phase3.webshop_loader import load_webshop_scenarios


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenario():
    if not (ROOT / "data" / "webshop" / "data").exists():
        pytest.skip("External WebShop data are not distributed with this repository")
    rows, _ = load_webshop_scenarios(ROOT / "data" / "webshop" / "data", 10)
    return rows[0]


def test_cognitive_plan_has_five_phases_and_seven_qa_types(scenario):
    plan = generate_cognitive_qa_plan(scenario)
    assert len(plan) == 100
    assert len({x["query"] for x in plan}) > 30
    assert {x["phase"] for x in plan} == {
        "encoding", "retrieval_practice", "updating_reconsolidation",
        "delayed_retrieval", "final_decision",
    }
    assert {x["qa_type"] for x in plan} == {
        "recommendation", "comparison", "justification", "memory_recall",
        "update_awareness", "decision", "consistency_reevaluation",
    }
    assert all(x["canonical_target_supplier"] == "A" for x in plan[:50])
    assert all(x["canonical_target_supplier"] == "A" for x in plan[50:]
               if x["qa_type"] == "memory_recall")
    assert all(x["canonical_target_supplier"] == "B" for x in plan[50:]
               if x["qa_type"] != "memory_recall")


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="Set RUN_MODEL_TESTS=1 to allow loading the external embedding model",
)
def test_semantic_evaluator_is_graded_and_target_aware():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    evaluator = SemanticAnswerEvaluator(model)
    canonical = "Select product Alpha, ASIN B0001, as the best match."
    correct = evaluator.evaluate(canonical, canonical, "B0001", "B0001")
    wrong_id = evaluator.evaluate(canonical, canonical, "B9999", "B0001")
    assert correct["semantic_similarity"] == pytest.approx(1.0, abs=1e-5)
    assert correct["answer_score"] == pytest.approx(1.0, abs=1e-5)
    assert wrong_id["answer_score"] == pytest.approx(0.7, abs=1e-5)
    assert wrong_id["semantic_grade"] == "partially_correct"

