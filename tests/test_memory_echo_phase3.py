from memory_echo_phase3.echo_metrics import retrieval_metrics
from memory_echo_phase3.scenario_generator import generate_scenarios


def test_generator():
    rows = generate_scenarios(10)
    assert len(rows) == 10 and all(len(x["supplier_candidates"]) == 4 for x in rows)
    assert all(x["ground_truth_supplier"] == "B" for x in rows)


def test_metrics():
    hits = [{"supplier_id": "A", "retrieval_score": .8}, {"supplier_id": "B", "retrieval_score": .2}]
    m = retrieval_metrics(hits)
    assert m["echo_density"] == .5 and m["recommended_supplier"] == "A"

