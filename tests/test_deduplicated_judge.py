from scripts.ah.run_deduplicated_judge import request_key


def test_request_key_ignores_experiment_metadata():
    base = {
        "question": "q", "gold_answer": "g", "echo_answer": "e",
        "freeform_answer": "a", "framework": "A-MEM", "condition": "control",
    }
    other = {**base, "framework": "Mem0", "condition": "update", "solution": "D"}
    assert request_key(base) == request_key(other)


def test_request_key_changes_when_candidate_changes():
    base = {"question": "q", "gold_answer": "g", "echo_answer": "e", "freeform_answer": "a"}
    assert request_key(base) != request_key({**base, "freeform_answer": "different"})
