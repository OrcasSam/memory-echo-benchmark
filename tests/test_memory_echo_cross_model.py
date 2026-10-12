from memory_echo_multimodel.judge import _extract_json
from memory_echo_multimodel.providers import RateLimiter, apply_authority_answer_constraint, build_prompt


def test_prompt_contains_no_benchmark_answer_labels():
    prompt = build_prompt({
        "question": "What is current?", "snippets": ["new fact"],
        "decision_instruction": "Use reliable evidence.",
    })
    assert "new fact" in prompt
    assert "ECHO" not in prompt
    assert "CORRECTION" not in prompt


def test_a_v3_short_authoritative_answer_constraint():
    row = {
        "snippets": ["verified value"],
        "freeform_answer": "I cannot determine the answer.",
        "decision_context_audit": {"policy": "authority_gated_decision_context_v3"},
    }
    result = apply_authority_answer_constraint(row)
    assert result["freeform_answer"] == "verified value"
    assert result["raw_freeform_answer"] == "I cannot determine the answer."
    assert result["authority_answer_constraint"]["uses_benchmark_label"] is False


def test_non_a_row_is_unchanged():
    row = {"snippets": ["verified value"], "freeform_answer": "other"}
    assert apply_authority_answer_constraint(row) == row


def test_json_extraction_accepts_fenced_output():
    parsed = _extract_json('```json\n{"correctness_score": 4}\n```')
    assert parsed["correctness_score"] == 4


def test_rate_limiter_can_be_disabled():
    limiter = RateLimiter(0)
    limiter.wait()
    assert limiter.interval == 0
