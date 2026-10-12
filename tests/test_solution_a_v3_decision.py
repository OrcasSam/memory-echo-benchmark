import unittest

from scripts.legacy.run_memory_echo_abcd_freeform import apply_authority_answer_constraint


def row(evidence, answer):
    return {
        "snippets": [evidence],
        "freeform_answer": answer,
        "decision_context_audit": {
            "policy": "authority_gated_decision_context_v3"
        },
    }


class AuthorityAnswerConstraintTest(unittest.TestCase):
    def test_binary_contradiction_is_replaced(self):
        result = apply_authority_answer_constraint([row("no", "Yes.")])[0]
        self.assertEqual(result["freeform_answer"], "no")
        self.assertEqual(result["raw_freeform_answer"], "Yes.")
        self.assertFalse(result["authority_answer_constraint"]["uses_benchmark_label"])

    def test_short_entity_is_extracted(self):
        result = apply_authority_answer_constraint([
            row("Badly Drawn Boy", "Wolf Alice has a higher ratio.")
        ])[0]
        self.assertEqual(result["freeform_answer"], "Badly Drawn Boy")

    def test_long_answer_remains_generative(self):
        evidence = "The verified record says the first event happened on March 10."
        answer = "It happened on March 10."
        result = apply_authority_answer_constraint([row(evidence, answer)])[0]
        self.assertEqual(result["freeform_answer"], answer)
        self.assertFalse(result["authority_answer_constraint"]["applied"])

    def test_unsupported_refusal_recovers_evidence(self):
        evidence = "The verified record identifies the earlier event as the Python webinar."
        result = apply_authority_answer_constraint([
            row(evidence, "I don't have enough information.")
        ])[0]
        self.assertEqual(result["freeform_answer"], evidence)
        self.assertEqual(
            result["authority_answer_constraint"]["reason"],
            "unsupported_refusal_recovery",
        )


if __name__ == "__main__":
    unittest.main()
