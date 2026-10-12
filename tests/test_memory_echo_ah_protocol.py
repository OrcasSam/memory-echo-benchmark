import numpy as np

from memory_echo_ah.adapters import CoMAdapter
from memory_echo_ah.protocol import run_protocol


class TinyEncoder:
    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        vocab = ("old", "new", "target", "unrelated", "superseded")
        rows = []
        for text in texts:
            row = np.asarray([text.lower().count(x) for x in vocab], dtype=float)
            if not row.any():
                row[-1] = 1
            if normalize_embeddings:
                row /= np.linalg.norm(row)
            rows.append(row)
        return np.asarray(rows)


SAMPLE = {
    "sample_id": "s1",
    "source": {"dataset": "test", "ability": "recall", "question": "target?"},
    "echo_seed": {"claim": "old target answer", "value": "old"},
    "correction": {"claim": "new target answer", "value": "new"},
    "ground_truth_value": "new",
    "echo_value": "old",
}


def test_conditions_separate_read_from_writeback(mock_com):
    model = TinyEncoder()
    repeated = run_protocol(
        SAMPLE, "read_only_repetition", CoMAdapter(model), model,
        practice_queries=3, delay_queries=30,
    )
    updated = run_protocol(
        SAMPLE, "retrieve_writeback", CoMAdapter(model), model,
        practice_queries=3, delay_queries=30,
    )
    assert repeated["metrics"]["derived_record_count"] == 0
    assert updated["metrics"]["derived_record_count"] == 3
    assert len(updated["trajectory"]["delayed_retrieval"]) == 30
