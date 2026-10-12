import numpy as np

from memory_echo_ah.adapters import CoMAdapter, LangMemAdapter, derived_record_count


class TinyEncoder:
    vocab = ("alpha", "beta", "noise")

    def encode(self, texts, normalize_embeddings=False, show_progress_bar=False):
        rows = []
        for text in texts:
            vector = np.asarray([text.lower().count(token) for token in self.vocab], dtype=float)
            if not vector.any():
                vector[-1] = 1.0
            if normalize_embeddings:
                vector /= np.linalg.norm(vector)
            rows.append(vector)
        return np.asarray(rows)


def _exercise(adapter):
    first = adapter.add("alpha fact", "ECHO", "2026-01-01T00:00:00Z", "echo_seed")
    adapter.add("beta correction", "CORRECTION", "2026-01-02T00:00:00Z", "verified_correction")
    hits = adapter.search("alpha", top_k=2)
    assert hits[0]["supplier_id"] == "ECHO"
    adapter.update(first, "beta supersedes alpha", "2026-01-03T00:00:00Z")
    assert len(adapter.search("beta", top_k=2)) == 2
    assert derived_record_count(adapter) == 0


def test_com_adapter(mock_com):
    _exercise(CoMAdapter(TinyEncoder()))


def test_langmem_adapter():
    _exercise(LangMemAdapter(TinyEncoder(), "test"))
