"""Semantic and target-aware graded answer evaluation."""
from typing import Dict

import numpy as np


THRESHOLDS = {
    "fully_correct": 0.90,
    "semantically_correct": 0.80,
    "partially_correct": 0.65,
}


class SemanticAnswerEvaluator:
    def __init__(self, model):
        self.model = model
        self._canonical_cache = {}

    def _embedding(self, text: str):
        if text not in self._canonical_cache:
            self._canonical_cache[text] = self.model.encode(
                [text], normalize_embeddings=True, show_progress_bar=False
            )[0]
        return self._canonical_cache[text]

    def preload(self, texts, batch_size: int = 256):
        """Batch-cache texts used repeatedly by large benchmark runs."""
        pending = list(dict.fromkeys(
            text for text in texts if text not in self._canonical_cache
        ))
        if not pending:
            return
        vectors = self.model.encode(
            pending,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._canonical_cache.update(zip(pending, vectors))

    def evaluate(self, answer: str, canonical_answer: str,
                 predicted_asin: str | None, canonical_asin: str) -> Dict:
        # Agent answers repeat across equivalent retrieval outcomes. Reusing
        # their vectors preserves exact scores while avoiding thousands of
        # redundant one-sentence model calls in the 6,000-turn experiment.
        answer_embedding = self._embedding(answer)
        semantic_similarity = float(np.dot(answer_embedding, self._embedding(canonical_answer)))
        semantic_similarity = max(0.0, min(1.0, semantic_similarity))
        target_match = float(predicted_asin == canonical_asin)
        # Preserve semantic credit while requiring target identity for a high score.
        answer_score = 0.70 * semantic_similarity + 0.30 * target_match
        if answer_score >= THRESHOLDS["fully_correct"]:
            grade = "fully_correct"
        elif answer_score >= THRESHOLDS["semantically_correct"]:
            grade = "semantically_correct"
        elif answer_score >= THRESHOLDS["partially_correct"]:
            grade = "partially_correct"
        else:
            grade = "incorrect"
        return {
            "semantic_similarity": semantic_similarity,
            "target_id_match": target_match,
            "answer_score": answer_score,
            "semantic_grade": grade,
            "thresholds": THRESHOLDS,
        }
