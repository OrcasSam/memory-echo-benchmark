"""Exact embedding cache used only to remove repeated encoder computation."""
from __future__ import annotations

import numpy as np


class CachedEncoder:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def encode(self, sentences, *args, **kwargs):
        scalar = isinstance(sentences, str)
        items = [sentences] if scalar else list(sentences)
        normalize = bool(kwargs.get("normalize_embeddings", False))
        keys = [(text, normalize) for text in items]
        missing = [text for text, key in zip(items, keys) if key not in self.cache]
        if missing:
            unique = list(dict.fromkeys(missing))
            vectors = self.model.encode(unique, *args, **kwargs)
            for text, vector in zip(unique, vectors):
                self.cache[(text, normalize)] = np.asarray(vector)
        result = np.stack([self.cache[key] for key in keys])
        return result[0] if scalar else result

    def __getattr__(self, name):
        return getattr(self.model, name)

