import os
import pickle
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from memory_layer import SimpleEmbeddingRetriever

os.environ.setdefault("MEM0_TELEMETRY", "false")

MODEL_NAME = "all-MiniLM-L6-v2"


@dataclass
class MemoryItem:
    memory_id: str
    text: str
    supplier_id: Optional[str]
    timestamp: str
    kind: str
    write_count: int = 1


class AMemoryAdapter:
    """A-MEM's embedding retrieval path, with cache-backed background memory."""
    framework = "A-MEM"

    def __init__(self, model: SentenceTransformer, cache_file: Optional[Path] = None):
        self.model = model
        self.items: List[MemoryItem] = []
        # Use the repository's actual A-MEM retriever. Inject the already loaded
        # model to avoid loading an identical SentenceTransformer per condition.
        self.retriever = SimpleEmbeddingRetriever.__new__(SimpleEmbeddingRetriever)
        self.retriever.model = model
        self.retriever.corpus = []
        self.retriever.embeddings = None
        self.retriever.document_ids = {}
        self.cache_source = str(cache_file) if cache_file else None
        if cache_file and cache_file.exists():
            with cache_file.open("rb") as fh:
                cached = pickle.load(fh)
            for j, note in enumerate(list(cached.values())[:2]):
                self.add(note.content, None, getattr(note, "timestamp", "cache"), "amem_cache")

    def _encode(self, texts):
        return self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

    def add(self, text, supplier_id, timestamp, kind):
        mid = f"amem-{len(self.items)}"
        self.items.append(MemoryItem(mid, text, supplier_id, timestamp, kind))
        self.retriever.add_documents([text])
        return mid

    def update(self, memory_id, text, timestamp):
        idx = next(i for i, x in enumerate(self.items) if x.memory_id == memory_id)
        item = self.items[idx]
        item.text, item.timestamp, item.write_count = text, timestamp, item.write_count + 1
        # Mirrors RobustAgenticMemorySystem.consolidate_memories after an update.
        self.retriever.corpus = [x.text for x in self.items]
        self.retriever.embeddings = self.model.encode(self.retriever.corpus)
        self.retriever.document_ids = {x: i for i, x in enumerate(self.retriever.corpus)}
        return memory_id

    def search(self, query, top_k=5):
        if not self.items:
            return []
        order = self.retriever.search(query, top_k)
        query_vec = self.model.encode([query])[0]
        scores = cosine_similarity([query_vec], self.retriever.embeddings)[0]
        return [{**asdict(self.items[i]), "retrieval_rank": rank, "retrieval_score": float(scores[i])}
                for rank, i in enumerate(order, 1)]

    def close(self):
        pass


class Mem0Adapter:
    """Official mem0ai Memory.add/search/update adapter."""
    framework = "Mem0"
    _shared_memory = None
    _shared_root = None

    def __init__(self, root: Path, namespace: str,
                 collection_name: str = "phase3_procurement",
                 embedding_model: str = MODEL_NAME):
        os.environ.setdefault("MEM0_DIR", str(root.parent))
        os.environ.setdefault("POSTHOG_DISABLED", "true")
        from mem0 import Memory
        self.user_id = namespace
        self.meta: Dict[str, MemoryItem] = {}
        config = {
            "vector_store": {"provider": "qdrant", "config": {
                "path": str(root / "qdrant"), "collection_name": collection_name,
                "embedding_model_dims": 384, "on_disk": False}},
            "embedder": {"provider": "huggingface", "config": {
                "model": embedding_model, "embedding_dims": 384}},
            "llm": {"provider": "ollama", "config": {"model": "unused"}},
            "history_db_path": str(root / "history.db"),
        }
        if Mem0Adapter._shared_memory is None:
            Mem0Adapter._shared_memory = Memory.from_config(config)
            Mem0Adapter._shared_root = root
        elif Mem0Adapter._shared_root != root:
            raise RuntimeError("Only one Mem0 benchmark root may be open per process")
        self.memory = Mem0Adapter._shared_memory

    def add(self, text, supplier_id, timestamp, kind):
        # Keep the benchmark timestamp in metadata. Mem0's dedicated temporal
        # reasoning timestamp parameter is a hosted-API feature, not OSS Memory.
        result = self.memory.add(text, user_id=self.user_id, infer=False,
                                 metadata={"supplier_id": supplier_id, "kind": kind, "timestamp": timestamp})
        mid = result["results"][0]["id"]
        self.meta[mid] = MemoryItem(mid, text, supplier_id, timestamp, kind)
        return mid

    def update(self, memory_id, text, timestamp):
        item = self.meta[memory_id]
        self.memory.update(memory_id, data=text, metadata={"supplier_id": item.supplier_id,
                           "kind": item.kind, "timestamp": timestamp})
        item.text, item.timestamp, item.write_count = text, timestamp, item.write_count + 1
        return memory_id

    def search(self, query, top_k=5):
        result = self.memory.search(query, top_k=top_k, filters={"user_id": self.user_id},
                                    threshold=0.0, rerank=False)
        hits = []
        for rank, row in enumerate(result.get("results", []), 1):
            mid = row["id"]
            item = self.meta.get(mid)
            md = row.get("metadata") or {}
            if item is None:
                item = MemoryItem(mid, row.get("memory", ""), md.get("supplier_id"),
                                  md.get("timestamp", ""), md.get("kind", "unknown"))
            hits.append({**asdict(item), "retrieval_rank": rank,
                         "retrieval_score": float(row.get("score", 0.0))})
        return hits

    def close(self):
        # Shared official Memory instance is closed by process teardown.
        pass

    @classmethod
    def close_shared(cls):
        if cls._shared_memory is None:
            return
        stores = [getattr(cls._shared_memory, "vector_store", None),
                  getattr(cls._shared_memory, "_telemetry_vector_store", None)]
        for store in stores:
            client = getattr(store, "client", None)
            if client is not None:
                client.close()
        cls._shared_memory = None
        cls._shared_root = None
