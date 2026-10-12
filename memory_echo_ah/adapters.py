"""Adapters used by the unified four-framework Memory Echo protocol.

Every adapter exposes the same explicit add/search/update operations.  The
benchmark controls *when* writes happen; a read-only condition must never gain
records as a side effect of retrieval.
"""
from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EchoMemoryItem:
    memory_id: str
    text: str
    supplier_id: str | None
    timestamp: str
    kind: str
    write_count: int = 1
    parent_id: str | None = None


class CoMAdapter:
    """Official CoM chain filtering over benchmark-controlled memory records."""

    framework = "CoM"

    def __init__(self, model, blocking_ratio: float = 0.5,
                 num_anchors: int = 3):
        source = ROOT / "third_party" / "CoM"
        if str(source) not in sys.path:
            sys.path.insert(0, str(source))
        from src.com.filter import com_filter
        from src.com.models import TurnNode

        self._com_filter = com_filter
        self._TurnNode = TurnNode
        self.model = model
        self.blocking_ratio = blocking_ratio
        self.num_anchors = num_anchors
        self.items: list[EchoMemoryItem] = []

    def add(self, text, supplier_id, timestamp, kind, parent_id=None):
        memory_id = f"com-{len(self.items)}"
        self.items.append(EchoMemoryItem(
            memory_id, text, supplier_id, timestamp, kind, parent_id=parent_id
        ))
        return memory_id

    def update(self, memory_id, text, timestamp):
        item = next(item for item in self.items if item.memory_id == memory_id)
        item.text = text
        item.timestamp = timestamp
        item.write_count += 1
        return memory_id

    def search(self, query, top_k=5):
        if not self.items:
            return []
        texts = [item.text for item in self.items]
        vectors = np.asarray(self.model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False
        ))
        query_vector = np.asarray(self.model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0])
        nodes = []
        for index, (item, vector) in enumerate(zip(self.items, vectors)):
            similarity = float(np.dot(query_vector, vector))
            nodes.append(self._TurnNode(
                turn_id=item.memory_id,
                text=item.text,
                embedding=vector,
                date=item.timestamp.split("T", 1)[0],
                session_idx=index,
                turn_idx=0,
                role="memory",
                similarity_to_query=similarity,
            ))
        result = self._com_filter(
            nodes, query_vector,
            blocking_ratio=self.blocking_ratio,
            num_anchors=min(self.num_anchors, len(nodes)),
            max_chain_length=max(top_k, 1),
            context_mode="last_node",
        )
        ordered = []
        seen = set()
        # CoM merges multiple evidence chains; unique memory records are needed
        # for a well-defined top-k occupancy denominator.
        for node in result.main_chain:
            if node.turn_id in seen:
                continue
            seen.add(node.turn_id)
            ordered.append(node)
        if len(ordered) < min(top_k, len(nodes)):
            for node in sorted(nodes, key=lambda item: item.similarity_to_query, reverse=True):
                if node.turn_id not in seen:
                    ordered.append(node)
                    seen.add(node.turn_id)
        by_id = {item.memory_id: item for item in self.items}
        return [
            {
                **asdict(by_id[node.turn_id]),
                "retrieval_rank": rank,
                "retrieval_score": float(node.similarity_to_query),
            }
            for rank, node in enumerate(ordered[:top_k], 1)
        ]

    def snapshot(self):
        return [asdict(item) for item in self.items]

    def close(self):
        pass


class _SentenceTransformerEmbeddings:
    """LangChain embedding interface backed by the frozen benchmark encoder."""

    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts):
        return self.model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        ).tolist()

    def embed_query(self, text):
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0].tolist()


class LangMemAdapter:
    """Official LangMem MemoryStoreManager with a local LangGraph vector store."""

    framework = "LangMem"

    def __init__(self, model, namespace: str):
        from langchain_openai import ChatOpenAI
        from langgraph.store.memory import InMemoryStore
        from langmem import create_memory_store_manager

        self.model = model
        self.items: dict[str, EchoMemoryItem] = {}
        embed = _SentenceTransformerEmbeddings(model)
        dims = len(embed.embed_query("dimension probe"))
        self.store = InMemoryStore(
            index={"dims": dims, "embed": embed.embed_documents}
        )
        # No language-model call is made by put/search.  A local placeholder is
        # supplied because the official manager constructor requires a chat
        # model for its optional extraction workflow.
        placeholder = ChatOpenAI(
            model="memory-manager-unused",
            api_key="not-used-by-explicit-put-search",
            base_url="http://127.0.0.1:9/v1",
        )
        self.manager = create_memory_store_manager(
            placeholder,
            namespace=("memory-echo", namespace),
            store=self.store,
            enable_inserts=True,
            enable_deletes=False,
        )

    def add(self, text, supplier_id, timestamp, kind, parent_id=None):
        memory_id = f"langmem-{len(self.items)}"
        item = EchoMemoryItem(
            memory_id, text, supplier_id, timestamp, kind, parent_id=parent_id
        )
        self.items[memory_id] = item
        self.manager.put(memory_id, {"content": text}, index=["content"])
        return memory_id

    def update(self, memory_id, text, timestamp):
        item = self.items[memory_id]
        item.text = text
        item.timestamp = timestamp
        item.write_count += 1
        self.manager.put(memory_id, {"content": text}, index=["content"])
        return memory_id

    def search(self, query, top_k=5):
        results = self.manager.search(query=query, limit=top_k)
        hits = []
        for rank, result in enumerate(results, 1):
            memory_id = result.key
            item = self.items.get(memory_id)
            if item is None:
                continue
            score = result.score
            hits.append({
                **asdict(item),
                "retrieval_rank": rank,
                "retrieval_score": 0.0 if score is None else float(score),
            })
        return hits

    def snapshot(self):
        return [asdict(item) for item in self.items.values()]

    def close(self):
        pass


def snapshot_adapter(adapter) -> list[dict[str, Any]]:
    """Return comparable stored records without changing framework state."""
    if hasattr(adapter, "snapshot"):
        return adapter.snapshot()
    if hasattr(adapter, "items"):
        return [asdict(item) for item in adapter.items]
    if hasattr(adapter, "meta"):
        return [asdict(item) for item in adapter.meta.values()]
    raise TypeError(f"Adapter {type(adapter).__name__} has no auditable storage view")


def derived_record_count(adapter) -> int:
    return sum(
        row.get("kind") in {"retrieval_writeback", "reconsolidated_echo"}
        for row in snapshot_adapter(adapter)
    )
