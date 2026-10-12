from __future__ import annotations

from dataclasses import asdict, dataclass

from memory_echo_solution.guard import EchoGuardMemory
from memory_echo_solution.trajectory import _evidence_decision


@dataclass
class _Item:
    memory_id: str
    text: str
    supplier_id: str
    timestamp: str
    kind: str


class _Adapter:
    framework = "Fake"

    def __init__(self):
        self.items = []

    def add(self, text, label, timestamp, kind):
        mid = f"m{len(self.items)}"
        self.items.append(_Item(mid, text, label, timestamp, kind))
        return mid

    def update(self, memory_id, text, timestamp):
        item = next(x for x in self.items if x.memory_id == memory_id)
        item.text, item.timestamp = text, timestamp

    def search(self, query, top_k=5):
        return [{**asdict(item), "retrieval_rank": rank, "retrieval_score": 1 / rank}
                for rank, item in enumerate(self.items[:top_k], 1)]

    def close(self):
        pass


def _add(memory, label="ECHO", family="echo", source_type="historical_unverified",
         parent=None):
    return memory.add(
        "claim", label, "t0", "kind", expected_family_id=family,
        source_id="source-1", source_type=source_type,
        independence_group="source-1", confidence=.35 if label == "ECHO" else .95,
        parent_memory_id=parent,
    )


def test_s1_inherits_family_and_source_lineage():
    memory = EchoGuardMemory(_Adapter(), "s1_identity")
    anchor = _add(memory)["memory_id"]
    child = _add(memory, source_type="retrieval_derived", parent=anchor)
    assert child["executed"]
    assert memory.claims[child["memory_id"]].family_id == memory.claims[anchor].family_id
    assert memory.audit()["family_pair_recall"] == 1.0
    assert memory.audit()["provenance_inheritance_accuracy"] == 1.0


def test_s2_blocks_retrieval_add_and_search_does_not_increase_confidence():
    memory = EchoGuardMemory(_Adapter(), "s2_retrieval_hygiene")
    anchor = _add(memory)["memory_id"]
    blocked = _add(memory, source_type="retrieval_derived", parent=anchor)
    before = memory.claims[anchor].confidence
    memory.search("query", 5, stage="retrieval_practice", session=1,
                  reinforce_activation=True)
    assert not blocked["executed"]
    assert memory.claims[anchor].confidence == before
    assert memory.audit()["blocked_derived_adds"] == 1
    assert memory.audit()["max_confidence_drift_on_search"] == 0


def test_s3_blocks_derived_update_and_binds_correction():
    memory = EchoGuardMemory(_Adapter(), "s3_reconsolidation")
    anchor = _add(memory)["memory_id"]
    blocked = memory.update(anchor, "reconsolidated", "t1",
                            kind="retrieval_reconsolidation",
                            source_type="retrieval_derived")
    memory.update(anchor, "claim [superseded]", "t2", kind="verified_supersession",
                  source_type="verified_external", status="superseded")
    _add(memory, label="CORRECTION", family="correction",
         source_type="verified_external")
    hits = memory.search("query", 5, stage="updating_reconsolidation", session=2)
    assert not blocked["executed"]
    assert {hit["label"] for hit in hits} >= {"ECHO", "CORRECTION"}


def test_s4_continuous_window_trigger_suppresses_persistent_echo():
    memory = EchoGuardMemory(_Adapter(), "s4_persistence",
                             persistence_trigger_windows=3,
                             persistence_excess_threshold=.05)
    _add(memory)
    for delay in (1, 2):
        assert not memory.observe_persistence(delay, .5, .1)["triggered"]
    assert memory.observe_persistence(3, .5, .1)["triggered"]
    assert memory.audit()["suppressed_claim_records"] == 1
    assert memory.audit()["remediation_triggers"] == 1


def test_full_counterfactual_evidence_rule_ignores_duplicate_votes():
    hits = [
        {"label": "ECHO", "confidence": .35, "status": "active",
         "source_type": "historical_unverified"},
        {"label": "ECHO", "confidence": .35, "status": "active",
         "source_type": "retrieval_derived"},
        {"label": "CORRECTION", "confidence": .95, "status": "active",
         "source_type": "verified_external"},
    ]
    assert _evidence_decision(hits) == "CORRECTION"


def test_baseline_executes_echo_copy_positive_control():
    memory = EchoGuardMemory(_Adapter(), "baseline")
    anchor = _add(memory)["memory_id"]
    result = _add(memory, source_type="retrieval_derived", parent=anchor)
    assert result["executed"]
    assert memory.audit()["stored_memories"] == 2
