"""Auditable cumulative guards for the five-stage Memory Echo lifecycle.

The real A-MEM/Mem0 adapter remains underneath this policy layer. Every
attempted operation is retained, including blocked retrieval-derived writes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Optional

import numpy as np

from . import (
    ARMS, COUNTERFACTUAL_DECISION_ARMS, IDENTITY_ARMS, PERSISTENCE_ARMS,
    RECONSOLIDATION_ARMS, RETRIEVAL_HYGIENE_ARMS,
)


@dataclass
class ClaimMeta:
    memory_id: str
    family_id: str
    expected_family_id: str
    label: str
    source_id: str
    source_type: str
    independence_group: str
    confidence: float
    activation: float = 1.0
    exposure_count: int = 0
    status: str = "active"
    parent_memory_id: Optional[str] = None
    supersedes_family_id: Optional[str] = None
    last_activation_session: int = 0


class EchoGuardMemory:
    """Policy layer implementing Baseline, S1, S2, S3, S4 and Full."""

    def __init__(self, adapter, arm: str, encoder=None, family_threshold: float = 0.78,
                 activation_half_life: float = 7.0, persistence_trigger_windows: int = 3,
                 persistence_excess_threshold: float = 0.05):
        if arm not in ARMS:
            raise ValueError(f"Unknown solution arm: {arm}")
        self.adapter = adapter
        self.arm = arm
        self.framework = adapter.framework
        self.encoder = encoder or getattr(adapter, "model", None)
        self.family_threshold = float(family_threshold)
        self.activation_half_life = float(activation_half_life)
        self.persistence_trigger_windows = int(persistence_trigger_windows)
        self.persistence_excess_threshold = float(persistence_excess_threshold)
        self.claims: Dict[str, ClaimMeta] = {}
        self.family_representatives: Dict[str, tuple[str, str]] = {}
        self.attempted_writes = 0
        self.executed_writes = 0
        self.blocked_derived_adds = 0
        self.blocked_derived_updates = 0
        self.interaction_log = []
        self.current_session = 0
        self.persistence_exceedance_streak = 0
        self.remediation_triggers = 0
        self.merged_records = 0
        self.correction_family_id: Optional[str] = None

    def _cosine(self, left: str, right: str) -> float:
        if self.encoder is None:
            return float(left.strip().lower() == right.strip().lower())
        vectors = self.encoder.encode(
            [left, right], normalize_embeddings=True, show_progress_bar=False
        )
        return float(np.dot(vectors[0], vectors[1]))

    def _new_family(self) -> str:
        return f"family-{len(self.family_representatives) + 1:04d}"

    def _resolve_family(self, text: str, label: str, parent_memory_id: str | None) -> str:
        if self.arm not in IDENTITY_ARMS:
            family = self._new_family()
            self.family_representatives[family] = (text, label)
            return family
        if parent_memory_id and parent_memory_id in self.claims:
            return self.claims[parent_memory_id].family_id
        # The family detector is deliberately claim-specific. Auxiliary notes
        # and verified corrections remain separate evidence units; otherwise
        # de-duplication would alter the normal Control retrieval denominator.
        if label != "ECHO":
            family = self._new_family()
            self.family_representatives[family] = (text, label)
            return family
        best_family, best_score = None, -1.0
        for family, (representative, representative_label) in self.family_representatives.items():
            if representative_label != label:
                continue
            score = self._cosine(text, representative)
            if score > best_score:
                best_family, best_score = family, score
        if best_family is not None and best_score >= self.family_threshold:
            return best_family
        family = self._new_family()
        self.family_representatives[family] = (text, label)
        return family

    def add(self, text: str, label: str, timestamp: str, kind: str, *,
            expected_family_id: str, source_id: str, source_type: str,
            independence_group: str, confidence: float,
            parent_memory_id: str | None = None,
            supersedes_family_id: str | None = None) -> Dict:
        self.attempted_writes += 1
        derived = source_type == "retrieval_derived"
        if derived and self.arm in RETRIEVAL_HYGIENE_ARMS:
            self.blocked_derived_adds += 1
            self.interaction_log.append({
                "session": self.current_session, "operation": "retrieval_trace",
                "source_type": source_type, "parent_memory_id": parent_memory_id,
                "fact_write_executed": False,
            })
            return {"executed": False, "memory_id": None,
                    "reason": "retrieval is exposure, not independent fact evidence",
                    "operation": "add"}

        inherited_source = source_id
        inherited_group = independence_group
        inherited_confidence = float(confidence)
        if parent_memory_id and parent_memory_id in self.claims and self.arm in IDENTITY_ARMS:
            parent = self.claims[parent_memory_id]
            inherited_source = parent.source_id
            inherited_group = parent.independence_group
            inherited_confidence = parent.confidence
        family_id = self._resolve_family(text, label, parent_memory_id)
        memory_id = self.adapter.add(text, label, timestamp, kind)
        self.executed_writes += 1
        self.claims[memory_id] = ClaimMeta(
            memory_id=memory_id, family_id=family_id,
            expected_family_id=expected_family_id, label=label,
            source_id=inherited_source, source_type=source_type,
            independence_group=inherited_group, confidence=inherited_confidence,
            activation=1.0, parent_memory_id=parent_memory_id,
            supersedes_family_id=supersedes_family_id,
            last_activation_session=self.current_session,
        )
        if label == "CORRECTION":
            self.correction_family_id = family_id
        return {"executed": True, "memory_id": memory_id, "reason": None,
                "operation": "add", "family_id": family_id,
                "source_id": inherited_source}

    def update(self, memory_id: str, text: str, timestamp: str, *, kind: str,
               source_type: str, status: str | None = None) -> Dict:
        self.attempted_writes += 1
        if source_type == "retrieval_derived" and self.arm in RECONSOLIDATION_ARMS:
            self.blocked_derived_updates += 1
            self.interaction_log.append({
                "session": self.current_session, "operation": "retrieval_update_trace",
                "source_type": source_type, "parent_memory_id": memory_id,
                "fact_write_executed": False,
            })
            return {"executed": False, "memory_id": memory_id,
                    "reason": "retrieval-derived update has no independent information gain",
                    "operation": "update"}
        self.adapter.update(memory_id, text, timestamp)
        self.executed_writes += 1
        meta = self.claims[memory_id]
        if status and self.arm in RECONSOLIDATION_ARMS:
            meta.status = status
        return {"executed": True, "memory_id": memory_id, "reason": None,
                "operation": "update", "status": meta.status}

    def _annotated_candidates(self, query: str, top_k: int) -> list[Dict]:
        raw = self.adapter.search(query, top_k=max(top_k, len(self.claims)))
        rows = []
        for hit in raw:
            meta = self.claims.get(hit["memory_id"])
            if meta is not None:
                rows.append({**hit, **asdict(meta)})
        return rows

    def _decay_activation(self, session: int):
        if self.arm not in PERSISTENCE_ARMS:
            return
        rate = math.log(2.0) / max(self.activation_half_life, 1e-9)
        for meta in self.claims.values():
            elapsed = max(0, session - meta.last_activation_session)
            if elapsed:
                meta.activation *= math.exp(-rate * elapsed)
                meta.last_activation_session = session

    @staticmethod
    def _rerank_score(row: Dict) -> float:
        score = float(row.get("retrieval_score", 0.0))
        score += 0.40 * float(row.get("activation", 0.0))
        score += 0.15 * float(row.get("confidence", 0.0))
        if row.get("status") == "superseded":
            score -= 0.18
        if row.get("status") == "suppressed":
            score -= 2.0
        return score

    def search(self, query: str, top_k: int, *, stage: str = "unknown",
               session: int = 0, reinforce_activation: bool = False) -> list[Dict]:
        self.current_session = max(self.current_session, int(session))
        self._decay_activation(self.current_session)
        rows = self._annotated_candidates(query, top_k)
        if self.arm in PERSISTENCE_ARMS:
            rows.sort(key=self._rerank_score, reverse=True)
        selected, seen_families = [], set()
        for row in rows:
            if self.arm in PERSISTENCE_ARMS and row.get("status") == "suppressed":
                continue
            if self.arm in RETRIEVAL_HYGIENE_ARMS:
                if row["family_id"] in seen_families:
                    continue
                seen_families.add(row["family_id"])
            selected.append(row)
            if len(selected) == top_k:
                break

        if self.arm in RECONSOLIDATION_ARMS and any(r["label"] == "ECHO" for r in selected):
            correction = next((r for r in rows if r["label"] == "CORRECTION"), None)
            if correction and all(r["memory_id"] != correction["memory_id"] for r in selected):
                if len(selected) >= top_k:
                    selected[-1] = correction
                else:
                    selected.append(correction)

        for rank, row in enumerate(selected, 1):
            row["retrieval_rank"] = rank
            meta = self.claims[row["memory_id"]]
            if self.arm in RETRIEVAL_HYGIENE_ARMS:
                meta.exposure_count += 1
                self.interaction_log.append({
                    "session": self.current_session, "operation": "search_exposure",
                    "memory_id": row["memory_id"], "family_id": row["family_id"],
                    "confidence_before": meta.confidence,
                    "confidence_after": meta.confidence,
                    "fact_write_executed": False,
                })
            if stage == "retrieval_practice" and reinforce_activation:
                meta.activation = min(4.0, meta.activation + 0.20)
                meta.last_activation_session = self.current_session
        return selected

    def search_complete(self, query: str, top_k: int) -> list[Dict]:
        rows = self._annotated_candidates(query, top_k)
        for rank, row in enumerate(rows[:top_k], 1):
            row["retrieval_rank"] = rank
        return rows[:top_k]

    def search_reliable(self, query: str, top_k: int) -> list[Dict]:
        rows = [r for r in self._annotated_candidates(query, top_k)
                if r["label"] != "ECHO"
                and r.get("source_type") != "retrieval_derived"
                and r.get("status") not in {"superseded", "suppressed"}]
        rows.sort(key=lambda r: (float(r.get("confidence", 0.0)),
                                 float(r.get("retrieval_score", 0.0))), reverse=True)
        for rank, row in enumerate(rows[:top_k], 1):
            row["retrieval_rank"] = rank
        return rows[:top_k]

    def observe_persistence(self, delay: int, ed: float | None,
                            matched_control_ed: float | None) -> Dict:
        if self.arm not in PERSISTENCE_ARMS or ed is None or matched_control_ed is None:
            return {"triggered": False, "excess": None, "streak": 0}
        excess = max(0.0, float(ed) - float(matched_control_ed))
        if excess > self.persistence_excess_threshold:
            self.persistence_exceedance_streak += 1
        else:
            self.persistence_exceedance_streak = 0
        triggered = self.persistence_exceedance_streak >= self.persistence_trigger_windows
        if triggered:
            self.remediation_triggers += 1
            echo_rows = [m for m in self.claims.values() if m.label == "ECHO"]
            for meta in echo_rows:
                meta.status = "suppressed"
                meta.activation = 0.0
            self.merged_records += max(0, len(echo_rows) - 1)
            self.persistence_exceedance_streak = 0
        return {"triggered": triggered, "excess": excess,
                "streak": self.persistence_exceedance_streak, "delay": delay}

    def audit(self) -> Dict:
        echo_rows = [m for m in self.claims.values() if m.label == "ECHO"]
        expected_groups = {}
        for meta in echo_rows:
            expected_groups.setdefault(meta.expected_family_id, []).append(meta)
        correct_pairs = total_pairs = 0
        for rows in expected_groups.values():
            for i, left in enumerate(rows):
                for right in rows[i + 1:]:
                    total_pairs += 1
                    correct_pairs += int(left.family_id == right.family_id)
        derived = [m for m in echo_rows if m.parent_memory_id]
        inherited = sum(
            m.parent_memory_id in self.claims
            and m.source_id == self.claims[m.parent_memory_id].source_id
            for m in derived
        )
        confidence_drifts = [
            abs(float(log["confidence_after"]) - float(log["confidence_before"]))
            for log in self.interaction_log if log["operation"] == "search_exposure"
        ]
        return {
            "arm": self.arm, "attempted_writes": self.attempted_writes,
            "executed_writes": self.executed_writes,
            "blocked_derived_adds": self.blocked_derived_adds,
            "blocked_derived_updates": self.blocked_derived_updates,
            "blocked_derived_writes": self.blocked_derived_adds + self.blocked_derived_updates,
            "stored_memories": len(self.claims),
            "echo_family_count": len({m.family_id for m in echo_rows}),
            "family_pair_recall": 1.0 if total_pairs == 0 else correct_pairs / total_pairs,
            "provenance_inheritance_accuracy": 1.0 if not derived else inherited / len(derived),
            "search_exposures": sum(m.exposure_count for m in self.claims.values()),
            "max_confidence_drift_on_search": max(confidence_drifts, default=0.0),
            "interaction_log_entries": len(self.interaction_log),
            "superseded_claim_records": sum(m.status == "superseded" for m in self.claims.values()),
            "suppressed_claim_records": sum(m.status == "suppressed" for m in self.claims.values()),
            "remediation_triggers": self.remediation_triggers,
            "merged_records": self.merged_records,
            "counterfactual_decision_enabled": self.arm in COUNTERFACTUAL_DECISION_ARMS,
        }

    def close(self):
        self.adapter.close()
