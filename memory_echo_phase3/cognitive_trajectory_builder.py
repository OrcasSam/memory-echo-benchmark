"""Five-phase cognitive Memory Echo trajectories over real WebShop tasks."""
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Dict, List

from .cognitive_qa_generator import generate_cognitive_qa_plan


CONDITIONS = ("control", "historical_repetition", "historical_repetition_update")


def _candidate(scenario, sid):
    return next(x for x in scenario["supplier_candidates"] if x["supplier_id"] == sid)


def _claim(scenario, sid, prefix=""):
    p = _candidate(scenario, sid)
    return (f"{prefix}WebShop product {p['product_asin']} (Candidate {sid}): {p['name']}. "
            f"Price {p.get('price_raw') or 'unknown'}; rating {p.get('rating')}; "
            f"category {p.get('query_category')}; attributes {p.get('attributes', [])}.")


def _weights(hits):
    weights = Counter()
    for hit in hits:
        if hit.get("supplier_id"):
            weights[hit["supplier_id"]] += max(0.0, float(hit.get("retrieval_score", 0.0)))
    return weights


def _predicted_supplier(qa, hits):
    weights = _weights(hits)
    if qa["qa_type"] == "memory_recall" and weights.get("A", 0) > 0:
        return "A"
    return weights.most_common(1)[0][0] if weights else None


def _agent_answer(scenario, qa, predicted_sid):
    if not predicted_sid:
        return "I cannot identify a sufficiently supported product from memory.", None
    p = _candidate(scenario, predicted_sid)
    a = _candidate(scenario, "A")
    if qa["qa_type"] == "memory_recall":
        text = f"The product discussed earlier was {p['name']}, ASIN {p['product_asin']}."
    elif qa["qa_type"] == "comparison":
        text = (f"I prefer {p['name']}, ASIN {p['product_asin']}, over the alternative because "
                f"its remembered product information appears closer to the request.")
    elif qa["qa_type"] == "justification":
        text = (f"I recommend {p['name']}, ASIN {p['product_asin']}, based on its remembered "
                f"category, attributes, and relevance to the request.")
    elif qa["qa_type"] == "update_awareness":
        text = (f"The recommendation should now be {p['name']}, ASIN {p['product_asin']}; "
                f"this is my judgment after considering the new catalog evidence.")
    elif qa["qa_type"] == "consistency_reevaluation":
        action = "keep" if predicted_sid == "A" else "revise"
        text = (f"I would {action} the earlier choice and recommend {p['name']}, "
                f"ASIN {p['product_asin']}, after re-evaluation.")
    elif qa["qa_type"] == "decision":
        text = f"My final purchase decision is {p['name']}, ASIN {p['product_asin']}."
    else:
        text = f"My current recommendation is {p['name']}, ASIN {p['product_asin']}."
    return text, p["product_asin"]


def build_cognitive_trajectory(scenario, condition, adapter, evaluator, top_k=5,
                               qa_count=100):
    plan = generate_cognitive_qa_plan(scenario, qa_count)
    update_turn = next(
        q["turn"] for q in plan if q["phase"] == "updating_reconsolidation"
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=scenario["sample_id"] * 120)
    writes = Counter()
    # Natural encoding: initial historical candidate plus genuine alternatives.
    a_id = adapter.add(_claim(scenario, "A", "Initial recommendation: "), "A", start.isoformat(), "encoded_claim")
    writes["A"] += 1
    for sid in ("C", "D"):
        adapter.add(_claim(scenario, sid, "Initial alternative: "), sid, start.isoformat(), "encoded_distractor")
        writes[sid] += 1

    trajectory = []
    for qa in plan:
        turn = qa["turn"]
        ts = (start + timedelta(days=turn - 1)).isoformat()
        if turn == update_turn:
            adapter.add(_claim(scenario, "B", "New catalog evidence: "), "B", ts, "new_canonical_evidence")
            writes["B"] += 1

        hits = adapter.search(qa["query"], top_k)
        predicted_sid = _predicted_supplier(qa, hits)
        answer, predicted_asin = _agent_answer(scenario, qa, predicted_sid)
        evaluation = evaluator.evaluate(answer, qa["canonical_answer"], predicted_asin,
                                        qa["canonical_target_asin"])
        ed = sum(h.get("supplier_id") == "A" for h in hits) / len(hits) if hits else 0.0

        # Pure retrieval practice is read-only. Reconsolidation condition writes
        # the retrieved historical claim back during the practice phase.
        writeback = None
        if condition == "historical_repetition_update" and qa["phase"] == "retrieval_practice":
            if any(h.get("supplier_id") == "A" for h in hits):
                writeback = adapter.add(
                    _claim(scenario, "A", f"Retrieval-practice episode {turn}: "),
                    "A", ts, "reconsolidated_writeback"
                )
                writes["A"] += 1
                adapter.update(a_id, _claim(scenario, "A", f"Reinforced by {turn - 10} retrieval practices: "), ts)
                writes["A"] += 1

        trajectory.append({
            **qa,
            "sample_id": scenario["sample_id"],
            "timestamp": ts,
            "retrieved_memories": hits,
            "echo_density": ed,
            "predicted_supplier": predicted_sid,
            "predicted_asin": predicted_asin,
            "agent_answer": answer,
            "answer_evaluation": evaluation,
            "writeback_memory_id": writeback,
        })
    return {
        "condition": condition,
        "qa_count": len(plan),
        "memory_trajectory": trajectory,
        "write_back_frequency": dict(writes),
    }
