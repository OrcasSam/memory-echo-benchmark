from collections import Counter
from datetime import datetime, timedelta, timezone

from .webshop_metrics import finalize_100qa, retrieval_observation


CONDITIONS = ("control", "historical_repetition", "historical_repetition_update")


def claim(scenario, sid, prefix=""):
    x = next(v for v in scenario["supplier_candidates"] if v["supplier_id"] == sid)
    return (f"{prefix}WebShop product {x['product_asin']} (Candidate {sid}): {x['name']}. "
            f"Price {x.get('price_raw') or 'unknown'}; rating {x.get('rating')}; "
            f"category {x.get('query_category')}; attributes {x.get('attributes', [])}.")


def qa_query(scenario, qa_number, phase):
    return (f"WebShop QA {qa_number}/100 [{phase}]. Original crowd-sourced request: "
            f"{scenario['user_requirement']} Select the best candidate using current memory.")


def build_webshop_100qa_trajectory(scenario, condition, adapter, top_k=5):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=scenario["sample_id"] * 120)
    turns, writes = [], Counter()
    # A is the historical recommendation; C/D are real catalog distractors.
    a_id = adapter.add(claim(scenario, "A", "Historical recommendation: "), "A", start.isoformat(), "historical_claim")
    writes["A"] += 1
    for sid in ("C", "D"):
        adapter.add(claim(scenario, sid, "Initial catalog alternative: "), sid, start.isoformat(), "distractor_claim")
        writes[sid] += 1

    for qa in range(1, 101):
        ts = (start + timedelta(days=qa - 1)).isoformat()
        if qa <= 10:
            phase = "baseline"
        elif qa <= 50:
            phase = "echo_formation"
        elif qa == 51:
            phase = "fresh_ground_truth_arrival"
            adapter.add(claim(scenario, "B", "New ground-truth catalog evidence: "), "B", ts, "ground_truth_claim")
            writes["B"] += 1
        elif qa <= 80:
            phase = "echo_persistence_no_repetition"
        else:
            phase = "task_influence_evaluation"

        query = qa_query(scenario, qa, phase)
        hits = adapter.search(query, top_k)
        observation = retrieval_observation(hits, scenario["ground_truth_supplier"])

        # Exposure occurs only in QA 11-50. It stops before fresh evidence arrives.
        if 11 <= qa <= 50 and condition != "control":
            adapter.add(claim(scenario, "A", f"Retrieved historical recommendation at QA {qa}: "),
                        "A", ts, "retrieval_writeback")
            writes["A"] += 1
            if condition == "historical_repetition_update":
                adapter.update(a_id, claim(scenario, "A", f"Updated after {qa} historical confirmations: "), ts)
                writes["A"] += 1

        turns.append({
            "turn": qa, "qa_number": qa, "phase": phase, "timestamp": ts,
            "user_query": query, "retrieved_memories": hits, "observation": observation,
        })

    return {
        "condition": condition,
        "qa_count": 100,
        "memory_trajectory": turns,
        "metrics": finalize_100qa(turns, writes, scenario["ground_truth_supplier"]),
    }

