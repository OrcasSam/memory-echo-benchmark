from collections import Counter


def retrieval_metrics(hits):
    if not hits:
        return {"echo_density": 0.0, "supplier_dominance_score": 0.0,
                "retrieval_frequency": {}, "recommended_supplier": None}
    old = [h for h in hits if h["supplier_id"] == "A"]
    weights = Counter()
    for h in hits:
        if h["supplier_id"]:
            weights[h["supplier_id"]] += max(0.0, h["retrieval_score"])
    total = sum(weights.values())
    recommended = weights.most_common(1)[0][0] if weights else None
    return {
        "echo_density": len(old) / len(hits),
        "supplier_dominance_score": weights["A"] / total if total else 0.0,
        "retrieval_frequency": dict(Counter(h["supplier_id"] for h in hits if h["supplier_id"])),
        "recommended_supplier": recommended,
    }


def finalize_condition(turns, write_counts, ground_truth="B"):
    before = turns[4]["metrics"]["echo_density"]
    after_values = [x["metrics"]["echo_density"] for x in turns[6:10]]
    after = sum(after_values) / len(after_values)
    final = turns[-1]["metrics"]
    return {
        "echo_density": after,
        "supplier_dominance_score": final["supplier_dominance_score"],
        # Echo-specific resistance: the obsolete historical supplier wins even
        # though B is objectively best. Distractor errors are not Memory Echo.
        "freshness_resistance_score": float(final["recommended_supplier"] == "A"),
        "echo_persistence_score": after / before if before > 0 else 0.0,
        "recommended_supplier": final["recommended_supplier"],
        "retrieval_frequency": dict(sum((Counter(t["metrics"]["retrieval_frequency"]) for t in turns), Counter())),
        "write_back_frequency": dict(write_counts),
    }
