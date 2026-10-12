def select_cases(results, count=5):
    candidates = []
    for sample in results:
        for run in sample["runs"]:
            if run["condition"] == "historical_repetition_update":
                m = run["metrics"]
                candidates.append((m["echo_density"] + m["supplier_dominance_score"], sample, run))
    cases = []
    for _, sample, run in sorted(candidates, reverse=True, key=lambda x: x[0])[:count]:
        final = run["memory_trajectory"][-1]
        cases.append({
            "sample_id": sample["sample_id"], "framework": run["framework"],
            "user_requirement": sample["user_requirement"],
            "memory_evolution": [{"turn": t["turn"], "echo_density": t["metrics"]["echo_density"]}
                                 for t in run["memory_trajectory"]],
            "final_retrieved_memories": final["retrieved_memories"],
            "recommended_supplier": run["metrics"]["recommended_supplier"],
            "ground_truth_supplier": sample["ground_truth_supplier"],
            "explanation": "历史 Supplier A 的重复检索写回占据 top-k，并通过 update 强化主记录。"
                           if run["metrics"]["recommended_supplier"] == "A" else
                           "存在历史回声，但新 Supplier B 最终仍获得更高检索权重。",
        })
    return cases

