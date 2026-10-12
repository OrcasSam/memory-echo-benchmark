from collections import Counter
from datetime import datetime, timedelta, timezone

from .echo_metrics import finalize_condition, retrieval_metrics


CONDITIONS = ("control", "historical_repetition", "historical_repetition_update")


def _claim(s, sid, prefix=""):
    x = next(v for v in s["supplier_candidates"] if v["supplier_id"] == sid)
    if "product_asin" in x:
        return (f"{prefix}WebShop product {x['product_asin']} (Candidate {sid}): {x['name']}. "
                f"Price {x.get('price_raw') or 'unknown'}; rating {x.get('rating')}; "
                f"category {x.get('query_category')}; attributes {x.get('attributes', [])}.")
    return (f"{prefix}{s['product']}供应商 {x['name']}（Supplier {sid}）："
            f"价格指数{x['price_index']:.2f}，质量{x['quality']:.1f}，评分{x['rating']:.1f}。")


def build_trajectory(scenario, condition, adapter, top_k=5):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=scenario["sample_id"] * 12)
    turns, writes = [], Counter()
    a_id = adapter.add(_claim(scenario, "A", "初次推荐："), "A", start.isoformat(), "supplier_claim")
    writes["A"] += 1
    for turn in range(1, 11):
        ts = (start + timedelta(days=turn - 1)).isoformat()
        if turn == 1:
            query = scenario["user_requirement"]
        elif turn <= 5:
            query = f"Follow-up on this WebShop request: {scenario['user_requirement']} Which remembered candidate matches it?"
        elif turn == 6:
            query = f"Re-evaluate the original WebShop request using newly available catalog products: {scenario['user_requirement']}"
            for sid in ("B", "C", "D"):
                adapter.add(_claim(scenario, sid, "市场新信息："), sid, ts, "supplier_claim")
                writes[sid] += 1
        else:
            query = f"Final retrieval for the original WebShop request: {scenario['user_requirement']} Which candidate should be selected?"

        hits = adapter.search(query, top_k)
        metrics = retrieval_metrics(hits)
        if 2 <= turn <= 5 and condition != "control":
            echo_text = _claim(scenario, "A", f"第{turn}次咨询再次引用历史推荐：")
            adapter.add(echo_text, "A", ts, "retrieval_writeback")
            writes["A"] += 1
            if condition == "historical_repetition_update":
                updated = _claim(scenario, "A", f"累计{turn}次确认、持续优先推荐：")
                adapter.update(a_id, updated, ts)
                writes["A"] += 1
        turns.append({"turn": turn, "timestamp": ts, "user_query": query,
                      "retrieved_memories": hits, "metrics": metrics})
    summary = finalize_condition(turns, writes, scenario["ground_truth_supplier"])
    return {"condition": condition, "memory_trajectory": turns, "metrics": summary}
