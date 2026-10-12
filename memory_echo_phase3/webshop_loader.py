"""Load real WebShop products and crowd-sourced shopping instructions."""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _price(value: str):
    nums = re.findall(r"\d+(?:\.\d+)?", value or "")
    return float(nums[0]) if nums else None


def _rating(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _text(p: Dict) -> str:
    desc = p.get("small_description") or p.get("full_description") or ""
    if isinstance(desc, list):
        desc = " ".join(str(x) for x in desc)
    return " ".join(str(x) for x in (p.get("name", ""), p.get("query", ""),
                                      p.get("product_category", ""), desc))


def load_webshop_scenarios(data_dir: Path, max_samples: int = 10) -> tuple[List[Dict], Dict]:
    if max_samples < 10:
        raise ValueError("WebShop benchmark requires max_samples >= 10")
    if (data_dir / "items_shuffle_10000.json").exists():
        product_path = data_dir / "items_shuffle_10000.json"
        attrs_path = data_dir / "items_ins_v2_10000.json"
        goals_path = data_dir / "items_human_ins_10000.json"
        subset_name = "WebShop 10K product subset"
    else:
        product_path = data_dir / "items_shuffle_1000.json"
        attrs_path = data_dir / "items_ins_v2_1000.json"
        goals_path = data_dir / "items_human_ins.json"
        subset_name = "WebShop official small package"
    for path in (product_path, attrs_path, goals_path):
        if not path.exists():
            raise FileNotFoundError(f"Missing official WebShop file: {path}")
    products = json.loads(product_path.read_text(encoding="utf-8"))
    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    goals = json.loads(goals_path.read_text(encoding="utf-8"))
    by_asin = {p["asin"]: p for p in products}
    eligible = [(asin, rows[0]) for asin, rows in goals.items() if asin in by_asin and rows]
    if len(eligible) < max_samples:
        raise ValueError(f"Only {len(eligible)} goal products overlap the small catalog")

    corpus = [_text(p) for p in products]
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20000)
    item_vectors = vectorizer.fit_transform(corpus)
    scenarios = []
    for sample_id, (goal_asin, goal) in enumerate(eligible[:max_samples]):
        instruction = goal["instruction"].strip()
        query_vec = vectorizer.transform([instruction])
        sims = cosine_similarity(query_vec, item_vectors)[0]
        ranked = sorted(range(len(products)), key=lambda i: (-sims[i], products[i]["asin"]))
        alternatives = [products[i] for i in ranked if products[i]["asin"] != goal_asin][:3]
        selected = [alternatives[0], by_asin[goal_asin], alternatives[1], alternatives[2]]
        candidates = []
        for role, label, p in zip(("historical", "ground_truth", "distractor", "distractor"), "ABCD", selected):
            candidates.append({
                "supplier_id": label,
                "product_asin": p["asin"],
                "name": p.get("name", ""),
                "price": _price(p.get("pricing", "")),
                "price_raw": p.get("pricing", ""),
                "rating": _rating(p.get("average_rating")),
                "reviews": p.get("total_reviews", ""),
                "query_category": p.get("query", ""),
                "product_category": p.get("product_category", ""),
                "attributes": attrs.get(p["asin"], {}).get("attributes", []),
                "retrieval_similarity": float(sims[by_asin_index(products, p["asin"])]),
                "role": role,
            })
        scenarios.append({
            "sample_id": sample_id,
            "dataset": f"Princeton WebShop (OpenDataLab catalog entry / {subset_name})",
            "instruction_id": goal.get("assignment_id", f"{goal_asin}:0"),
            "user_requirement": instruction,
            "product": by_asin[goal_asin].get("query", "product"),
            "instruction_attributes": goal.get("instruction_attributes", []),
            "instruction_options": goal.get("instruction_options", []),
            "supplier_candidates": candidates,
            "ground_truth_supplier": "B",
            "ground_truth_asin": goal_asin,
        })
    manifest = {
        "source_catalog": "https://opendatalab.org.cn/OpenDataLab/WebShop",
        "official_source": "https://github.com/princeton-nlp/WebShop",
        "loaded_subset": subset_name,
        "mirror_used_for_official_small_package": "https://huggingface.co/datasets/zhangdw/webshop/raw/webshop-small.tar.gz",
        "catalog_products": len(products), "human_instruction_products": len(goals),
        "overlapping_goal_products": len(eligible), "selected_samples": len(scenarios),
        "unique_dataset_qa": len(scenarios), "qa_turns_per_sample": 100,
        "sha256": {p.name: _sha256(p) for p in (product_path, attrs_path, goals_path)},
    }
    return scenarios, manifest


def by_asin_index(products: List[Dict], asin: str) -> int:
    # Catalog is only 1k rows; explicit lookup keeps selection deterministic.
    return next(i for i, p in enumerate(products) if p["asin"] == asin)
