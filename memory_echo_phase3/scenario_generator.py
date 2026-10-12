from dataclasses import asdict, dataclass
from typing import Dict, List


PRODUCTS = [
    ("复古茶具", "景德镇古韵陶瓷", "新瓷源工贸", "雅茗器皿", "陶然供应链"),
    ("人体工学办公椅", "恒坐家具", "优脊办公", "广联座椅", "森木办公"),
    ("不锈钢轴承", "久航工业", "精轴科技", "华北机件", "锐虎机械"),
    ("纯棉卫衣批发", "潮源服饰", "棉尚制衣", "东港服装", "衣联商贸"),
    ("可降解快递袋", "绿包材料", "新环包装", "速达包材", "宏泰塑业"),
    ("酒店一次性用品", "悦客日化", "净旅用品", "宾至百货", "舒洁实业"),
    ("LED工矿灯", "明拓照明", "星曜光电", "亮能灯具", "华辉电器"),
    ("宠物冻干食品", "宠鲜生物", "原野宠粮", "萌伴食品", "爱宠供应链"),
    ("儿童积木玩具", "童创玩具", "智趣工厂", "乐搭塑胶", "启星文创"),
    ("商用咖啡豆", "醇香贸易", "山屿咖啡", "南岸食品", "豆客供应链"),
    ("铝合金门窗型材", "坚铝建材", "轻固金属", "华创建材", "鼎盛铝业"),
    ("餐厅陶瓷餐盘", "白瓷工坊", "新釉陶瓷", "佳宴器皿", "瓷海商贸"),
    ("户外露营帐篷", "野途户外", "轻峰装备", "拓营用品", "山客工贸"),
    ("实验室丁腈手套", "安研防护", "净盾医疗", "科卫耗材", "康护实业"),
    ("家用收纳箱", "整家塑业", "空间优品", "好纳家居", "洁居百货"),
    ("汽车雨刮器", "驰安汽配", "清途汽配", "车联配件", "路虎机电"),
    ("婚庆仿真花", "花语工艺", "新绽装饰", "美境花艺", "喜尚用品"),
    ("防水标签纸", "恒印纸业", "锐彩标签", "标识工坊", "佳印包装"),
    ("直播补光灯", "映亮器材", "光域科技", "主播装备", "视界电子"),
    ("厨房硅胶用品", "厨悦硅胶", "耐热生活", "美厨塑胶", "佳品日用"),
]


@dataclass
class Supplier:
    supplier_id: str
    name: str
    price_index: float
    quality: float
    rating: float
    role: str


def generate_scenarios(max_samples: int = 10) -> List[Dict]:
    if not 1 <= max_samples <= 20:
        raise ValueError("max_samples must be between 1 and 20")
    scenarios = []
    for i, (product, a, b, c, d) in enumerate(PRODUCTS[:max_samples]):
        suppliers = [
            Supplier("A", a, 1.00, 4.1, 4.2, "historical"),
            Supplier("B", b, 0.82, 4.8, 4.9, "new_best"),
            Supplier("C", c, 0.91, 3.8, 4.0, "distractor"),
            Supplier("D", d, 1.08, 4.4, 4.1, "distractor"),
        ]
        scenarios.append({
            "sample_id": i,
            "user_requirement": f"寻找可靠的{product}供应商，兼顾价格、质量和评分。",
            "product": product,
            "supplier_candidates": [asdict(x) for x in suppliers],
            "ground_truth_supplier": "B",
        })
    return scenarios

