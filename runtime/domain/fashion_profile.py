# agent/runtime/domain/fashion_profile.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional
from .profiles import DomainProfile, register_profile

class FashionProfile:
    name = "fashion"

    BRANDS = ["nike","adidas","uniqlo","zara","hm","h&m","levi","gap","puma","new balance","under armour"]

    @staticmethod
    def _to_float(x, default=0.0) -> float:
        try:
            return float(x if x is not None else default)
        except Exception:
            return default

    @staticmethod
    def auto_score(text: str) -> Tuple[float, Dict[str, Any]]:
        t = (text or "").lower()
        score, ev = 0.0, {"hits": []}
        if any(b in t for b in FashionProfile.BRANDS):
            score += 0.5; ev["hits"].append("brand")
        if any(k in t for k in ["t-shirt","shirt","hoodie","jacket","jeans","pants","sneaker","dress","skirt","coat","outerwear","sweater","cardigan","polo","tee"]):
            score += 0.4; ev["hits"].append("category")
        if any(s in t for s in ["size","xs","s ","m ","l ","xl","xxl","us ","eu "]):
            score += 0.2; ev["hits"].append("size")
        return (score if score >= 0.8 else 0.0), ev

    @staticmethod
    def preprocess_queries(text: str, prefs: Dict[str, Any]) -> List[str]:
        # 不做负关键词；可按需加 -kids 等
        return [text]

    @staticmethod
    def entity_extract(text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        brand = None
        for b in FashionProfile.BRANDS:
            if b in t:
                brand = b; break
        category = None
        for c in ["t-shirt","shirt","hoodie","jacket","jeans","pants","sneaker","dress","skirt","coat","outerwear","sweater","cardigan","polo","tee"]:
            if c in t:
                category = c; break
        size = None
        m = re.search(r"\b(xs|s|m|l|xl|xxl)\b", t)
        if m: size = m.group(1)
        return {"brand": brand, "category": category, "size": size}

    @staticmethod
    def filter_model(item: Dict[str, Any], entities: Dict[str, Any], *, strict: bool) -> bool:
        title = (str(item.get("title") or "") + " " + str(item.get("subtitle") or "")).lower()
        brand = entities.get("brand")
        category = entities.get("category")
        if brand and brand not in title:
            return False
        if strict and category and category not in title:
            return False
        return True

    @staticmethod
    def normalize_price(item: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
        try:
            p = float(item.get("price"))
            return True, p, None
        except Exception:
            return False, None, "missing_price"

    @staticmethod
    def keep_after_accessory(item: Dict[str, Any], *, is_accessory_intent: bool) -> bool:
        # Fashion 默认不过滤配件（包/皮带/帽子等），除非你希望“只服饰不配件”
        return True

    @staticmethod
    def dedup_key(title: str, item: Dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", (title or "").lower()).strip()

    @staticmethod
    def fallback_plan() -> List[str]:
        return ["relax_model_suffix"]

register_profile(FashionProfile())
