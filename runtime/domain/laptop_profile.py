# agent/runtime/domain/laptop_profile.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional

from .profiles import DomainProfile, register_profile

class LaptopProfile:
    name = "electronics_laptop"

    BRANDS = [
        "macbook", "thinkpad", "ideapad", "yoga", "legion",
        "latitude", "xps", "inspiron", "precision",
        "elitebook", "spectre", "envy", "omen", "pavilion",
        "surface", "zenbook", "vivobook", "rog", "nitro", "predator",
        "msi", "acer", "asus", "lenovo", "dell", "hp", "huawei", "xiaomi"
    ]

    CPU_CUES = [r"i[3579]-?\d{3,5}u?", r"ryzen\s?[3579]\s?\d{3,5}", r"apple\s?(m1|m2|m3)\w*"]

    @staticmethod
    def _to_float(x, default=0.0) -> float:
        try:
            return float(x if x is not None else default)
        except Exception:
            return default

    # ---------- auto score ----------
    @staticmethod
    def auto_score(text: str) -> Tuple[float, Dict[str, Any]]:
        t = (text or "").lower()
        score, ev = 0.0, {"hits": []}
        if any(b in t for b in LaptopProfile.BRANDS):
            score += 0.6; ev["hits"].append("brand")
        if any(re.search(p, t) for p in LaptopProfile.CPU_CUES):
            score += 0.3; ev["hits"].append("cpu")
        if "laptop" in t or "notebook" in t or "ultrabook" in t:
            score += 0.2; ev["hits"].append("category")
        return (score if score >= 0.8 else 0.0), ev

    # ---------- preprocess ----------
    @staticmethod
    def preprocess_queries(text: str, prefs: Dict[str, Any]) -> List[str]:
        # 笔记本暂不加负关键词；必要时可加入 -refurbished/-used 等
        return [text]

    # ---------- entities ----------
    @staticmethod
    def entity_extract(text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        brand = None
        for b in LaptopProfile.BRANDS:
            if b in t:
                brand = b; break
        # 型号粗提（如 x1 carbon / xps 13 / macbook air 15 等）
        model = None
        m = re.search(r"(x1\s*carbon|xps\s*\d+|macbook\s*(air|pro)\s*\d*|surface\s*(laptop|book)\s*\d*)", t)
        if m: model = m.group(0)
        return {"brand": brand, "model": model}

    # ---------- model filter ----------
    @staticmethod
    def filter_model(item: Dict[str, Any], entities: Dict[str, Any], *, strict: bool) -> bool:
        title = (str(item.get("title") or "") + " " + str(item.get("subtitle") or "")).lower()
        brand = entities.get("brand")
        model = entities.get("model")
        if brand and brand not in title:
            return False
        if strict and model:
            return model in title
        return True

    # ---------- price normalize ----------
    @staticmethod
    def normalize_price(item: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
        try:
            p = float(item.get("price"))
            # 简单屏蔽过低价格（< 100 可能是配件/分期，后续可扩展合约识别）
            if 0 < p < 100:
                return False, None, "suspicious_low"
            return True, p, None
        except Exception:
            return False, None, "missing_price"

    # ---------- accessory filter ----------
    @staticmethod
    def keep_after_accessory(item: Dict[str, Any], *, is_accessory_intent: bool) -> bool:
        if is_accessory_intent:
            return True
        t = " ".join([str(item.get("title") or ""), str(item.get("category") or "")]).lower()
        # 过滤常见配件
        NEG = ["sleeve","bag","backpack","dock","docking","stand","cooler","keyboard","mouse","charger","adapter","hub","skin","sticker"]
        return not any(k in t for k in NEG)

    # ---------- dedup ----------
    @staticmethod
    def dedup_key(title: str, item: Dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", (title or "").lower()).strip()

    # ---------- fallbacks ----------
    @staticmethod
    def fallback_plan() -> List[str]:
        return ["relax_model_suffix", "allow_installment_only_as_null_price"]

register_profile(LaptopProfile())
