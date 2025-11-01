# agent/runtime/domain/books_profile.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional
from .profiles import DomainProfile, register_profile

class BooksProfile:
    name = "books"

    @staticmethod
    def auto_score(text: str) -> Tuple[float, Dict[str, Any]]:
        t = (text or "").lower()
        score, ev = 0.0, {"hits": []}
        if any(k in t for k in ["book","novel","paperback","hardcover","ebook","isbn","author"]):
            score += 0.8; ev["hits"].append("bookish")
        return (score if score >= 0.8 else 0.0), ev

    @staticmethod
    def preprocess_queries(text: str, prefs: Dict[str, Any]) -> List[str]:
        return [text]

    @staticmethod
    def entity_extract(text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        isbn = None
        m = re.search(r"\b97[89][- ]?\d{1,5}[- ]?\d{1,7}[- ]?\d{1,7}[- ]?[\dxX]\b", t)
        if m: isbn = m.group(0)
        fmt = None
        for f in ["paperback","hardcover","ebook","audiobook"]:
            if f in t: fmt = f; break
        return {"isbn": isbn, "format": fmt}

    @staticmethod
    def filter_model(item: Dict[str, Any], entities: Dict[str, Any], *, strict: bool) -> bool:
        # 若有 ISBN，要求命中；否则放行
        title = (str(item.get("title") or "") + " " + str(item.get("subtitle") or "") + " " + str(item.get("description") or "")).lower()
        isbn = entities.get("isbn")
        if isbn:
            return isbn.replace("-", "").replace(" ", "") in title.replace("-", "").replace(" ", "")
        return True

    @staticmethod
    def normalize_price(item: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
        try:
            return True, float(item.get("price")), None
        except Exception:
            return False, None, "missing_price"

    @staticmethod
    def keep_after_accessory(item: Dict[str, Any], *, is_accessory_intent: bool) -> bool:
        # 过滤周边（海报/贴纸），除非是配件意图
        if is_accessory_intent:
            return True
        t = " ".join([str(item.get("title") or ""), str(item.get("category") or "")]).lower()
        return not any(k in t for k in ["poster","sticker","bookmark","book light","cover"])

    @staticmethod
    def dedup_key(title: str, item: Dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", (title or "").lower()).strip()

    @staticmethod
    def fallback_plan() -> List[str]:
        return ["relax_model_suffix"]

register_profile(BooksProfile())
