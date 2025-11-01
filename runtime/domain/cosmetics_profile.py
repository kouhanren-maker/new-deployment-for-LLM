# agent/runtime/domain/cosmetics_profile.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional
from .profiles import DomainProfile, register_profile

class CosmeticsProfile:
    """
    Beauty / Cosmetics domain:
    - Query multi-variant (synonyms × unit variants)
    - Size parsing (mL / fl oz) with normalization
    - Bundle/sample filtering（放到结果阶段，而非查询负词）
    - Price parsing (no installment logic)
    - De-dup by (provider + brand + canonical_name + size_bucket)
    """
    name = "cosmetics"

    # strong domain cues
    CATEGORIES = [
        "serum","essence","ampoule","toner","lotion","cream","booster",
        "moisturizer","sunscreen","cleanser","mask"
    ]
    ACTIVES = [
        "vitamin c","ascorbic","ascorbyl","tetrahexyldecyl","map","sap",
        "niacinamide","retinol","ferulic","laa","thd","bha","aha"
    ]

    # likely unwanted forms unless fallback enables（用于结果阶段过滤）
    BUNDLE_NEG = [
        "sample","trial","mini","travel","kit","set","bundle","refill",
        "sachet","discovery","value set","gift set"
    ]

    BRAND_HINTS = [
        "the ordinary","timeless","skinceuticals","cerave","la roche-posay",
        "paula's choice","innisfree","cosrx","kiehl","vichy","aveeno",
        "olay","neutrogena","biologique recherche","estee lauder","lancome",
        "clarins","clinique","shiseido","l'oreal","loreal","maybelline",
        "tatcha","fresh","drunk elephant","bioderma","benton","isntree",
        "medik8","obagi","it cosmetics","first aid beauty","glow recipe"
    ]

    @staticmethod
    def _to_float(x) -> Optional[float]:
        if x is None: return None
        if isinstance(x, (int, float)): return float(x)
        if isinstance(x, str):
            s = x.replace(",", "")
            m = re.search(r"[-+]?\d*\.?\d+", s)
            return float(m.group()) if m else None
        return None

    # ---------- auto score ----------
    @staticmethod
    def auto_score(text: str) -> Tuple[float, Dict[str, Any]]:
        t = (text or "").lower()
        score, ev = 0.0, {"hits": []}
        if any(k in t for k in CosmeticsProfile.CATEGORIES):
            score += 0.5; ev["hits"].append("category")
        if any(k in t for k in CosmeticsProfile.ACTIVES):
            score += 0.4; ev["hits"].append("active")
        if re.search(r"\b(\d+)\s*(ml|mL|fl\s*oz|fluid\s*ounce)\b", t):
            score += 0.2; ev["hits"].append("size")
        return (score if score >= 0.8 else 0.0), ev

    # ---------- preprocess (multi-variant to expand recall) ----------
    @staticmethod
    def preprocess_queries(text: str, prefs: Dict[str, Any]) -> List[str]:
        # 1) 去掉“price”等噪声词
        base = re.sub(r"\b(price|prices|cost|buy)\b", " ", text, flags=re.IGNORECASE)
        base = re.sub(r"\s+", " ", base).strip()

        # 2) 解析容量 → 生成 mL 与 fl oz 两种表达（都保留）
        tgt_ml = CosmeticsProfile._parse_size_ml(base)
        ml_phrase = None
        oz_phrase = None
        if tgt_ml:
            ml_phrase = f"{int(round(tgt_ml))} mL"
            oz_val = round(tgt_ml / 29.5735, 1)
            # 常见两种写法：'1 oz' / '1 fl oz' 都能命中
            oz_phrase = f"{oz_val} oz"

        # 3) 形态同义词，生成多版本查询
        forms = ["serum", "essence", "ampoule"]

        queries = []
        for f in forms:
            q = re.sub(r"\b(serum|essence|ampoule)\b", f, base, flags=re.IGNORECASE)
            if ml_phrase:
                queries.append(f"{q} {ml_phrase}")
            if oz_phrase:
                queries.append(f"{q} {oz_phrase}")
            queries.append(q)  # 无容量版本

        # 去重并限制最多 6 个，避免过多请求
        seen, deduped = set(), []
        for q in queries:
            k = q.lower().strip()
            if k not in seen:
                seen.add(k); deduped.append(q)
            if len(deduped) >= 6:
                break
        return deduped

    # ---------- entity extract ----------
    @staticmethod
    def _oz_to_ml(oz: float) -> float:
        return round(float(oz) * 29.5735, 2)

    @staticmethod
    def _parse_size_ml(s: str) -> Optional[float]:
        if not s: return None
        s = s.replace("\u2009"," ").replace("\xa0"," ")
        # 30 mL / 30ml
        m = re.search(r"(\d+(?:\.\d+)?)\s*mL\b", s, flags=re.IGNORECASE)
        if m: return float(m.group(1))
        # 1 fl oz / 1 oz
        m = re.search(r"(\d+(?:\.\d+)?)\s*(fl\.?\s*oz|fluid\s*ounce|oz)\b", s, flags=re.IGNORECASE)
        if m: return CosmeticsProfile._oz_to_ml(float(m.group(1)))
        return None

    @staticmethod
    def _brand_from_text(s: str) -> Optional[str]:
        t = (s or "").lower()
        for b in CosmeticsProfile.BRAND_HINTS:
            if b in t: return b
        # fallback: first token heuristic (Capitalized word at start)
        m = re.search(r"\b([A-Z][a-zA-Z]+)\b", s or "")
        return m.group(1).lower() if m else None

    @staticmethod
    def _canonical_name(title: str, brand: Optional[str]) -> str:
        t = (title or "").lower()
        if brand:
            t = re.sub(re.escape(brand), " ", t)
        # remove filler marketing words but keep active cues & percentages
        t = re.sub(r"\b(with|advanced|intense|ultimate|classic|original|new|latest)\b", " ", t)
        t = re.sub(r"[^a-z0-9%.\s]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        return t

    @staticmethod
    def entity_extract(text: str) -> Dict[str, Any]:
        target_ml = CosmeticsProfile._parse_size_ml(text)
        want_serum = ("serum" in (text or "").lower())
        return {"target_ml": target_ml, "want_serum": want_serum}

    # ---------- model filter (category & size window) ----------
    @staticmethod
    def filter_model(item: Dict[str, Any], entities: Dict[str, Any], *, strict: bool) -> bool:
        title = (str(item.get("title") or "") + " " + str(item.get("subtitle") or "")).lower()

        # category: 如果查询偏向 serum，则优先 serum/essence/ampoule
        if entities.get("want_serum"):
            if not any(k in title for k in ["serum","essence","ampoule"]):
                return False

        # 套装/小样过滤（查询阶段不加负词，这里过滤）
        if any(k in title for k in CosmeticsProfile.BUNDLE_NEG):
            return False

        # 容量窗口（若目标容量存在）
        tgt = entities.get("target_ml")
        if tgt:
            size_ml = CosmeticsProfile._parse_size_ml(title) or CosmeticsProfile._parse_size_ml(str(item.get("description") or ""))
            if size_ml is None:
                # 新逻辑：即使严格模式，容量缺失也不淘汰，交给排序降权
                return True
            low, high = tgt * 0.7, tgt * 1.4
            if not (low <= size_ml <= high):
                return False

        return True

    # ---------- price normalize ----------
    @staticmethod
    def normalize_price(item: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
        # Cosmetics seldom have installments; treat /mo as noise → missing_price
        blob = " ".join([str(item.get("price_str") or ""), str(item.get("priceText") or ""), str(item.get("snippet") or "")]).lower()
        if re.search(r"(/mo|per\s*month|\b\d+\s*mo(nths)?\b)", blob, re.IGNORECASE):
            return False, None, "missing_price"

        v = item.get("price")
        p = CosmeticsProfile._to_float(v)
        if p is not None and p > 0:
            return True, p, None

        # try parse from strings
        text_price = str(item.get("price_str") or item.get("priceText") or "")
        p2 = CosmeticsProfile._to_float(text_price)
        if p2 is not None and p2 > 0:
            return True, p2, None

        return False, None, "missing_price"

    # ---------- accessory filter ----------
    @staticmethod
    def keep_after_accessory(item: Dict[str, Any], *, is_accessory_intent: bool) -> bool:
        # Cosmetics: drop only tools/organizers unless accessory intent
        if is_accessory_intent:
            return True
        t = " ".join([str(item.get("title") or ""), str(item.get("category") or "")]).lower()
        tool_neg = ["brush","applicator","organizer","bag","pouch","spatula","mixing bowl"]
        return not any(k in t for k in tool_neg)

    # ---------- dedup key ----------
    @staticmethod
    def dedup_key(title: str, item: Dict[str, Any]) -> str:
        provider = str(item.get("provider") or item.get("source") or "").lower()
        brand = CosmeticsProfile._brand_from_text(title) or CosmeticsProfile._brand_from_text(str(item.get("brand") or ""))
        canon = CosmeticsProfile._canonical_name(title, brand)
        # size bucket (round to nearest 5 mL)
        size_ml = CosmeticsProfile._parse_size_ml(title) or CosmeticsProfile._parse_size_ml(str(item.get("description") or ""))
        bucket = None
        if size_ml is not None:
            bucket = int(round(size_ml / 5.0) * 5)
        return f"{provider}::{brand or 'na'}::{canon[:80]}::{bucket or 'size-na'}"

    # ---------- fallbacks ----------
    @staticmethod
    def fallback_plan() -> List[str]:
        return ["second_pass_query", "expand_size_window", "allow_missing_price_as_null_price"]

register_profile(CosmeticsProfile())
