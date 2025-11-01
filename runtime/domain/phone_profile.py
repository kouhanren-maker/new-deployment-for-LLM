# runtime/domain/phone_profile.py
from __future__ import annotations
import re
import hashlib
from typing import Dict, Any, List, Tuple, Optional

from .profiles import register_profile, DomainProfile


class PhoneProfile(DomainProfile):
    """
    手机域 Profile：实体抽取、模型守门、价格口径、配件过滤、去重键、auto_score
    """
    name = "electronics_phone"

    # -------------------------
    # auto_score：用于 auto_detect 判域打分
    # -------------------------
    def auto_score(self, text: str):
        """
        返回 (score, evidence)
        命中 iPhone/Galaxy 及其代际/后缀/容量越多，得分越高。
        """
        t = (text or "").lower()
        score = 0.0
        hits = []

        # 品牌/家族
        if "iphone" in t:
            score += 0.6
            hits.append("family")
        if "galaxy" in t:
            score += 0.6
            hits.append("family")

        # 代际
        if re.search(r"\b(1[0-9])\b", t) or re.search(r"\bs\s*-?\s*\d+\b", t):
            score += 0.4
            hits.append("gen")

        # 后缀
        if re.search(r"\bpro\s*max\b", t) or "ultra" in t or re.search(r"\bpro\b", t):
            score += 0.3
            hits.append("suffix")

        # 容量
        if re.search(r"\b(64|128|256|512|1024)\s*gb\b", t):
            score += 0.1
            hits.append("capacity")

        # 无锁/运营商词
        if re.search(r"\bunlocked\b|\bsim\s*free\b", t):
            score += 0.2
            hits.append("carrier")

        # 价格意图词
        if "price" in t:
            score += 0.2
            hits.append("price_intent")

        # 上限 & 证据
        score = min(score, 1.5)
        ev = {"hits": list(set(hits))}
        return score, ev

    # -------------------------
    # 查询预处理（两轮：先窄后宽）
    # -------------------------
    def preprocess_queries(self, text: str, prefs: Dict[str, Any]) -> List[str]:
        NEG = (
            " -case -cover -magsafe -screen protector -tempered glass -charger -cable "
            "-adapter -dock -stand -holder -skin -sticker -band -strap"
        )
        t = text.strip()
        low = t.lower()
        # 补品牌，避免只写 iPhone/Galaxy 时召回到配件页
        if "iphone" in low and "apple" not in low:
            t = f"apple {t}"
        if "galaxy" in low and "samsung" not in low:
            t = f"samsung {t}"
        # 第一轮：强负词；第二轮：弱负词；第三轮：原文
        return [f"{t}{NEG}", f"{t} -case -cover -screen protector", t]

    # -------------------------
    # 实体抽取
    # -------------------------
    def entity_extract(self, text: str) -> Dict[str, Any]:
        t = (text or "").lower()
        ent: Dict[str, Any] = {}

        ent["brand"] = "apple" if "iphone" in t else ("samsung" if "galaxy" in t else None)
        ent["family"] = "iphone" if "iphone" in t else ("galaxy" if "galaxy" in t else None)

        # iPhone 15 / 16
        m = re.search(r"\b(1[0-9])\b", t)
        if m and ent["family"] == "iphone":
            ent["gen"] = m.group(1)

        # Galaxy S24 -> 24
        m2 = re.search(r"\bs\s*-?\s*([0-9]{1,2})\b", t)
        if m2 and ent["family"] == "galaxy":
            ent["gen"] = m2.group(1)

        if re.search(r"\bpro\s*max\b", t):
            ent["suffix"] = "pro max"
        elif re.search(r"\bpro\b", t):
            ent["suffix"] = "pro"
        elif re.search(r"\bultra\b", t):
            ent["suffix"] = "ultra"

        cap = None
        m3 = re.search(r"\b(64|128|256|512|1024)\s*gb\b", t)
        if m3:
            cap = int(m3.group(1))
        ent["capacity"] = cap

        ent["carrier"] = ("unlocked" if re.search(r"\bunlocked\b|\bsim\s*free\b", t) else None)
        return ent

    # -------------------------
    # 型号守门
    # -------------------------
    def filter_model(self, d: Dict[str, Any], entities: Dict[str, Any], strict: bool = True) -> bool:
        title = f"{d.get('title') or d.get('name') or ''} {d.get('subtitle') or ''}".lower()

        fam = entities.get("family")
        gen = entities.get("gen")
        suf = entities.get("suffix")

        if fam == "iphone":
            if "iphone" not in title:
                return False
            if gen and not re.search(rf"\biphone\s*{re.escape(gen)}\b", title):
                return False
            if strict and suf:
                if suf == "pro max" and "pro max" not in title:
                    return False
                if suf == "pro" and "pro" not in title:
                    return False

        if fam == "galaxy":
            if "galaxy" not in title or "s" not in title:
                return False
            if gen and not re.search(rf"\bs\s*-?\s*{re.escape(gen)}\b", title):
                return False
            if strict and suf:
                if suf == "ultra" and "ultra" not in title:
                    return False

        return True

    # -------------------------
    # 价格口径（成色/分期过滤 + 正价解析）
    # 返回：(ok, price, reason)
    # -------------------------
    def normalize_price(self, d: Dict[str, Any]) -> Tuple[bool, Optional[float], str]:
        blob = " ".join([
            str(d.get("price_str") or ""),
            str(d.get("priceText") or ""),
            str(d.get("snippet") or ""),
            str(d.get("title") or "")
        ]).lower()

        # 1) 运营商分期
        if re.search(r"(/mo|per\s*month|\b\d+\s*mo(nths)?\b|\$\s*[\d,.]+\s*/\s*mo)", blob):
            m1 = re.search(r"\$?\s*([\d,.]+)\s*(/|per)?\s*mo", blob)
            m2 = re.search(r"for\s*(\d+)\s*mo(nths)?", blob)
            if m1 and m2:
                monthly = float(m1.group(1).replace(",", ""))
                months = int(m2.group(1))
                return True, round(monthly * months, 2), "total_from_installment"
            return False, None, "installment_only"

        # 2) 二手/翻新/开箱（强过滤）
        NEG_COND = ("renewed", "refurb", "pre-owned", "preowned", "used", "open box",
                    "seller refurbished", "good condition", "fair condition")
        if any(k in blob for k in NEG_COND):
            return False, None, "condition_bad"

        # 3) 标价解析
        if d.get("price") is not None:
            try:
                return True, float(d.get("price")), "priced"
            except Exception:
                pass

        m = re.search(r"\$\s*([\d,.]+)", blob)
        if m:
            try:
                return True, float(m.group(1).replace(",", "")), "priced_from_blob"
            except Exception:
                pass

        return False, None, "missing_price"

    # -------------------------
    # 配件过滤
    # -------------------------
    def keep_after_accessory(self, d: Dict[str, Any], is_accessory_intent: bool = False) -> bool:
        if is_accessory_intent:
            return True
        t = " ".join([str(d.get("title") or ""), str(d.get("category") or ""), str(d.get("product_type") or "")]).lower()
        NEG = ("case", "cover", "magsafe", "screen protector", "tempered", "glass",
               "charger", "cable", "adapter", "dock", "stand", "holder", "skin", "sticker", "band", "strap", "watch")
        return not any(k in t for k in NEG)

    # -------------------------
    # 去重键（引入 productid/offer_docid；回退 url_hash）
    # -------------------------
    def parse_docids(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        # 形如 prds=productid:XXX,headlineOfferDocid:YYY,imageDocid:...
        m = re.search(r"prds=([^&]+)", url)
        if not m:
            return None, None
        blob = m.group(1)
        pid = None
        offer = None
        m1 = re.search(r"productid%3A([^%,]+)", blob) or re.search(r"productid:([^%,]+)", blob)
        if m1:
            pid = m1.group(1)
        m2 = re.search(r"headlineOfferDocid%3A([^%,]+)", blob) or re.search(r"headlineOfferDocid:([^%,]+)", blob)
        if m2:
            offer = m2.group(1)
        return pid, offer

    def dedup_key(self, title: str, d: Dict[str, Any], entities: Dict[str, Any]) -> str:
        url = str(d.get("url") or "")
        pid, offer = self.parse_docids(url)
        # 规格归一
        brand = "apple" if "iphone" in title.lower() else ("samsung" if "galaxy" in title.lower() else entities.get("brand") or "")
        fam = entities.get("family") or ""
        gen = entities.get("gen") or ""
        suf = entities.get("suffix") or ""
        cap = str(entities.get("capacity") or "")
        carrier = "unlocked" if re.search(r"\bunlocked\b|\bsim\s*free\b", title.lower()) else (entities.get("carrier") or "")
        vendor_sig = offer or pid or hashlib.md5(url.encode("utf-8")).hexdigest()[:10]
        return "|".join([brand, fam, str(gen), suf, cap, carrier, vendor_sig])

    # -------------------------
    # Fallback 计划
    # -------------------------
    def fallback_plan(self) -> List[str]:
        return [
            "relax_model_suffix",
            "softscore_relax_suffix",
            "allow_installment_only_as_null_price",
            "allow_missing_price_as_null_price",
            # 需要时可再扩展：比如 "unlocked_probe"
        ]

    # -------------------------
    # 软评分（宽松回补排序权重）
    # -------------------------
    def soft_score(self, d: Dict[str, Any], entities: Dict[str, Any], has_total_price: bool) -> float:
        score = 0.0
        title = f"{d.get('title') or ''} {d.get('subtitle') or ''}".lower()
        if entities.get("suffix") == "pro max" and "pro max" in title:
            score += 1.5
        if entities.get("suffix") == "ultra" and "ultra" in title:
            score += 1.5
        if entities.get("capacity") and str(entities["capacity"]) in title:
            score += 0.8
        if "unlocked" in title or "sim free" in title:
            score += 0.8
        if has_total_price:
            score += 0.5
        return score


# ✅ 注册（只传实例一个参数）
register_profile(PhoneProfile())
