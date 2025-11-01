# agent/runtime/intent_decider.py
from __future__ import annotations
import re
from typing import Dict, Any, Tuple

# -------- Accessory detection (English) --------
ACCESSORY_KEYWORDS = [
    "case", "cover", "magsafe", "screen protector", "tempered glass",
    "charger", "cable", "adapter", "dock", "stand", "holder",
    "skin", "sticker", "film", "band", "strap"
]

def looks_like_accessory_query(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in ACCESSORY_KEYWORDS)

# -------- Very-light domain detector (English) --------
def auto_detect_domain(text: str) -> Tuple[str, float, Dict[str, Any]]:
    """
    Returns: (domain_name, score, evidence)
      - "electronics_phone" if phone cues are strong, else "generic"
    """
    t = (text or "").lower()
    score = 0.0
    ev = {"hits": []}

    brands = [
        "iphone", "galaxy", "pixel", "oneplus", "xiaomi", "redmi",
        "huawei", "mate", "poco", "oppo", "vivo", "nothing phone"
    ]
    if any(b in t for b in brands):
        score += 0.7; ev["hits"].append("brand")

    if re.search(r"\biphone\s*1[0-9]\b", t):
        score += 0.5; ev["hits"].append("iphone_gen")

    if "galaxy" in t and re.search(r"\bs\s?([2-9][0-9])\b", t):
        score += 0.4; ev["hits"].append("galaxy_sxx")

    if re.search(r"\bpixel\s?[4-9]\b", t):
        score += 0.4; ev["hits"].append("pixel_x")

    if re.search(r"\boneplus\s?[4-9]\b", t):
        score += 0.3; ev["hits"].append("oneplus_x")

    if ("pro" in t) or ("ultra" in t):
        score += 0.2; ev["hits"].append("suffix")

    domain = "electronics_phone" if score >= 0.8 else "generic"
    return domain, score, ev

# -------- Intent scoring (price vs recommend) --------
PRICE_KEYWORDS = [
    # English
    "price", "cheapest", "deal", "where to buy", "under $", "buy", "link",
    "price compare", "compare price", "price comparison", "compare prices",
    # Chinese
    "比价", "价格对比", "价格比较", "哪里买", "在哪里买", "购买链接", "便宜", "折扣"
]
RECO_KEYWORDS = [
    "recommend", "best ", "which", "versus", " vs ", "for "
]

def decide_intent(
    text: str,
    prefs: Dict[str, Any] | None = None,
    history: Any | None = None,
    executor: Any | None = None
) -> Tuple[str, float, Dict[str, Any]]:
    """
    Returns: (planner_intent, confidence, evidence)
      planner_intent: "price" | "recommend"
    """
    t = (text or "").lower()

    price_score = 0.0
    for k in PRICE_KEYWORDS:
        if k in t:
            price_score += 0.3

    reco_score = 0.0
    for k in RECO_KEYWORDS:
        if k in t:
            reco_score += 0.25

    # If user mentions compare + price explicitly, strongly lean price
    if ("compare" in t and "price" in t) or ("比较" in t and "价格" in t):
        price_score += 0.4

    # Currency symbol or explicit currency words can indicate pricing intent
    if re.search(r"(\$|€|£|¥|aud|usd|eur|rmb|cny)\s*\d", t):
        price_score += 0.2

    # Specificity boosts
    if re.search(r"\b(iphone|galaxy|pixel|oneplus)\b", t) and re.search(r"\b(1[0-9]|s[2-9][0-9]|[4-9])\b", t):
        price_score += 0.3  # concrete model → price leaning

    # Accessory flag & domain
    is_acc = looks_like_accessory_query(text)
    domain, d_score, d_ev = auto_detect_domain(text)

    evidence: Dict[str, Any] = {
        "price_score": round(price_score, 2),
        "reco_score": round(reco_score, 2),
        "flags": {"is_accessory": is_acc},
        "domain_probe": {"domain": domain, "score": d_score, "evidence": d_ev},
        "decision": None
    }

    if price_score >= reco_score + 0.2:
        evidence["decision"] = "rule_price"
        return "price", min(1.0, price_score), evidence
    if reco_score >= price_score + 0.2:
        evidence["decision"] = "rule_reco"
        return "recommend", min(1.0, reco_score), evidence

    # Revised tie handling: prefer price when price cues present
    has_model = bool(re.search(r"\b(iphone|galaxy|pixel|oneplus)\b", t))
    has_price_cue = (
        any(k in t for k in ["price", "比价", "价格", "价格对比", "价格比较"]) or
        bool(re.search(r"(\$|€|£|¥)\s*\d", t))
    )
    if has_price_cue or has_model:
        evidence["decision"] = "tie_default_price"
        return "price", 0.55, evidence

    # tie → default to price if a concrete model is present, else recommend
    has_model = bool(re.search(r"\b(iphone|galaxy|pixel|oneplus)\b", t))
    if has_model:
        evidence["decision"] = "tie_default_price"
        return "price", 0.55, evidence
    else:
        evidence["decision"] = "tie_default_reco"
        return "recommend", 0.55, evidence
