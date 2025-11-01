# runtime/search/serp.py
import os
import requests
from typing import List, Dict, Any

SERPAPI_KEY = os.getenv("SERPAPI_KEY", "")

def google_shopping(query: str, location: str = "Australia", gl: str = "us", hl: str = "en") -> Dict[str, Any]:
    """
    Query SerpAPI Google Shopping. Returns parsed JSON dict.
    """
    url = "https://serpapi.com/search.json"
    params = {
        "engine": "google_shopping",
        "q": query,
        "location": location,
        "gl": gl,
        "hl": hl,
        "api_key": SERPAPI_KEY,
        "device": "desktop",
        "num": 40,
        "safe": "active",
        "udm": "28",
    }
    r = requests.get(url, params=params, timeout=25)
    r.raise_for_status()
    return r.json()

def map_items(serp: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Map SerpAPI shopping_results to a normalized list.
    """
    out = []
    for it in serp.get("shopping_results", [])[:40]:
        title = it.get("title") or ""
        link  = it.get("product_link") or it.get("link") or ""
        price = it.get("extracted_price")
        currency = None
        # Serp 没直接给 currency；简单从字符串 price 里猜（足够用于过滤）
        price_str = it.get("price") or ""
        if "AED" in price_str: currency = "AED"
        elif "$" in price_str: currency = "USD"  # 你的 gl=us + google.com 大多是 $
        provider = (it.get("source") or "").lower()
        install = it.get("installment", {})
        out.append({
            "title": title,
            "url": link,
            "price": price,
            "currency": currency,
            "provider": provider,
            "raw": it,
            "installment_only": bool(install and not price),  # 极端兜底
        })
    return out
