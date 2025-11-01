# providers/google_shopping.py
import os, httpx, re
from typing import List, Tuple
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit, quote
from dotenv import load_dotenv
from models import CompareQuery, PriceItem
from typing import Optional

load_dotenv()
SERPAPI_KEY = os.getenv("SERPAPI_KEY")

CURRENCY_MAP = {
    "A$":"AUD","AU$":"AUD","$":"USD","US$":"USD","€":"EUR","£":"GBP","¥":"JPY"
}
PRICE_RE = re.compile(r"(\d+(?:\.\d+)?)")

def safe_url(u: str) -> str:
    """把包含空格等非法字符的 URL 规范化，避免 Pydantic HttpUrl 校验报错"""
    if not u:
        return "https://www.google.com/shopping"
    p = urlsplit(u)
    # 只对 path 和 query 做编码，保留主机名等
    path = quote(p.path) if p.path else ""
    query = "&".join([f"{quote(k)}={quote(v)}" for k,v in
                      [q.split("=",1) if "=" in q else (q,"") for q in p.query.split("&") if q]]) if p.query else ""
    return urlunsplit((p.scheme or "https", p.netloc or "www.google.com", path, query, ""))

def parse_price(price_raw, fallback_currency: str) -> Optional[Tuple[float, str]]:
    """兼容 extracted_price(float)、'A$1,299'、'From $899'、'$1,099 – $1,299' 等"""
    if price_raw is None:
        return None
    if isinstance(price_raw, (int, float)):
        return float(price_raw), fallback_currency
    s = str(price_raw).replace(",", "").strip()
    cur = fallback_currency
    for sym, code in CURRENCY_MAP.items():
        if sym in s:
            cur = code
            s = s.replace(sym, "")
    parts = re.split(r"(?:–|-|to|from|From)", s)
    nums = []
    for p in parts:
        m = PRICE_RE.search(p)
        if m:
            try: nums.append(float(m.group(1)))
            except: pass
    if not nums:
        m = PRICE_RE.search(s)
        if not m: return None
        nums = [float(m.group(1))]
    return min(nums), cur

class GoogleShoppingProvider:
    name = "google_shopping"

    async def search(self, q: CompareQuery, limit: int = 12) -> List[PriceItem]:
        if not SERPAPI_KEY:
            raise RuntimeError("SERPAPI_KEY not set. Put it in .env")

        params = {
            "engine": "google_shopping",
            "q": q.text,
            "location": "Australia" if q.region == "AU" else "United States",
            "hl": "en",
            "api_key": SERPAPI_KEY
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=6.0)) as client:
            r = await client.get("https://serpapi.com/search.json", params=params)
            r.raise_for_status()
            data = r.json()

        results = data.get("shopping_results") or []
        print(f"[SERPAPI] query='{q.text}', got {len(results)} results, error={data.get('error')}")

        items: List[PriceItem] = []
        for it in results[:limit]:
            try:
                title = it.get("title") or q.text
                # Google 有时没有商家直链（link），只有 product_link（带空格）：
                link = it.get("link") or it.get("product_link") or "https://www.google.com/shopping"
                link = safe_url(link)

                source = it.get("source") or it.get("merchant") or "unknown"

                # 价格优先用 extracted_price，否则解析 price 字符串
                price_raw = it.get("extracted_price", it.get("price"))
                parsed = parse_price(price_raw, q.currency)
                if not parsed:
                    # 再兜底：个别在 'prices' 列表
                    for p in (it.get("prices") or []):
                        parsed = parse_price(p.get("extracted_price", p.get("price")), q.currency)
                        if parsed: break
                if not parsed:
                    # 打印一条诊断但继续处理后续条目
                    print(f"[SERPAPI] skip (no price): {title}")
                    continue

                price_val, currency = parsed

                items.append(PriceItem(
                    title=title,
                    brand=None, model=None, variant=None,
                    price=price_val, currency=currency,
                    shipping_cost=0.0, tax_cost=0.0,   # 如需更准，可对 TopN 做二次补抓
                    seller=source, seller_rating=None,
                    condition=(it.get("condition") or it.get("second_hand_condition") or "unknown").lower(),
                    url=link, source=self.name, updated_at=datetime.utcnow()
                ))
            except Exception as e:
                # 单条异常不影响全局；打印诊断即可
                print(f"[SERPAPI] item error: {e}")

        print(f"[SERPAPI] mapped {len(items)} items")
        return items
