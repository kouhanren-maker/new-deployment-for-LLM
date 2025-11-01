# orchestrator.py
from typing import List, Dict, Tuple
from models import CompareQuery, CompareResult, PriceItem

# 简单汇率表（需要可接真实 FX）
DEFAULT_FX: Dict[Tuple[str, str], float] = {
    ("AUD", "AUD"): 1.0,
    ("USD", "AUD"): 1.48,
    ("EUR", "AUD"): 1.62,
}

def normalize_currency(items: List[PriceItem], fx: Dict[Tuple[str,str], float], target: str) -> List[PriceItem]:
    out: List[PriceItem] = []
    for it in items:
        if it.currency != target:
            rate = fx.get((it.currency, target))
            if rate:
                it = it.copy(update={
                    "price": round(it.price * rate, 2),
                    "shipping_cost": round(it.shipping_cost * rate, 2),
                    "tax_cost": round(it.tax_cost * rate, 2),
                    "currency": target
                })
        out.append(it)
    return out

def canonical_key(it: PriceItem) -> str:
    # 可升级为：型号+容量+颜色+向量相似度
    return f"{(it.brand or '').lower()}|{(it.model or '').lower()}|{(it.variant or '').lower()}|{it.condition.lower()}"

def deduplicate(items: List[PriceItem]) -> List[PriceItem]:
    bag: Dict[str, PriceItem] = {}
    for it in items:
        k = canonical_key(it)
        keep = it
        if k in bag:
            keep = it if it.total_cost < bag[k].total_cost else bag[k]
        bag[k] = keep
    return list(bag.values())

def policy_filter(items: List[PriceItem]) -> List[PriceItem]:
    ans: List[PriceItem] = []
    for it in items:
        if it.total_cost <= 0:
            continue
        if it.seller_rating is not None and it.seller_rating < 3.0:
            continue
        ans.append(it)
    return ans

def sort_items(items: List[PriceItem]) -> List[PriceItem]:
    # 主：总拥有成本；次：评分高优先
    return sorted(items, key=lambda x: (x.total_cost, -(x.seller_rating or 0.0)))

class PriceCompareOrchestrator:
    def __init__(self, providers: List, fx_rates: Dict[Tuple[str,str], float] = None):
        self.providers = providers
        self.fx = fx_rates or DEFAULT_FX

    async def run(self, q: CompareQuery) -> CompareResult:
        # 1) 并发检索
        import asyncio
        tasks = [p.search(q, q.prefs.get("max_results", 20)) for p in self.providers]
        results_nested = await asyncio.gather(*tasks)
        items = [it for sub in results_nested for it in sub]
        print(f"[ORC] fetched {len(items)} raw items from providers")

        # 2) 币种归一化
        items = normalize_currency(items, self.fx, q.currency)

        # 3) 去重
        before = len(items)
        items = deduplicate(items)
        deduped = before - len(items)

        # 4) 策略过滤
        before2 = len(items)
        items = policy_filter(items)
        filtered = before2 - len(items)

        # 5) 排序
        items = sort_items(items)

        # 6) 偏好过滤（如只要全新）
        if q.prefs.get("only_new"):
            items = [i for i in items if i.condition.lower() == "new"]

        # 7) TopN
        topn = int(q.prefs.get("max_results", 10))
        items = items[:topn]

        return CompareResult(items=items, deduped=deduped, filtered=filtered)
