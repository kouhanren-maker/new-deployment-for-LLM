# agent/tools_impl.py
from typing import List, Dict, Any, Optional
import re

from runtime.tool_schemas import (
    register_tool,
    # full pipeline
    CompareFullInput, CompareFullOutput, CompareItem,
    # multi-step pipeline
    PriceSearchInput, PriceSearchOutput, PriceItem as PriceSearchItem,
    NormalizeFxTaxInput, NormalizeFxTaxOutput,
    MergeRankInput, MergeRankOutput,
    # recommend
    RecommendInput, RecommendOutput, RecommendItem,
)

from orchestrator import PriceCompareOrchestrator
from providers.google_shopping import GoogleShoppingProvider
from models import CompareQuery, CompareResult
from recommender.recommend_agent import generate_recommendations

# Profile Registry
from runtime.domain.profiles import get_profile, auto_detect
from runtime.domain import phone_profile as _load_phone_profile  # noqa: F401
from runtime.domain import generic_profile as _load_generic_profile  # noqa: F401
from runtime.domain import laptop_profile as _load_laptop_profile  # noqa: F401
from runtime.domain import fashion_profile as _load_fashion_profile  # noqa: F401
from runtime.domain import books_profile as _load_books_profile  # noqa: F401
from runtime.domain import cosmetics_profile as _load_cosmetics_profile  # noqa: F401

register_tool("price.compare_full", CompareFullInput, CompareFullOutput, "Full price comparison via orchestrator")
register_tool("price.search",       PriceSearchInput,  PriceSearchOutput, "Search raw price items from providers")
register_tool("normalize.fx_tax",   NormalizeFxTaxInput, NormalizeFxTaxOutput, "Normalize currency/tax for items")
register_tool("merge.rank",         MergeRankInput,    MergeRankOutput,  "Merge & rank items with dedup")
register_tool("reco.generate",      RecommendInput,    RecommendOutput,  "Generate recommendations")

_providers = [GoogleShoppingProvider()]
_orc = PriceCompareOrchestrator(providers=_providers)

def _to_float(x, default=0.0) -> float:
    try:
        return float(x if x is not None else default)
    except Exception:
        return default

def _first_nonempty(d: Dict[str, Any], *keys) -> Optional[Any]:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None

# ---------------------------------------------------------
# 价格比较（含 Debug 与更强去重）
# ---------------------------------------------------------
async def price_compare_full(inp: CompareFullInput, ctx: Dict[str, Any]) -> CompareFullOutput:
    MIN_RESULTS = 12

    prefs = dict(inp.prefs or {})
    DEBUG = bool(prefs.get("debug"))
    def dbg(*args):
        if DEBUG:
            print("[COMPARE-DEBUG]", *args)

    # 选择/自动判域
    domain_name = str(prefs.get("domain") or "").strip().lower()
    prof = None if domain_name in ("", "generic", "auto") else get_profile(domain_name)
    auto_ev = {}; auto_score = 0.0
    if prof is None:
        prof, auto_score, auto_ev = auto_detect(inp.text)
        domain_name = prof.name
        prefs["domain"] = domain_name

    is_accessory_intent = bool(prefs.get("is_accessory") or False)
    queries = prof.preprocess_queries(inp.text, prefs)
    dbg("queries =", queries)

    # 抓取
    all_raw, round_sizes = [], []
    async def run_query(q_text: str):
        q = CompareQuery(text=q_text, region=inp.region, currency=inp.currency, prefs=prefs)
        res: CompareResult = await _orc.run(q)
        round_sizes.append(len(res.items))
        all_raw.extend(res.items)
        dbg(f"query='{q_text[:80]}...' -> got {len(res.items)}")

    for q_text in queries:
        await run_query(q_text)
        if len(all_raw) >= MIN_RESULTS:
            break

    diag: Dict[str, Any] = {
        "domain": domain_name, "raw": len(all_raw), "round_sizes": round_sizes,
        "kept_after_model": None, "kept_after_pricing": None, "kept_after_accessory": None,
        "kept_after_dedup": None, "final": None,
        "reasons": {
            "model_mismatch": 0, "installment_only": 0, "accessory": 0,
            "missing_required": 0, "missing_price": 0, "condition_bad": 0
        },
        "fallbacks": [],
        "auto_detect": {"score": auto_score, "evidence": auto_ev} if auto_ev else {},
        "debug": {
            "queries": queries,
            "stage_records": {
                "model_drop": [], "pricing_drop": [], "accessory_drop": [], "dedup_drop": []
            },
            "kept_titles": {"A": [], "B": [], "C": [], "D": []},
            "dedup_keys": [],
            "parsed_ids": [],
            "condition_flags": []
        }
    }

    def rec_stage(bucket: str, title: str, d: Dict[str, Any], reason: str = ""):
        if not DEBUG: return
        diag["debug"]["stage_records"][bucket].append({
            "title": title[:160],
            "provider": str(d.get("provider") or d.get("source") or "unknown"),
            **({"reason": reason} if reason else {})
        })

    def rec_keep(stage: str, title: str, d: Dict[str, Any]):
        if not DEBUG: return
        diag["debug"]["kept_titles"][stage].append({
            "title": title[:160],
            "provider": str(d.get("provider") or d.get("source") or "unknown")
        })

    # 实体
    entities = prof.entity_extract(inp.text)
    diag["entity"] = entities
    dbg("entity =", entities, "| raw =", len(all_raw))

    # A：型号守门
    kept_a: List[tuple] = []
    for it in all_raw:
        d = it.model_dump() if hasattr(it, "model_dump") else it.dict()
        url = str(d.get("url") or "https://example.com/unknown")
        title = str(_first_nonempty(d, "title", "name") or "").strip()
        if not title or not url:
            diag["reasons"]["missing_required"] += 1
            rec_stage("model_drop", "(missing-title/url)", d, "missing_required")
            continue
        if not prof.filter_model(d, entities, strict=True):
            diag["reasons"]["model_mismatch"] += 1
            rec_stage("model_drop", title, d, "strict_model_mismatch")
            continue
        kept_a.append((d, title, url))
        rec_keep("A", title, d)

    # 仍不足 → 宽松守门
    if len(kept_a) < MIN_RESULTS and "relax_model_suffix" in prof.fallback_plan():
        diag["fallbacks"].append("relax_model_suffix")
        tmp = []
        for it in all_raw:
            d = it.model_dump() if hasattr(it, "model_dump") else it.dict()
            url = str(d.get("url") or "https://example.com/unknown")
            title = str(_first_nonempty(d, "title", "name") or "").strip()
            if not title or not url:
                continue
            if not prof.filter_model(d, entities, strict=False):
                rec_stage("model_drop", title, d, "relax_model_mismatch")
                continue
            tmp.append((d, title, url)); rec_keep("A", title, d)
        kept_a = tmp

    # 手机域：软评分补位
    if domain_name == "electronics_phone" and len(kept_a) < MIN_RESULTS and "softscore_relax_suffix" in prof.fallback_plan():
        diag["fallbacks"].append("softscore_relax_suffix")
        cands: List[tuple] = []
        for it in all_raw:
            d = it.model_dump() if hasattr(it, "model_dump") else it.dict()
            url = str(d.get("url") or "https://example.com/unknown")
            title = str(_first_nonempty(d, "title", "name") or "").strip()
            if not title or not url:
                continue
            t = (title + " " + str(d.get("subtitle") or "")).lower()
            fam = entities.get("family"); gen = entities.get("gen")
            fam_ok = (fam in t) if fam else True
            gen_ok = True
            if gen:
                if "iphone" in t:
                    gen_ok = bool(re.search(rf"\biphone\s*{re.escape(gen)}\b", t))
                elif "galaxy" in t:
                    gen_ok = bool(re.search(rf"\bs\s*-?\s*{re.escape(gen)}\b", t))
            if fam_ok and gen_ok:
                ok, price, reason = prof.normalize_price(d)
                score = prof.soft_score(d, entities, ok)
                cands.append((score, d, title, url, price if ok else None, reason if not ok else None))
        cands.sort(key=lambda x: x[0], reverse=True)
        seen_keys = {prof.dedup_key(t, d, entities) for d, t, _ in [(a[0], a[1], a[2]) for a in kept_a]} if kept_a else set()
        for sc, d, title, url, price, reason in cands:
            key = prof.dedup_key(title, d, entities)
            if key in seen_keys: continue
            kept_a.append((d, title, url)); seen_keys.add(key); rec_keep("A", title, d)
            if len(kept_a) >= MIN_RESULTS: break

    diag["kept_after_model"] = len(kept_a)
    dbg("A kept =", len(kept_a))

    # B：价格口径
    kept_b: List[tuple] = []
    installment_pool: List[tuple] = []
    missing_price_pool: List[tuple] = []
    for d, title, url in kept_a:
        ok, price, reason = prof.normalize_price(d)
        if not ok:
            if reason == "installment_only":
                diag["reasons"]["installment_only"] += 1; installment_pool.append((d, title, url, None))
                rec_stage("pricing_drop", title, d, "installment_only")
            elif reason == "condition_bad":
                diag["reasons"]["condition_bad"] += 1
                rec_stage("pricing_drop", title, d, "condition_bad")
            else:
                diag["reasons"]["missing_price"] += 1; missing_price_pool.append((d, title, url, None))
                rec_stage("pricing_drop", title, d, "missing_price")
            continue
        kept_b.append((d, title, url, price)); rec_keep("B", title, d)

        # Debug：记录成色/分期标志
        if DEBUG:
            diag["debug"]["condition_flags"].append({
                "title": title[:160],
                "is_installment": False,
                "is_refurb_or_used": False
            })

    diag["kept_after_pricing"] = len(kept_b)
    dbg("B kept =", len(kept_b))

    # C：配件过滤
    kept_c: List[tuple] = []
    for d, title, url, price in kept_b:
        if not prof.keep_after_accessory(d, is_accessory_intent=is_accessory_intent):
            diag["reasons"]["accessory"] += 1
            rec_stage("accessory_drop", title, d)
            continue
        kept_c.append((d, title, url, price)); rec_keep("C", title, d)

    diag["kept_after_accessory"] = len(kept_c)
    dbg("C kept =", len(kept_c))

    # D：去重（使用 phone_profile 的 dedup_key，并记录解析到的 docids）
    seen = set()
    items: List[CompareItem] = []
    for d, title, url, price in kept_c:
        key = prof.dedup_key(title, d, entities)
        if DEBUG:
            pid, offer = getattr(prof, "parse_docids")(str(d.get("url") or ""))
            diag["debug"]["parsed_ids"].append({"title": title[:160], "productid": pid, "offer_docid": offer})
            diag["debug"]["dedup_keys"].append({"title": title[:160], "key": key})
        if key in seen:
            rec_stage("dedup_drop", title, d); continue
        seen.add(key)
        items.append(CompareItem(
            title=title,
            url=url,
            currency=str(d.get("currency") or inp.currency or "AUD"),
            price=_to_float(price) if price is not None else 0.0,
            shipping=_to_float(d.get("shipping")),
            tax=_to_float(d.get("tax")),
            provider=str(d.get("provider") or d.get("source") or "unknown"),
        ))
        rec_keep("D", title, d)

    diag["kept_after_dedup"] = len(items)
    diag["final"] = len(items)
    dbg("D kept =", len(items))

    return CompareFullOutput(items=items, diagnostics=diag)

# ---------------------------------------------------------
# 多步：search / normalize / merge-rank（轻量实现）
# ---------------------------------------------------------
async def price_search(inp: PriceSearchInput, ctx: Dict[str, Any]) -> PriceSearchOutput:
    # 直接使用 orchestrator 抓取一轮，作为“搜索原始结果”近似
    q = CompareQuery(text=inp.query, region="AU", currency="AUD", prefs={"providers": inp.providers})
    res: CompareResult = await _orc.run(q)

    items: List[PriceSearchItem] = []
    for it in res.items:
        d = it.model_dump() if hasattr(it, "model_dump") else it.dict()
        items.append(PriceSearchItem(
            title=str(d.get("title") or "")[:200],
            url=str(d.get("url") or d.get("link") or "")[:1000],
            currency=str(d.get("currency") or "AUD"),
            price=_to_float(d.get("price")),
            shipping=_to_float(d.get("shipping_cost")),
            tax=_to_float(d.get("tax_cost")),
            provider=str(d.get("source") or d.get("provider") or "unknown")[:64],
        ))

    # 限制返回条数（近似 inp.limit）
    if len(items) > max(1, int(inp.limit)):
        items = items[: int(inp.limit)]
    return PriceSearchOutput(items=items)


def _ctx_latest_items(ctx: Dict[str, Any]) -> List[dict]:
    keys = [k for k in ctx.keys() if k.startswith("step_") and k.endswith("_output")]
    if not keys:
        return []
    keys_sorted = sorted(keys, key=lambda k: int(k.split("_")[1]))
    last = ctx[keys_sorted[-1]]
    raw = getattr(last, "items", None) or (last.get("items") if isinstance(last, dict) else None) or []
    out: List[dict] = []
    for it in raw:
        if hasattr(it, "model_dump"):
            d = it.model_dump()
        elif hasattr(it, "dict"):
            d = it.dict()
        else:
            try:
                d = dict(it)
            except Exception:
                d = {}
        out.append(d)
    return out


def normalize_fx_tax(inp: NormalizeFxTaxInput, ctx: Dict[str, Any]) -> NormalizeFxTaxOutput:
    prev = _ctx_latest_items(ctx)
    items: List[PriceSearchItem] = []
    target = str(inp.target or "AUD")
    for d in prev:
        items.append(PriceSearchItem(
            title=str(d.get("title") or d.get("name") or "")[:200],
            url=str(d.get("url") or d.get("link") or "")[:1000],
            currency=target,  # 简化：直接目标币种
            price=_to_float(d.get("price")),
            shipping=_to_float(d.get("shipping") or d.get("shipping_cost")),
            tax=_to_float(d.get("tax") or d.get("tax_cost")),
            provider=str(d.get("provider") or d.get("source") or "unknown")[:64],
        ))
    return NormalizeFxTaxOutput(items=items)


def merge_rank(inp: MergeRankInput, ctx: Dict[str, Any]) -> MergeRankOutput:
    prev = _ctx_latest_items(ctx)
    # 去重键：url 或 title + provider
    seen = set()
    kept: List[PriceSearchItem] = []
    for d in prev:
        if inp.dedup == "semantic":
            def norm(s: str) -> str:
                s = s.lower()
                s = re.sub(r"[^a-z0-9 ]+", " ", s)
                s = re.sub(r"\s+", " ", s).strip()
                return s
            key = (norm(str(d.get("title") or "")), norm(str(d.get("provider") or d.get("source") or "")))
        else:
            key = (str(d.get("url") or "").lower(), str(d.get("provider") or d.get("source") or "").lower())
        if not key[0]:
            key = (str(d.get("title") or "").lower(), key[1])
        if key in seen:
            continue
        seen.add(key)
        kept.append(PriceSearchItem(
            title=str(d.get("title") or "")[:200],
            url=str(d.get("url") or d.get("link") or "")[:1000],
            currency=str(d.get("currency") or "AUD"),
            price=_to_float(d.get("price")),
            shipping=_to_float(d.get("shipping") or d.get("shipping_cost")),
            tax=_to_float(d.get("tax") or d.get("tax_cost")),
            provider=str(d.get("provider") or d.get("source") or "unknown")[:64],
        ))

    if inp.strategy == "best_total_cost":
        kept.sort(key=lambda x: (x.price + x.shipping + x.tax))
    else:
        kept.sort(key=lambda x: x.price)

    return MergeRankOutput(items=kept)

# ---------------------------------------------------------
# 推荐
# ---------------------------------------------------------
def reco_generate(inp: RecommendInput, ctx: Dict[str, Any]) -> RecommendOutput:
    def parse_price(v: Any) -> Optional[float]:
        if v is None: return None
        if isinstance(v, (int, float)): return float(v)
        if isinstance(v, str):
            s = v.replace(",", "")
            m = re.search(r"[-+]?\d*\.?\d+", s)
            return float(m.group()) if m else None
        return None

    rec = generate_recommendations(inp.goal)
    cleaned: List[RecommendItem] = []
    for it in rec.items:
        if hasattr(it, "model_dump"): d = it.model_dump()
        elif hasattr(it, "dict"): d = it.dict()
        else:
            try: d = dict(it)
            except Exception: d = {"title": str(it)}
        title = _first_nonempty(d, "title", "name", "product", "label")
        url   = _first_nonempty(d, "url", "link", "product_url", "href")
        reason = _first_nonempty(d, "reason", "why", "rationale", "explain") or ""
        currency = _first_nonempty(d, "currency", "curr") or "AUD"
        price = parse_price(d.get("price"))
        if not title or not url: continue
        cleaned.append(RecommendItem(title=title, reason=reason, url=url, currency=currency, price=price))

    if not cleaned:
        cleaned.append(RecommendItem(
            title=f"Top pick for {inp.goal}",
            reason="fallback item due to missing fields in provider data",
            url="https://example.com/fallback",
            currency="AUD",
            price=None
        ))

    rationale = getattr(rec, "rationale_topk", None) or getattr(rec, "reasoning", None) or []
    if isinstance(rationale, str): rationale = [rationale]
    return RecommendOutput(items=cleaned, rationale_topk=rationale or [])

TOOLS_IMPL = {
    "price.compare_full": price_compare_full,
    "price.search":       price_search,
    "normalize.fx_tax":   normalize_fx_tax,
    "merge.rank":         merge_rank,
    "reco.generate":      reco_generate,
}
