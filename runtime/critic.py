# agent/runtime/critic.py
from typing import Any, Tuple
from .trace import Trace

class CritiqueResult:
    def __init__(self, ok: bool, message: str = "", fix_hint: str = ""):
        self.ok = ok
        self.message = message
        self.fix_hint = fix_hint

def simple_critic(result: Any, trace: Trace) -> CritiqueResult:
    """最小可用版：检查 currency 统一与 total_cost 非负之类的硬性规则。"""
    try:
        items = getattr(result, "items", []) or []
        if not items:
            return CritiqueResult(False, "empty result", "adjust providers or relax filters")

        # 统一币种（先只要求 AUD，后续再做自动归一）
        currencies = {getattr(it, "currency", "AUD") for it in items}
        if len(currencies) > 1:
            return CritiqueResult(False, f"mixed currency: {currencies}", "append normalize_fx_tax step")

        # 总价校验（若有）
        for it in items:
            price = getattr(it, "price", 0.0) or 0.0
            shipping = getattr(it, "shipping", 0.0) or 0.0
            tax = getattr(it, "tax", 0.0) or 0.0
            if price < 0 or shipping < 0 or tax < 0:
                return CritiqueResult(False, "negative cost detected", "clip negatives to zero")

        return CritiqueResult(True, "ok")
    except Exception as e:
        return CritiqueResult(False, f"critic exception: {e}", "re-run merge/rank")
