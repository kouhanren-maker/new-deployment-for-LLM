# runtime/domain/profiles.py
from __future__ import annotations
from typing import Dict, Tuple, Any

_REG: Dict[str, "DomainProfile"] = {}


class DomainProfile:
    """
    各领域 Profile 的基类。
    需要子类实现（或重写）的常用接口：
      - name: str
      - preprocess_queries(text, prefs) -> list[str]
      - entity_extract(text) -> dict
      - filter_model(d, entities, strict=True) -> bool
      - normalize_price(d) -> (ok: bool, price: float|None, reason: str)
      - keep_after_accessory(d, is_accessory_intent=False) -> bool
      - dedup_key(title, d, entities) -> str
      - fallback_plan() -> list[str]
      - soft_score(d, entities, has_total_price: bool) -> float
    """

    name: str = "generic"

    # —— 子类会实现；这里给出保底实现，避免未实现时报错 —— #

    def preprocess_queries(self, text: str, prefs: Dict[str, Any]) -> list[str]:
        # 默认只返回原始查询
        return [text]

    def entity_extract(self, text: str) -> Dict[str, Any]:
        return {}

    def filter_model(self, d: Dict[str, Any], entities: Dict[str, Any], strict: bool = True) -> bool:
        return True

    def normalize_price(self, d: Dict[str, Any]) -> Tuple[bool, float | None, str]:
        # 默认尝试读取 d['price']
        p = d.get("price")
        if p is None:
            return False, None, "missing_price"
        try:
            return True, float(p), "priced"
        except Exception:
            return False, None, "missing_price"

    def keep_after_accessory(self, d: Dict[str, Any], is_accessory_intent: bool = False) -> bool:
        return True

    def dedup_key(self, title: str, d: Dict[str, Any], entities: Dict[str, Any]) -> str:
        # 非严格场景：标题 + 源
        return f"{(title or '').strip().lower()}|{d.get('provider') or d.get('source') or ''}"

    def fallback_plan(self) -> list[str]:
        return []

    def soft_score(self, d: Dict[str, Any], entities: Dict[str, Any], has_total_price: bool) -> float:
        return 0.0

    # ✅ 新增：安全的默认 auto_score，保证 auto_detect 不会因 None 崩溃
    def auto_score(self, text: str) -> Tuple[float, Dict[str, Any]]:
        return 0.0, {}


def register_profile(prof: "DomainProfile") -> None:
    """
    注册 profile 实例。注意：只接收实例一个参数。
    """
    _REG[prof.name] = prof


def get_profile(name: str) -> "DomainProfile":
    return _REG[name]


def all_profiles() -> Dict[str, "DomainProfile"]:
    return dict(_REG)


def auto_detect(text: str) -> Tuple["DomainProfile", float, Dict[str, Any]]:
    """
    在已注册的 profiles 中，根据 auto_score 选出得分最高的。
    返回：(profile, score, evidence)
    具备健壮性：任何异常或非二元返回都会被兜底为 (0.0, {})，不至于抛错。
    """
    best_prof = None
    best_score = -1.0
    best_ev: Dict[str, Any] = {}

    for prof in _REG.values():
        try:
            res = prof.auto_score(text)
            if not isinstance(res, tuple) or len(res) != 2:
                score, ev = 0.0, {}
            else:
                score, ev = res
        except Exception:
            score, ev = 0.0, {}

        if score > best_score:
            best_prof, best_score, best_ev = prof, score, ev

    # 兜底：若都没分，返回 generic 或第一个注册的 profile
    if best_prof is None:
        best_prof = _REG.get("generic") or next(iter(_REG.values()))
        best_score, best_ev = 0.0, {}

    return best_prof, best_score, best_ev
