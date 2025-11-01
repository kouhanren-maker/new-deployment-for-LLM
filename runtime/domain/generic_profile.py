# agent/runtime/domain/generic_profile.py
from __future__ import annotations
import re
from typing import Dict, Any, List, Tuple, Optional

from .profiles import DomainProfile, register_profile

class GenericProfile:
    name = "generic"

    @staticmethod
    def auto_score(text: str) -> Tuple[float, Dict[str, Any]]:
        # 默认域，低分
        return 0.1, {"hits": []}

    @staticmethod
    def preprocess_queries(text: str, prefs: Dict[str, Any]) -> List[str]:
        return [text]

    @staticmethod
    def entity_extract(text: str) -> Dict[str, Any]:
        return {}

    @staticmethod
    def filter_model(item: Dict[str, Any], entities: Dict[str, Any], *, strict: bool) -> bool:
        return True  # 不做型号守门

    @staticmethod
    def normalize_price(item: Dict[str, Any]) -> Tuple[bool, Optional[float], Optional[str]]:
        # 直接尝试取一口价
        try:
            p = float(item.get("price"))
            return True, p, None
        except Exception:
            return False, None, "missing_price"

    @staticmethod
    def keep_after_accessory(item: Dict[str, Any], *, is_accessory_intent: bool) -> bool:
        return True  # generic 域不做配件过滤

    @staticmethod
    def dedup_key(title: str, item: Dict[str, Any]) -> str:
        return re.sub(r"\s+", " ", (title or "").lower()).strip()

    @staticmethod
    def fallback_plan() -> List[str]:
        return ["allow_installment_only_as_null_price"]  # 与手机域保持一个公共策略名


register_profile(GenericProfile())
