# models.py
from pydantic import BaseModel, HttpUrl, Field
from pydantic import AliasChoices, field_validator
from typing import Optional, Dict, Any, List
from datetime import datetime
from typing import Any, Dict, Literal
from pydantic import BaseModel

class CompareQuery(BaseModel):
    text: str                           # 如 "iPhone 15 Pro 256GB Blue"
    region: str = "AU"                  # 区域影响税费/运费策略（此版先不细分）
    currency: str = "AUD"               # 统一目标币种
    prefs: Dict[str, Any] = {}          # 偏好：only_new, max_results, domains_whitelist, budget...

class PriceItem(BaseModel):
    title: str
    brand: Optional[str] = None
    model: Optional[str] = None
    variant: Optional[str] = None
    price: float
    currency: str = "AUD"
    shipping_cost: float = 0.0
    tax_cost: float = 0.0
    seller: str
    seller_rating: Optional[float] = None     # 0~5
    condition: str = "new"                    # new/used/refurbished/unknown
    url: HttpUrl
    source: str                               
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def total_cost(self) -> float:
        return self.price + self.shipping_cost + self.tax_cost

class CompareResult(BaseModel):
    items: List[PriceItem]
    deduped: int
    filtered: int
    citations: List[Dict[str, str]] = []      # 可选：来源链接 [{title,url}]


class AgentQuery(BaseModel):
    # Accept both 'text' and alias 'question' from backend
    # Ensure both field-name and alias work across Pydantic v1/v2
    text: str = Field(..., alias="question")
    merchant_id: Optional[int] = None
    intent: Optional[str] = None
    user_id: Optional[str] = None
    # Accept several backend variants: prefs | preference | preferences
    prefs: Optional[Dict[str, Any]] = Field(
        default=None,
        validation_alias=AliasChoices("prefs", "preference", "preferences"),
    )
    history: Optional[List[Dict[str, Any]]] = None
    region: Optional[str] = None
    currency: Optional[str] = None

    # Normalize prefs: allow plain string like "I like black shirt"
    @field_validator("prefs", mode="before")
    @classmethod
    def _coerce_prefs(cls, v):
        if v is None:
            return None
        if isinstance(v, str):
            # Wrap string into a note so downstream can still read prefs dict
            return {"note": v}
        # Try to coerce list/tuple of pairs to dict
        if isinstance(v, (list, tuple)):
            try:
                return dict(v)
            except Exception:
                return {"list": list(v)}
        if isinstance(v, dict):
            return v
        try:
            return dict(v)
        except Exception:
            return {"value": v}

    # Pydantic v2 config
    model_config = {
        "populate_by_name": True,
        "from_attributes": True,
    }
