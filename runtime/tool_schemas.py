# agent/runtime/tool_schemas.py
from typing import List, Literal, Optional, Dict, Any
from pydantic import BaseModel, Field, validator

# ====== 输入/输出 Schema ======
class PriceSearchInput(BaseModel):
    query: str
    providers: List[str] = Field(default_factory=lambda: ["mock_a", "mock_b"])
    limit: int = 20

class PriceItem(BaseModel):
    title: str
    url: str
    currency: str = "AUD"
    price: float
    shipping: float = 0.0
    tax: float = 0.0
    provider: str

    @property
    def total_cost(self) -> float:
        return self.price + self.shipping + self.tax

class PriceSearchOutput(BaseModel):
    items: List[PriceItem]

class MergeRankInput(BaseModel):
    strategy: Literal["best_total_cost", "best_price"] = "best_total_cost"
    dedup: Literal["exact", "semantic"] = "exact"

class MergeRankOutput(BaseModel):
    items: List[PriceItem]

class NormalizeFxTaxInput(BaseModel):
    target: str = "AUD"
    region: str = "AU"

class NormalizeFxTaxOutput(BaseModel):
    items: List[PriceItem]

class RecommendInput(BaseModel):
    goal: str
    budget_aud: Optional[float] = None
    topk: int = 5

class RecommendItem(BaseModel):
    title: str
    reason: str
    url: str
    currency: str = "AUD"
    price: Optional[float] = None

class RecommendOutput(BaseModel):
    items: List[RecommendItem]
    rationale_topk: List[str] = []

# ====== 工具签名（统一调用规范）======
class ToolSpec(BaseModel):
    name: str
    input_model: Any
    output_model: Any
    description: str

ToolRegistry: Dict[str, ToolSpec] = {}

def register_tool(name: str, input_model: Any, output_model: Any, description: str):
    ToolRegistry[name] = ToolSpec(
        name=name, input_model=input_model, output_model=output_model, description=description
    )

# ====== Compare (full pipeline via orchestrator) ======
class CompareFullInput(BaseModel):
    text: str
    region: str = "AU"
    currency: str = "AUD"
    prefs: Dict[str, Any] = {}

class CompareItem(BaseModel):
    title: str
    url: str
    currency: str = "AUD"
    price: float
    shipping: float = 0.0
    tax: float = 0.0
    provider: str

class CompareFullOutput(BaseModel):
    items: List[CompareItem]
    diagnostics: Dict[str, Any] = {}   # <<< 新增（可选、向后兼容）

