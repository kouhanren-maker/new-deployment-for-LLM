# agent/runtime/planner.py
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from .tool_schemas import ToolRegistry

class Step(BaseModel):
    tool_name: str
    inputs: Dict[str, Any]
    success_criteria: Optional[str] = None

class Plan(BaseModel):
    steps: List[Step]
    rationale: str = ""
    risks: List[str] = []

class AgentQuery(BaseModel):
    intent: str
    text: str
    user_id: Optional[str] = None
    prefs: Dict[str, Any] = {}
    history: List[Dict[str, Any]] = []

class Planner:
    """最小可运行 Planner：根据意图拼装固定步骤。"""
    @staticmethod
    def plan(q: AgentQuery) -> Plan:
        steps: List[Step] = []
        # 优先：若多步工具齐全，则直接返回多步流水线
        if q.intent in ("price", "compare", "price_compare") and all(name in ToolRegistry for name in ("price.search", "normalize.fx_tax", "merge.rank")):
            steps.append(Step(
                tool_name="price.search",
                inputs={
                    "query": q.text,
                    "providers": q.prefs.get("providers", ["mock_a", "mock_b"]),
                    "limit": int(q.prefs.get("search_limit", 20)),
                },
            ))
            steps.append(Step(
                tool_name="normalize.fx_tax",
                inputs={
                    "target": q.prefs.get("currency", "AUD"),
                    "region": q.prefs.get("region", "AU"),
                },
            ))
            steps.append(Step(
                tool_name="merge.rank",
                inputs={
                    "strategy": q.prefs.get("strategy", "best_total_cost"),
                    "dedup": q.prefs.get("dedup", "exact"),
                },
            ))
            return Plan(steps=steps, rationale="Multi-step: search -> normalize -> merge&rank.", risks=["provider timeout", "low precision intent"])
        if q.intent in ("price", "compare", "price_compare"):
            # 用整合工具（真实 orchestrator），一步到位
            if "price.compare_full" in ToolRegistry:
                steps.append(Step(tool_name="price.compare_full",
                                inputs={"text": q.text,
                                        "region": q.prefs.get("region", "AU"),
                                        "currency": q.prefs.get("currency", "AUD"),
                                        "prefs": q.prefs}))
            rationale = "Full price comparison via orchestrator."

        elif q.intent in ("recommend", "reco"):
            if "reco.generate" in ToolRegistry:
                steps.append(Step(tool_name="reco.generate",
                                  inputs={"goal": q.text,
                                          "budget_aud": q.prefs.get("budget"),
                                          "topk": int(q.prefs.get("topk", 5))}))
            rationale = "Recommendation based on goal and (optional) budget."
        else:
            # 默认走推荐
            if "reco.generate" in ToolRegistry:
                steps.append(Step(tool_name="reco.generate",
                                  inputs={"goal": q.text,
                                          "budget_aud": q.prefs.get("budget"),
                                          "topk": int(q.prefs.get("topk", 5))}))
            rationale = "Fallback to recommendation."
        return Plan(steps=steps, rationale=rationale, risks=["provider timeout", "low precision intent"])
