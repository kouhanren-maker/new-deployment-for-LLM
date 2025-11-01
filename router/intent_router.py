# router/intent_router.py
import os, time
from typing import Optional, Literal
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI


# ==============================================================
# 🔹 一、关键词规则表（仅作为 LLM 提示，不再强制分类）
# ==============================================================

KEYS = {
    "general_recommend": [
        "recommend", "suggest", "idea", "gift", "present",
        "outfit", "wear", "clothing", "dress", "style",
        "gadget", "device", "electronics", "product"
    ],
    "price_compare": [
        "price", "cheap", "buy", "cost", "compare", "deal"
    ],
    "seasonal_report": [
        "report", "trend", "quarter", "kpi", "analysis"
    ],
    "user_profile": [
    "profile", "preference", "interest", "budget",
    "persona", "buyer persona", "user persona",
    "audience", "target", "target customer", "target audience",
    "customer profile", "customer segment", "segmentation",
    "demographic", "psychographic", "market segment",
    "用户画像", "画像", "目标用户", "目标人群", "受众", "人群", "人设", "细分", "年龄", "性别"
    ]

}


def rule_based(text: str) -> Optional[str]:
    """
    简单规则匹配：若命中关键词，则作为 hint 提示给 LLM。
    不再直接决定 intent。
    """
    t = text.lower()
    for intent, words in KEYS.items():
        if any(w in t for w in words):
            return intent
    return None


# ==============================================================
# 🔹 二、意图数据模型
# ==============================================================

class IntentSchema(BaseModel):
    intent: Literal[
        "general_recommend",
        "price_compare",
        "seasonal_report",
        "user_profile",
        "other"
    ]
    confidence: float
    reason: str
    latency_ms: Optional[int] = None


# ==============================================================
# 🔹 三、Prompt 模板（LLM 主导判断）
# ==============================================================

prompt = ChatPromptTemplate.from_template(
    """You are an intent classifier for a multi-skill AI agent.

Categories:
- "general_recommend": any request for recommendations, ideas, gifts, outfits, gadgets, products, or suggestions
- "price_compare": any request asking for cheapest, price, compare, cost, or deals
- "seasonal_report": any request for reports, KPIs, or trends
- "user_profile": any request for user/customer personas or audience profiling (target customers, demographics, psychographics, segments), as well as user preferences or budgets
- "other": anything else

If a rule-based hint is provided, prefer it **only if it clearly fits the user query**.
If the user asks to "build a persona" or "target customer/audience" for a product, classify as "user_profile".

HINT (may be empty): {rule_hint}

Return ONLY valid JSON matching this schema:
{{
  "intent": "general_recommend|price_compare|seasonal_report|user_profile|other",
  "confidence": 0.0-1.0,
  "reason": "short reason (<=10 words)"
}}

User query: {query}
Return JSON only.
"""
)


# ==============================================================
# 🔹 四、主函数：detect_intent
# ==============================================================

def detect_intent(text: str) -> IntentSchema:
    """
    综合 rule-based hint + LLM 判断。
    LLM 永远是最终决策者；规则只做提示。
    """
    hint = rule_based(text) or ""

    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=int(os.getenv("LLM_TIMEOUT_S", "10")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3"))
    )

    parser = JsonOutputParser(pydantic_object=IntentSchema)
    chain = prompt | llm | parser

    t0 = time.time()
    try:
        out_dict = chain.invoke({"query": text, "rule_hint": hint})
        out = IntentSchema(**out_dict)
        out.latency_ms = int((time.time() - t0) * 1000)
        return out
    except Exception as e:
        # 若 LLM 出错则 fallback
        return IntentSchema(
            intent="other",
            confidence=0.0,
            reason=f"fallback_error: {str(e)}",
            latency_ms=int((time.time() - t0) * 1000)
        )
