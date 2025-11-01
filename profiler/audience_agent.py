# agent/profiler/audience_agent.py
import os, time
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# =========================
# 数据模型
# =========================
class Demographics(BaseModel):
    gender: List[str] = Field(default_factory=list)
    age_range: str = "unknown"          # e.g., "25-40"
    income_level: str = "unknown"       # e.g., "mid-to-high"
    location: List[str] = Field(default_factory=list)

class AudienceProfile(BaseModel):
    product: str
    category: str
    demographics: Demographics
    psychographics: List[str] = Field(default_factory=list)
    purchase_motivations: List[str] = Field(default_factory=list)
    objections: List[str] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    price_band: str = "unknown"
    channels: List[str] = Field(default_factory=list)          # marketing / acquisition channels
    keywords: List[str] = Field(default_factory=list)          # SEO / ads
    similar_products: List[str] = Field(default_factory=list)
    summary: str = ""
    latency_ms: Optional[int] = None

# =========================
# Prompt
# =========================
prompt = ChatPromptTemplate.from_template(
    """You are a market segmentation analyst. Build a crisp, data-shaped audience profile for the given product.

Return ONLY valid JSON matching this schema:
{{
  "product": "<the product as understood>",
  "category": "<short category name>",
  "demographics": {{
    "gender": ["female","male","unisex", "..."],
    "age_range": "e.g., 25-40",
    "income_level": "low|mid|mid-to-high|high",
    "location": ["global","US","EU","AU", "..."]
  }},
  "psychographics": ["style/values/persona..."],
  "purchase_motivations": ["..."],
  "objections": ["..."],
  "use_cases": ["..."],
  "price_band": "e.g., $60–$120",
  "channels": ["Instagram Reels","TikTok","Pinterest","Google Shopping","Retail boutiques","..."],
  "keywords": ["SEO/ad keywords..."],
  "similar_products": ["..."],
  "summary": "2–3 sentences concise explanation"
}}

Guidelines:
- Be specific but concise.
- If the product implies gender or age, reflect it, otherwise set unisex/neutral.
- Include both functional and emotional motivations.
- Always fill all fields (if unknown, infer sensibly).

User product/query: {query}
"""
)

# =========================
# 主函数
# =========================
def generate_audience_profile(query: str, market_hint: str = "global") -> AudienceProfile:
    """
    输入一个产品/品类短语，输出结构化目标客户画像。
    market_hint 可传 'US'/'AU' 等，对语言与地域有轻微引导（如需，可在 prompt 中扩展）。
    """
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        timeout=int(os.getenv("LLM_TIMEOUT_S", "20")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
    )

    parser = JsonOutputParser(pydantic_object=AudienceProfile)
    chain = prompt | llm | parser

    t0 = time.time()
    try:
        out_dict = chain.invoke({"query": query})
        # 兼容性处理：parser 可能返回 dict
        if isinstance(out_dict, AudienceProfile):
            prof = out_dict
        else:
            prof = AudienceProfile(**out_dict)
        prof.latency_ms = int((time.time() - t0) * 1000)
        return prof
    except Exception as e:
        # 失败时，返回结构化降级
        return AudienceProfile(
            product=query,
            category="unknown",
            demographics=Demographics(gender=["unisex"], age_range="unknown", income_level="unknown", location=[market_hint]),
            psychographics=[],
            purchase_motivations=[],
            objections=[],
            use_cases=[],
            price_band="unknown",
            channels=[],
            keywords=[],
            similar_products=[],
            summary=f"fallback_error: {str(e)}",
            latency_ms=int((time.time() - t0) * 1000),
        )
