import os, time, requests
from typing import List, Optional
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# ==============================================================
# 🔹 一、数据结构
# ==============================================================

class RecommendationItem(BaseModel):
    name: str
    reason: str
    link: Optional[str] = None          # ✅ 新增字段
    price: Optional[str] = None         # ✅ 新增字段
    source: Optional[str] = None        # ✅ 新增字段

class Recommendation(BaseModel):
    category: str
    recommend_type: Optional[str] = None
    items: List[RecommendationItem]
    reasoning: str
    latency_ms: Optional[int] = None
    extract_latency_ms: Optional[int] = None


# ==============================================================
# 🔹 二、主 Prompt：生成推荐
# ==============================================================

prompt = ChatPromptTemplate.from_template(
    """You are a general-purpose recommendation agent (outfits, gifts, electronics, etc.).
Understand the user's request and return 3–5 recommendations in JSON.

Schema:
{{
  "category": "short summary of what these recommendations are for",
  "items": [{{"name":"item name","reason":"one-sentence reason"}}],
  "reasoning": "2–3 sentences summarizing your reasoning"
}}

User request: {query}

Return only valid JSON following the schema exactly.
"""
)


# ==============================================================
# 🔹 三、类型识别 Prompt
# ==============================================================

type_prompt = ChatPromptTemplate.from_template(
    """Determine which recommendation type best fits the user's request.

Categories:
- "outfit": clothing, fashion, matching, what to wear
- "gift": presents, ideas, things to give to others
- "electronics": gadgets, tech devices, accessories
- "food": recipes, restaurants, drinks
- "other": anything else

User query: {query}

Return ONLY valid JSON:
{{"recommend_type": "outfit|gift|electronics|food|other"}}
"""
)


# ==============================================================
# 🔹 四、SerpAPI 辅助函数（用于富化推荐结果）
# ==============================================================

SERP_API_KEY = os.getenv("SERPAPI_KEY")

def find_product_link(query: str, gl: str = "us", hl: str = "en") -> Optional[dict]:
    """
    调用 SerpAPI (Google Shopping) 获取首条结果信息。
    返回 {"title","price","url","source"} 或 None。
    """
    if not SERP_API_KEY:
        return None
    params = {
        "engine": "google_shopping",
        "q": query,
        "gl": gl,
        "hl": hl,
        "num": 3,
        "api_key": SERP_API_KEY,
    }
    try:
        r = requests.get("https://serpapi.com/search.json", params=params, timeout=8)
        j = r.json()
        for s in j.get("shopping_results", []):
            return {
                "title": s.get("title"),
                "price": s.get("price") or s.get("extracted_price"),
                "url": s.get("link") or s.get("product_link"),
                "source": s.get("source"),
            }
    except Exception:
        return None
    return None


# ==============================================================
# 🔹 五、主函数：生成推荐 + 类型识别 + 链接富化
# ==============================================================

def generate_recommendations(query: str) -> Recommendation:
    # 初始化 LLM
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0.7,
        timeout=int(os.getenv("LLM_TIMEOUT_S", "25")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "5")),
    )

    parser = JsonOutputParser(pydantic_object=Recommendation)
    chain = prompt | llm | parser

    # Step 1️⃣: 生成推荐
    t0 = time.time()
    try:
        raw = chain.invoke({"query": query})
        if isinstance(raw, Recommendation):
            data = raw.model_dump()
        elif hasattr(raw, "model_dump"):
            data = raw.model_dump()
        elif hasattr(raw, "dict"):
            data = raw.dict()
        else:
            data = dict(raw) if isinstance(raw, dict) else {}

        data["latency_ms"] = int((time.time() - t0) * 1000)
    except Exception as e:
        return Recommendation(
            category="unknown",
            items=[],
            reasoning=f"Fallback: unable to generate ({str(e)})",
            latency_ms=int((time.time() - t0) * 1000),
        )

    # Step 2️⃣: 识别推荐类型
    type_parser = JsonOutputParser()
    type_chain = type_prompt | llm | type_parser

    t1 = time.time()
    try:
        type_result = type_chain.invoke({"query": query})
        if isinstance(type_result, dict):
            recommend_type = type_result.get("recommend_type", "other")
        else:
            recommend_type = getattr(type_result, "recommend_type", "other")
        data["recommend_type"] = recommend_type
        data["extract_latency_ms"] = int((time.time() - t1) * 1000)
    except Exception as e:
        data["recommend_type"] = f"extract_error({str(e)})"
        data["extract_latency_ms"] = int((time.time() - t1) * 1000)

    # Step 3️⃣: 富化推荐结果（SerpAPI 链接/价格/来源）
    enrich_start = time.time()
    items = data.get("items", [])
    success_count = 0
    for it in items[:5]:  # 仅前5条
        q = f"{it['name']} {data.get('category','')} buy"
        hit = find_product_link(q)
        if hit:
            it["link"] = hit.get("url")
            it["price"] = hit.get("price")
            it["source"] = hit.get("source")
            success_count += 1
    data["extract_latency_ms"] = (data.get("extract_latency_ms") or 0) + int((time.time() - enrich_start) * 1000)

    # Step 4️⃣: 记录富化情况
    data["reasoning"] += f"\n\n(Enriched with {success_count} product links via SerpAPI.)"

    # Step 5️⃣: 返回结果
    return Recommendation(**data)
