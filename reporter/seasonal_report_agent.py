# agent/reporter/seasonal_report_agent.py
import requests, random, time, os, hashlib
from typing import List
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

class Product(BaseModel):
    rank: int
    name: str
    category: str
    price: float
    sales: int

class SeasonalReport(BaseModel):
    quarter: str
    top_products: List[Product]
    summary: str
    latency_ms: int

def quarter_sales(product_id: int, year: int, quarter: int):
    """根据季度与年份固定随机种子，生成季度销量"""
    seed_str = f"{product_id}-{year}-Q{quarter}"
    seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (10**8)
    random.seed(seed)
    multiplier = {1: 0.8, 2: 1.0, 3: 1.1, 4: 1.3}[quarter]  # Q4 节日加成
    base_sales = random.randint(400, 9000)
    return int(base_sales * multiplier)

def generate_seasonal_report(quarter: str = "2025-Q4", limit: int = 50) -> SeasonalReport:
    start = time.time()
    trace_steps = []
    try:
        # Step 1️⃣: 解析季度与年份
        year, q = quarter.split("-Q")
        year, q = int(year), int(q)
        trace_steps.append({"name": "parse_quarter", "note": f"Year={year}, Quarter={q}"})

        # Step 2️⃣: 调用 FakeStore API
        response = requests.get("https://fakestoreapi.com/products", timeout=10)
        response.raise_for_status()
        data = response.json()
        trace_steps.append({"name": "data_fetch", "note": f"Fetched {len(data)} products"})

        # Step 3️⃣: 生成季度销量
        for p in data:
            p["sales"] = quarter_sales(p["id"], year, q)

        # Step 4️⃣: 排序 Top N
        sorted_data = sorted(data, key=lambda x: x["sales"], reverse=True)[:limit]
        top_products = [
            Product(rank=i+1, name=item["title"], category=item["category"], price=item["price"], sales=item["sales"])
            for i, item in enumerate(sorted_data)
        ]
        trace_steps.append({"name": "data_sort", "note": f"Selected top {limit} products"})

        # Step 5️⃣: LLM 生成季度总结
        llm = ChatOpenAI(model=os.getenv("LLM_MODEL", "gpt-4o-mini"), temperature=0)
        prompt = ChatPromptTemplate.from_template(
            """You are a market analyst.
            Based on the following top-selling products in {quarter}, write a short 3-sentence summary
            covering trends, dominant categories, and insights.
            Top products:
            {products}"""
        )
        products_text = "\n".join([f"{p.rank}. {p.name} ({p.category}) - ${p.price} - Sales: {p.sales}" for p in top_products])

        chain = prompt | llm
        summary = chain.invoke({"quarter": quarter, "products": products_text}).content


    except Exception as e:
        summary = f"Failed to generate report: {str(e)}"
        top_products = []
        trace_steps.append({"name": "error", "note": str(e)})

    latency = int((time.time() - start) * 1000)
    return SeasonalReport(quarter=quarter, top_products=top_products, summary=summary, latency_ms=latency), trace_steps
