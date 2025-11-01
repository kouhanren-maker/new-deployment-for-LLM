# app.py
import os
import re
import uuid
import time
from datetime import datetime
from fastapi import FastAPI
from models import CompareQuery, CompareResult, AgentQuery
from orchestrator import PriceCompareOrchestrator
from providers.google_shopping import GoogleShoppingProvider
from router.intent_router import detect_intent
from recommender.recommend_agent import generate_recommendations
from profiler.audience_agent import generate_audience_profile
from reporter.seasonal_report_agent import generate_seasonal_report

# ==== 三层骨架 ====
from runtime.planner import Planner, AgentQuery as RAgentQuery
from runtime.executor import Executor
from runtime.critic import simple_critic
from runtime.trace import Trace
from runtime.memory import mem
from runtime.intent_decider import decide_intent   # 自动意图判断
from tools_impl import TOOLS_IMPL

app = FastAPI(title="AI Agent - OpenAI Cloud Version")
SERVICE_VERSION = os.getenv("AGENT_SERVICE_VERSION", "0.2.0")
SCHEMA_VERSION = "1.0"

@app.get("/test")
def test():
    return {"status": "✅ Backend is running!"}

# --- 保留你现有的比价直达接口 ---
providers = [GoogleShoppingProvider()]
orc = PriceCompareOrchestrator(providers=providers)

@app.post("/compare", response_model=CompareResult)
async def compare(q: CompareQuery):
    return await orc.run(q)

# ====== 兼容 pydantic v1/v2 的安全导出 ======
def _dump_model(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "dict"):
        return obj.dict()
    return obj


def _items_to_backend_products(items):
    """Map internal item objects to backend-consumable 'product' dicts.

    Backend expects at least: name, source, and optional attributes.
    We map url -> source, title -> name, provider -> type, and include one
    single-key attributes dict so Django can create an Attribute if desired.
    """
    out = []
    for it in items or []:
        # Handle both pydantic models and plain dicts
        if hasattr(it, "model_dump"):
            d = it.model_dump()
        elif hasattr(it, "dict"):
            d = it.dict()
        else:
            try:
                d = dict(it)
            except Exception:
                d = {}

        title = str(d.get("title") or d.get("name") or "").strip()
        url = str(d.get("url") or d.get("link") or d.get("product_url") or "").strip()
        provider = str(d.get("provider") or d.get("source") or "").strip()
        price = d.get("price")

        if not url and not title:
            continue

        # 兜底：若无 url，则生成稳定的伪链接，避免后端唯一约束冲突
        if not url:
            import hashlib
            key = f"{title}|{provider}"
            h = hashlib.md5(key.encode("utf-8")).hexdigest()[:16]
            url = f"agent://item/{h}"

        out.append({
            "name": title[:100] if title else "Unknown",
            "description": (title or provider or "")[:100],
            "type": provider[:100] if provider else "",
            "brand": "",  # unknown; optional in backend flow
            "price": price,
            "source": url[:1000] if url else "",
            # Backend picks one attribute if provided
            "attributes": {"provider": provider} if provider else None,
        })
    # 按环境变量限制返回给后端的最大条数（默认 100）
    try:
        topk = int(os.getenv("AGENT_BACKEND_TOPK", "100"))
    except Exception:
        topk = 100
    return out[:max(1, topk)]


def _infer_quarter_from_text(text: str) -> str:
    """从问题文本中推断季度，形如 '2025-Q4'。
    简化规则：
    - 命中 '20xx-Q[1-4]' 直接使用
    - 命中季节词映射为 Q1/Q2/Q3/Q4（Spring=Q1, Summer=Q2, Autumn/Fall=Q3, Winter=Q4），年份用当年
    - 否则返回环境变量 AGENT_DEFAULT_QUARTER 或当前季度
    """
    t = (text or "").lower()
    m = re.search(r"(20\d{2})[-\s]*q([1-4])", t)
    if m:
        return f"{m.group(1)}-Q{m.group(2)}"

    season_map = {
        "spring": 1,
        "summer": 2,
        "autumn": 3,
        "fall": 3,
        "winter": 4,
    }
    for k, qn in season_map.items():
        if k in t:
            year = datetime.utcnow().year
            return f"{year}-Q{qn}"

    env_q = os.getenv("AGENT_DEFAULT_QUARTER")
    if env_q:
        return env_q
    # 计算当前季度
    month = datetime.utcnow().month
    qn = (month - 1) // 3 + 1
    return f"{datetime.utcnow().year}-Q{qn}"


def _refine_prefs(intent: str, prefs: dict, attempt: int) -> dict:
    """根据上一次失败尝试，调整偏好以进行下一轮探索。
    - price: 提升 search_limit，切换 dedup=semantic，固定 strategy=best_total_cost
    - recommend: 提升 topk
    """
    cp = dict(prefs or {})
    try:
        if intent == "price":
            cap = int(os.getenv("AGENT_MAX_SEARCH_LIMIT", "80"))
            cur = int(cp.get("search_limit", 20))
            cp["search_limit"] = min(cur + 20, cap)
            cp["dedup"] = "semantic"
            cp.setdefault("strategy", "best_total_cost")
        else:
            cap = int(os.getenv("AGENT_MAX_TOPK", "10"))
            cur = int(cp.get("topk", 5))
            cp["topk"] = min(cur + 2, cap)
    except Exception:
        pass
    return cp


def _clarification_check(text: str, intent: str, prefs: dict) -> dict:
    """轻量级槽位澄清检测（启发式）。
    返回: {need: bool, missing: {slot:bool}, questions: [str]}
    """
    t = (text or "").lower()
    missing = {}
    questions = []

    if intent == "price":
        has_brand = any(k in t for k in [
            "iphone", "galaxy", "pixel", "oneplus", "xiaomi", "huawei", "ipad", "macbook"
        ])
        has_model_hint = any(ch.isdigit() for ch in t)
        if not (has_brand and has_model_hint):
            missing["product"] = True
            questions.append("你要比较哪款具体型号？例如：iPhone 15 128GB")

    if intent == "recommend":
        if not prefs.get("budget") and "budget" not in prefs:
            missing["budget"] = True
            questions.append("你的预算范围是多少？例如：800-1200 AUD")
        if not any(k in t for k in ["gaming", "office", "travel", "gift", "outfit", "work"]):
            missing["purpose"] = True
            questions.append("你的主要用途/场景是？例如：办公/游戏/送礼")

    return {"need": bool(missing), "missing": missing, "questions": questions}

# --- 统一入口：Planner → Executor → Critic → Trace ---
@app.post("/agent")
async def agent_entry(q: AgentQuery):
    """
    统一入口：
    1. 优先使用请求体 intent
    2. 无 intent 时自动判断（规则 + 实体粒度 + 双路试探 + Critic反验）
    3. 执行 → 验证 → Trace 输出
    """

    # 0) 会话级短期记忆（无需数据库）
    user_id = getattr(q, "user_id", None) or "anon"
    mem.update(user_id, "last_query", q.text)

    # 1) 意图识别
    raw_intent = getattr(q, "intent", None)
    intent_conf = None
    intent_note = ""
    intent_latency = 0

    executor = Executor(TOOLS_IMPL)  # Decider 需要 executor 做双路试探

    if not raw_intent:
        # 优先用 detect_intent 识别季报/用户画像
        try:
            det = detect_intent(q.text)
            if getattr(det, "intent", "") in ("seasonal_report", "user_profile"):
                raw_intent = det.intent
                intent_conf = getattr(det, "confidence", 1.0)
                intent_note = getattr(det, "reason", "router")
        except Exception:
            pass
        # 若识别为季报/画像，直接执行并返回，作为独立功能
        if raw_intent in ("seasonal_report", "user_profile"):
            request_id = str(uuid.uuid4())
            try:
                _prefs = getattr(q, "prefs", None) or {}
            except Exception:
                _prefs = {}
            quarter = str(_prefs.get("quarter") or "").strip() or _infer_quarter_from_text(q.text)
            if not quarter:
                quarter = os.getenv("AGENT_DEFAULT_QUARTER", "2025-Q4")
            if raw_intent == "seasonal_report":
                try:
                    topk = int(os.getenv("AGENT_SEASONAL_TOPK", "50"))
                except Exception:
                    topk = 50
                rep, trace_steps = generate_seasonal_report(quarter=quarter, limit=topk)
                return {
                    "skill": "seasonal_report",
                    "ok": True,
                    "answer": f"Top {len(getattr(rep, 'top_products', []) or [])} best-selling products in {rep.quarter}.",
                    "facts": _dump_model(rep),
                    "trace": {
                        "plan": f"direct seasonal report for {getattr(rep, 'quarter', quarter)}",
                        "steps": trace_steps,
                        "providers": [],
                        "metrics": {"request_id": request_id, "service_version": SERVICE_VERSION},
                    },
                    "request_id": request_id,
                    "schema_version": SCHEMA_VERSION,
                    "service_version": SERVICE_VERSION,
                }
            else:
                prof = generate_audience_profile(q.text)
                return {
                    "skill": "user_profile",
                    "ok": True,
                    "answer": f"Target audience profile generated for '{getattr(prof, 'product', '')}'. Quarter: {quarter}",
                    "facts": _dump_model(prof),
                    "trace": {
                        "plan": f"direct audience profile for {quarter}",
                        "steps": [{
                            "name": "audience_profile_generate",
                            "note": getattr(prof, 'summary', ''),
                            "latency_ms": getattr(prof, 'latency_ms', None)
                        }],
                        "providers": [],
                        "metrics": {"request_id": request_id, "service_version": SERVICE_VERSION},
                    },
                    "quarter": quarter,
                    "request_id": request_id,
                    "schema_version": SCHEMA_VERSION,
                    "service_version": SERVICE_VERSION,
                }
        # 自动判定（规则 + 实体粒度 + 双路试探 + Critic反验）
        planner_intent, intent_conf, evidence = decide_intent(
            text=q.text,
            prefs=getattr(q, "prefs", None) or {},
            history=getattr(q, "history", None) or [],
            executor=executor
        )
        raw_intent = {"price": "price_compare", "recommend": "general_recommend"}[planner_intent]
        intent_note = evidence  # 决策证据写入 trace
    else:
        # 显式指定 intent 时，直接信任
        # 特殊：季报/画像，直接走对应功能并返回
        if raw_intent in ("seasonal_report", "user_profile"):
            request_id = str(uuid.uuid4())
            trace_steps = [{
                "name": "intent_select",
                "result": raw_intent,
                "confidence": 1.0,
                "note": "forced by request.intent",
                "latency_ms": 0,
            }]
            try:
                _prefs = getattr(q, "prefs", None) or {}
            except Exception:
                _prefs = {}
            quarter = str(_prefs.get("quarter") or "").strip() or _infer_quarter_from_text(q.text)
            if not quarter:
                quarter = os.getenv("AGENT_DEFAULT_QUARTER", "2025-Q4")

            if raw_intent == "seasonal_report":
                try:
                    topk = int(os.getenv("AGENT_SEASONAL_TOPK", "50"))
                except Exception:
                    topk = 50
                rep, more_steps = generate_seasonal_report(quarter=quarter, limit=topk)
                if isinstance(more_steps, list):
                    trace_steps.extend(more_steps)
                return {
                    "skill": "seasonal_report",
                    "ok": True,
                    "answer": f"Top {len(getattr(rep, 'top_products', []) or [])} best-selling products in {rep.quarter}.",
                    "facts": _dump_model(rep),
                    "trace": {
                        "plan": f"direct seasonal report for {getattr(rep, 'quarter', quarter)}",
                        "steps": trace_steps,
                        "providers": [],
                        "metrics": {"request_id": request_id, "service_version": SERVICE_VERSION},
                    },
                    "request_id": request_id,
                    "schema_version": SCHEMA_VERSION,
                    "service_version": SERVICE_VERSION,
                }
            else:
                prof = generate_audience_profile(q.text)
                return {
                    "skill": "user_profile",
                    "ok": True,
                    "answer": f"Target audience profile generated for '{getattr(prof, 'product', '')}'. Quarter: {quarter}",
                    "facts": _dump_model(prof),
                    "trace": {
                        "plan": f"direct audience profile for {quarter}",
                        "steps": trace_steps + [{
                            "name": "audience_profile_generate",
                            "note": getattr(prof, 'summary', ''),
                            "latency_ms": getattr(prof, 'latency_ms', None)
                        }],
                        "providers": [],
                        "metrics": {"request_id": request_id, "service_version": SERVICE_VERSION},
                    },
                    "quarter": quarter,
                    "request_id": request_id,
                    "schema_version": SCHEMA_VERSION,
                    "service_version": SERVICE_VERSION,
                }

        if raw_intent in ("price_compare", "price", "compare"):
            planner_intent = "price"
        elif raw_intent in ("general_recommend", "recommend", "reco"):
            planner_intent = "recommend"
        else:
            planner_intent = "recommend"
        intent_conf = 1.0
        intent_note = "forced by request.intent"

    # 防止漏定义
    if "planner_intent" not in locals():
        if raw_intent in ("price_compare", "price", "compare"):
            planner_intent = "price"
        else:
            planner_intent = "recommend"

    # 1.5) 合并/清洗 prefs：禁止 "generic" 盖掉工具层的自动判域
    merged_prefs = dict(getattr(q, "prefs", None) or {})
    dom = str(merged_prefs.get("domain") or "").strip().lower()
    if not dom or dom == "generic":
        merged_prefs.pop("domain", None)   # 让 tools_impl.auto_detect() 自行判域

    # 2) 自主多轮（Autonomous）规划-执行-评审闭环
    # 1.6) 预澄清（可通过环境变量开启）：明显缺槽位时提前返回问题
    try:
        clarify_on = str(os.getenv("AGENT_CLARIFY_PRECHECK", "0")).lower() in ("1", "true", "yes")
    except Exception:
        clarify_on = False
    if clarify_on:
        intended = "price" if raw_intent in ("price_compare", "price", "compare") else "recommend"
        pre = _clarification_check(q.text, intended, merged_prefs)
        if pre.get("need"):
            request_id = str(uuid.uuid4())
            return {
                "ok": False,
                "code": "need_clarification",
                "answer": "为给出更准确结果，请先补充关键信息。",
                "product": [],
                "products": [],
                "need_clarification": True,
                "questions": pre.get("questions", []),
                "missing_slots": pre.get("missing", {}),
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
                "service_version": SERVICE_VERSION,
            }

    runtime_trace = Trace()
    max_iters = int(os.getenv("AGENT_MAX_ITERS", "3"))
    exec_budget_ms = int(os.getenv("AGENT_EXEC_BUDGET_MS", "0") or 0)
    max_steps_budget = int(os.getenv("AGENT_MAX_STEPS", "0") or 0)
    try:
        max_iters = int((getattr(q, "prefs", None) or {}).get("max_iters", max_iters))
    except Exception:
        pass

    current_prefs = dict(merged_prefs)
    attempt = 0
    last_plan = None
    budget_exceeded = False
    steps_exceeded = False
    while True:
        r_query = RAgentQuery(
            intent=planner_intent,
            text=q.text,
            user_id=user_id,
            prefs=current_prefs,
            history=getattr(q, "history", None) or [],
        )
        plan = Planner.plan(r_query)
        last_plan = plan
        result_obj, runtime_trace = await executor.run_plan(plan, runtime_trace)
        critique = simple_critic(result_obj, runtime_trace)
        attempt += 1

        # 预算与步数守护
        step_dicts_now = runtime_trace.to_dict()
        if max_steps_budget > 0 and len(step_dicts_now) >= max_steps_budget:
            steps_exceeded = True
            break
        if exec_budget_ms > 0:
            try:
                total_ms_now = sum(int(s.get("latency_ms") or 0) for s in step_dicts_now)
            except Exception:
                total_ms_now = 0
            if total_ms_now >= exec_budget_ms:
                budget_exceeded = True
                break

        if critique.ok or attempt >= max_iters:
            break
        # 调整偏好，继续下一轮
        current_prefs = _refine_prefs(planner_intent, current_prefs, attempt)

    plan = last_plan if last_plan is not None else plan

    # 6) 组装 Trace（以最终 plan/结果为准）
    # 统计执行指标
    step_dicts = runtime_trace.to_dict()
    try:
        total_latency = sum(int(s.get("latency_ms") or 0) for s in step_dicts)
    except Exception:
        total_latency = None

    highlevel_trace = {
        "plan": plan.rationale,
        "steps": [
            {
                "name": "intent_select",
                "result": raw_intent,
                "confidence": intent_conf,
                "note": intent_note,
                "latency_ms": intent_latency,
            }
        ] + step_dicts,
        "providers": [],
        "metrics": {
            "request_id": None,  # 稍后回填
            "service_version": SERVICE_VERSION,
            "total_latency_ms": total_latency,
            "steps": len(step_dicts),
        },
    }

    # 7) 返回结果
    items = getattr(result_obj, "items", []) or []
    request_id = str(uuid.uuid4())
    if not critique.ok:
        # 多技能回退：尝试另一意图
        try:
            alt_on = str(os.getenv("AGENT_ALT_FALLBACK", "1")).lower() in ("1", "true", "yes")
        except Exception:
            alt_on = True
        if alt_on:
            alt_intent = "recommend" if planner_intent == "price" else "price"
            alt_query = RAgentQuery(
                intent=alt_intent,
                text=q.text,
                user_id=user_id,
                prefs=current_prefs,
                history=getattr(q, "history", None) or [],
            )
            alt_plan = Planner.plan(alt_query)
            alt_result, runtime_trace = await executor.run_plan(alt_plan, runtime_trace)
            alt_crit = simple_critic(alt_result, runtime_trace)
            alt_items = getattr(alt_result, "items", []) or []
            if alt_crit.ok and len(alt_items) > 0:
                plan = alt_plan
                result_obj = alt_result
                critique = alt_crit
                items = alt_items
                planner_intent = alt_intent

        # 若仍失败，构造失败响应（带预算与步数守护码）
        if not critique.ok:
            # 重新汇总 trace（包含回退尝试的步骤）
            step_dicts = runtime_trace.to_dict()
            try:
                total_latency = sum(int(s.get("latency_ms") or 0) for s in step_dicts)
            except Exception:
                total_latency = None
            highlevel_trace = {
                "plan": plan.rationale,
                "steps": [
                    {
                        "name": "intent_select",
                        "result": raw_intent,
                        "confidence": intent_conf,
                        "note": intent_note,
                        "latency_ms": intent_latency,
                    }
                ] + step_dicts,
                "providers": [],
                "metrics": {
                    "request_id": None,
                    "service_version": SERVICE_VERSION,
                    "total_latency_ms": total_latency,
                    "steps": len(step_dicts),
                },
            }
            error_code = "validation_failed"
            if 'budget_exceeded' in locals() and budget_exceeded:
                error_code = "budget_exceeded"
            elif 'steps_exceeded' in locals() and steps_exceeded:
                error_code = "steps_exceeded"
            return {
                "skill": raw_intent,
                "ok": False,
                "code": error_code,
                "answer": "The result didn't pass validation.",
                "hint": critique.fix_hint,
                "product": [],
                "products": [],
                "facts": _dump_model(result_obj),
                "trace": highlevel_trace,
                "plan": plan.dict() if hasattr(plan, "dict") else plan.model_dump(),
                "critic_message": critique.message,
                "need_clarification": True,
                "questions": [
                    "你主要想买什么？请提供产品名或型号",
                    "你的预算范围是多少？例如 800-1200 AUD",
                ],
                "missing_slots": {"product": True, "budget": True},
                "request_id": request_id,
                "schema_version": SCHEMA_VERSION,
                "service_version": SERVICE_VERSION,
            }

    if planner_intent == "price":
        answer = f"Found {len(items)} products after normalization."
        skill = "price_compare"
    else:
        answer = f"I generated {len(items)} recommendations."
        skill = "recommendation"

    try:
        mem.update(user_id, "last_success", {
            "intent": planner_intent,
            "items": len(items),
            "prefs": current_prefs,
        })
    except Exception:
        pass

    # Map to backend shape for Django: { product: [...], answer: str }
    backend_products = _items_to_backend_products(items)

    # facts 可选输出（减小负载）：默认为 True，可通过 prefs.include_facts=false 关闭
    include_facts = True
    if isinstance(merged_prefs, dict):
        include_facts = bool(merged_prefs.get("include_facts", True))

    # 回填 trace.metrics 中的 request_id
    try:
        highlevel_trace["metrics"]["request_id"] = request_id
    except Exception:
        pass

    resp = {
        # New fields for Django backend consumption
        "product": backend_products,
        "products": backend_products,  # keep both keys for tolerance
        "answer": answer,

        # Keep original fields for other clients/tools
        "skill": skill,
        "ok": True,
        **({"facts": _dump_model(result_obj)} if include_facts else {}),
        "trace": highlevel_trace,
        "plan_rationale": plan.rationale,
        "request_id": request_id,
        "schema_version": SCHEMA_VERSION,
        "service_version": SERVICE_VERSION,
    }

    # 商家画像：对接季度报告 -> merchant_hot_products
    # 注意：后端当前模型与写库字段存在不一致（merchant_id 字段缺失），
    # 若后端未调整，写库会报参数错误；此处仍按其期望结构回传。
    attach_insights = str(os.getenv("AGENT_ATTACH_INSIGHTS", "0")).lower() in ("1", "true", "yes")
    if attach_insights and getattr(q, "merchant_id", None) is not None:
        try:
            quarter = _infer_quarter_from_text(q.text)
            # 生成季度报告（使用已有 reporter）
            rep, trace_steps = generate_seasonal_report(quarter=quarter, limit=int(os.getenv("AGENT_HOT_TOPK", "10")))

            # 映射为后端期望的 hot products 结构
            mhp_list = []
            for p in getattr(rep, "top_products", []) or []:
                sales = int(getattr(p, "sales", 0) or 0)
                view_count = max(sales * 3, sales)  # 简单估算：浏览量>=购买量
                mhp_list.append({
                    "merchant_id": q.merchant_id,
                    "name": getattr(p, "name", "")[:50],
                    "view_count": int(view_count),
                    "purchase_count": sales,
                    "season": quarter,
                })

            # 生成用户画像（Audience Profile）并映射为 product_user_portraits
            prof = generate_audience_profile(q.text)
            # 解析年龄段到平均年龄
            def _age_avg_from_range(r: str) -> int:
                try:
                    nums = [int(x) for x in re.findall(r"\d+", r or "")]
                    if len(nums) >= 2:
                        return int((nums[0] + nums[1]) / 2)
                    if len(nums) == 1:
                        return int(nums[0])
                except Exception:
                    pass
                return 25

            age_avg = _age_avg_from_range(getattr(getattr(prof, "demographics", None), "age_range", "") or "")
            genders = getattr(getattr(prof, "demographics", None), "gender", None) or []
            gender = (genders[0] if isinstance(genders, list) and genders else "unisex").title()
            locations = getattr(getattr(prof, "demographics", None), "location", None) or []
            region = (locations[0] if isinstance(locations, list) and locations else "global").title()

            pup_list = []
            # 与热销商品对应数量生成画像（若无热销则至少给 1 条）
            base_count = len(mhp_list) or 1
            for _ in range(base_count):
                pup_list.append({
                    # 后端当前写库期望的键名；若后端未对齐模型，该字段可能被忽略或需允许 null
                    "merchant_hot_product_id": None,
                    "age_avg": age_avg,
                    "gender": gender,
                    "region": region,
                })

            # 附加到返回值与 trace
            resp["merchant_hot_products"] = mhp_list
            resp["product_user_portraits"] = pup_list
            # 把季报步骤追加进 trace
            if isinstance(trace_steps, list):
                highlevel_trace["steps"].extend(trace_steps)
            # 把季报与画像摘要拼接进 answer（不改变前端解析 product 的逻辑）
            extra = []
            if getattr(rep, "summary", None):
                extra.append(f"Seasonal insight: {rep.summary}")
            if getattr(prof, "summary", None):
                extra.append(f"Audience insight: {prof.summary}")
            if extra:
                resp["answer"] = f"{resp['answer']}\n" + "\n".join(extra)
        except Exception:
            # 保底：即使季报/画像失败，也不影响主路径
            resp.setdefault("merchant_hot_products", [])
            resp.setdefault("product_user_portraits", [])
    return resp


@app.get("/health")
async def health():
    return {"ok": True, "service": "ai-agent", "version": SERVICE_VERSION}


@app.get("/version")
async def version():
    return {"service_version": SERVICE_VERSION}

# ======================
# 旧直达接口（保留用于对比）
# ======================

@app.post("/agent/recommend")
async def agent_recommend(q: AgentQuery):
    rec = generate_recommendations(q.text)
    trace = {
        "plan": "legacy: direct recommendation",
        "steps": [
            {
                "name": "recommendation_generate",
                "note": getattr(rec, 'reasoning', ''),
                "latency_ms": getattr(rec, 'latency_ms', None)
            }
        ],
        "providers": [],
        "metrics": {},
    }
    return {
        "skill": "recommendation",
        "answer": f"I generated {len(rec.items)} recommendations under category '{getattr(rec, 'category', 'N/A')}'.",
        "facts": _dump_model(rec),
        "trace": trace,
    }

@app.post("/agent/seasonal")
async def agent_seasonal(q: AgentQuery):
    # 动态识别季度：优先使用 prefs.quarter，其次从 question 文本解析，最后退回环境变量默认值
    try:
        prefs = getattr(q, "prefs", None) or {}
    except Exception:
        prefs = {}
    quarter = str(prefs.get("quarter") or "").strip() or _infer_quarter_from_text(q.text)
    if not quarter:
        quarter = os.getenv("AGENT_DEFAULT_QUARTER", "2025-Q4")
    try:
        topk = int(os.getenv("AGENT_SEASONAL_TOPK", "50"))
    except Exception:
        topk = 50
    rep, trace_steps = generate_seasonal_report(quarter=quarter, limit=topk)
    trace = {
        "plan": f"legacy: direct seasonal report for {rep.quarter}",
        "steps": trace_steps,
        "providers": [],
        "metrics": {},
    }
    return {
        "skill": "seasonal_report",
        "answer": f"Top {len(rep.top_products)} best-selling products in {rep.quarter}.",
        "facts": _dump_model(rep),
        "trace": trace,
    }

@app.post("/agent/profile")
async def agent_profile(q: AgentQuery):
    # 动态识别季度：优先 prefs.quarter，其次从 question 文本解析，最后退回默认值
    try:
        prefs = getattr(q, "prefs", None) or {}
    except Exception:
        prefs = {}
    quarter = str(prefs.get("quarter") or "").strip() or _infer_quarter_from_text(q.text)
    if not quarter:
        quarter = os.getenv("AGENT_DEFAULT_QUARTER", "2025-Q4")

    prof = generate_audience_profile(q.text)
    trace = {
        "plan": f"legacy: direct audience profile for {quarter}",
        "steps": [
            {
                "name": "audience_profile_generate",
                "note": prof.summary,
                "latency_ms": getattr(prof, 'latency_ms', None)
            }
        ],
        "providers": [],
        "metrics": {},
    }
    return {
        "skill": "user_profile",
        "answer": f"Target audience profile generated for '{prof.product}'. Quarter: {quarter}",
        "facts": _dump_model(prof),
        "trace": trace,
        "quarter": quarter,
    }
