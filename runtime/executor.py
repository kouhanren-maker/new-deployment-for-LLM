# agent/runtime/executor.py
import asyncio
import time
from typing import Any, Dict, Tuple
from pydantic import ValidationError
from .tool_schemas import ToolRegistry
from .trace import Span, Trace
from .validators import basic_struct_checks

class ExecutionError(Exception):
    pass

class ExecContext(dict):
    """跨步骤的上下文总线，可存取上一步输出、配置等。"""
    pass

class Executor:
    def __init__(self, tools_impl: Dict[str, Any]):
        """
        tools_impl: {tool_name: callable(input_model, ctx) -> output_model | awaitable}
        """
        self.impl = tools_impl

    async def _maybe_await(self, fn, *args, **kwargs):
        res = fn(*args, **kwargs)
        if asyncio.iscoroutine(res):
            return await res
        return res

    async def run_plan(self, plan, trace: Trace) -> Tuple[Any, Trace]:
        last_output = None
        ctx = ExecContext()
        for idx, step in enumerate(plan.steps):
            spec = ToolRegistry.get(step.tool_name)
            if not spec:
                raise ExecutionError(f"Tool not registered: {step.tool_name}")

            fn = self.impl.get(step.tool_name)
            if not fn:
                raise ExecutionError(f"Tool impl missing: {step.tool_name}")

            # 入参校验
            try:
                input_obj = spec.input_model(**step.inputs)
            except ValidationError as e:
                raise ExecutionError(f"Input validation failed for {step.tool_name}: {e}") from e

            # 执行 + 计时
            span = Span(tool=step.tool_name, inputs=step.inputs)
            t0 = time.time()
            out = await self._maybe_await(fn, input_obj, ctx)
            span.end(time.time() - t0)

            # 统一成 dict 再做出参校验（兼容 pydantic v1/v2）
            if hasattr(out, "model_dump"):
                out_dict = out.model_dump()
            elif hasattr(out, "dict"):
                out_dict = out.dict()
            elif isinstance(out, dict):
                out_dict = out
            else:
                out_dict = out.__dict__

            try:
                output_obj = spec.output_model(**out_dict)
            except ValidationError as e:
                raise ExecutionError(f"Output validation failed for {step.tool_name}: {e}") from e

            basic_struct_checks(step.tool_name, output_obj)

            span.out_summary = {
                "size": len(getattr(output_obj, "items", []) or []),
                "keys": list(getattr(output_obj, "model_dump", getattr(output_obj, "dict", lambda: {}) )().keys())
                if hasattr(output_obj, "model_dump") or hasattr(output_obj, "dict") else []
            }
            trace.add(span)

            # 写入上下文，供下一步使用
            ctx[f"step_{idx}_output"] = output_obj
            last_output = output_obj

        return last_output, trace
