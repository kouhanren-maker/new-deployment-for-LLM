# agent/runtime/trace.py
from typing import Any, Dict, List, Optional

class Span:
    def __init__(self, tool: str, inputs: Dict[str, Any]):
        self.tool = tool
        self.inputs = inputs
        self.latency_ms: Optional[int] = None
        self.out_summary: Dict[str, Any] = {}

    def end(self, elapsed_s: float):
        self.latency_ms = int(elapsed_s * 1000)

class Trace:
    def __init__(self):
        self.spans: List[Span] = []

    def add(self, span: Span):
        self.spans.append(span)

    def to_dict(self):
        return [
            {"tool": s.tool, "latency_ms": s.latency_ms, "out": s.out_summary}
            for s in self.spans
        ]
