# agent/runtime/validators.py
from .tool_schemas import PriceSearchOutput, MergeRankOutput, NormalizeFxTaxOutput, RecommendOutput

def basic_struct_checks(tool_name: str, output_obj):
    # 这里先做最小结构校验，后面我们再加业务规则与策略阈值
    if isinstance(output_obj, (PriceSearchOutput, MergeRankOutput, NormalizeFxTaxOutput)):
        assert hasattr(output_obj, "items"), f"{tool_name}: items missing"
    if isinstance(output_obj, RecommendOutput):
        assert hasattr(output_obj, "items"), f"{tool_name}: items missing"
