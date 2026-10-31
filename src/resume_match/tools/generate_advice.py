"""生成投递建议的工具。"""
from resume_match.tools.core.base import Tool, ToolInput, ToolOutput
from resume_match.services.adviser import advise


class GenerateAdviceInput(ToolInput):
    """生成建议的输入（从 state 中读取）"""
    pass


class GenerateAdviceTool(Tool):
    """生成最终的投递建议和简历优化建议"""

    @property
    def name(self) -> str:
        return "generate_advice"

    @property
    def description(self) -> str:
        return """生成最终的投递建议和简历优化建议。
输入：JD、岗位质量、差距分析
输出：是否推荐投递、简历改写建议、岗位准备重点

使用场景：岗位质量评估和差距分析完成后，生成投递与准备建议。
high 同时生成简历改写；low/uncertain 跳过改写，仍提供建议和核实问题。"""

    @property
    def input_schema(self) -> type[ToolInput]:
        return GenerateAdviceInput

    async def execute(self, input: GenerateAdviceInput, state: dict) -> ToolOutput:
        quality = state.get("jd_quality")
        if not quality:
            return ToolOutput(success=False, result=None, error="岗位质量尚未评估，不能生成建议")
        # 检查前置条件：必须有 gap_analysis 和 jd_structured
        if not state.get("gap_analysis") or not state.get("jd_structured"):
            error_msg = "无法生成建议：缺少差距分析或 JD 信息"
            return ToolOutput(
                success=False,
                result=None,
                error=error_msg
            )

        try:
            advise_result = await advise(state)
            state.update(advise_result)  # 合并返回的 dict 到 state

            if state.get("errors"):
                return ToolOutput(
                    success=False,
                    result=None,
                    error=state['errors'][-1]  # 直接返回原始错误，不再包装
                )

            advice = state["advice"]
            return ToolOutput(
                success=True,
                result={
                    "should_apply": advice.should_apply,
                    "reason": advice.should_apply_reason,
                    "bullet_rewrites_count": len(advice.bullet_rewrites),
                    "rewrite_skipped_reason": state.get("rewrite_skipped_reason"),
                    "interview_prep_focus": advice.interview_prep_focus,
                }
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                result=None,
                error=f"生成建议时出错: {str(e)}"
            )
