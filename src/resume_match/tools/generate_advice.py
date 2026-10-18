"""生成投递建议的工具。"""
from resume_match.tools.base import Tool, ToolInput, ToolOutput
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
输入：JD、差距分析、公司风险（如果有）
输出：是否推荐投递、简历改写建议、岗位准备重点

使用场景：完成差距分析后，生成最终的投递建议和简历优化方案。"""

    @property
    def input_schema(self) -> type[ToolInput]:
        return GenerateAdviceInput

    async def execute(self, input: GenerateAdviceInput, state: dict) -> ToolOutput:
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
                    "interview_prep_focus": advice.interview_prep_focus,
                }
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                result=None,
                error=f"生成建议时出错: {str(e)}"
            )
