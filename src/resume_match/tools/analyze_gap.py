"""分析简历与 JD 差距的工具。"""
from resume_match.tools.base import Tool, ToolInput, ToolOutput
from resume_match.services.gap_analyzer import analyze


class AnalyzeGapInput(ToolInput):
    """分析差距的输入（从 state 中读取）"""
    pass


class AnalyzeGapTool(Tool):
    """分析简历与 JD 的差距"""

    @property
    def name(self) -> str:
        return "analyze_gap"

    @property
    def description(self) -> str:
        return """分析简历与 JD 的差距。
输入：JD 结构化信息 + 简历
输出：匹配度评分、技能匹配情况、优势、关键缺口

使用场景：在提取 JD 后，分析用户简历与岗位要求的匹配程度。"""

    @property
    def input_schema(self) -> type[ToolInput]:
        return AnalyzeGapInput

    async def execute(self, input: AnalyzeGapInput, state: dict) -> ToolOutput:
        # 检查前置条件：必须有 jd_structured
        if not state.get("jd_structured"):
            error_msg = "无法进行差距分析：JD 未成功提取"
            return ToolOutput(
                success=False,
                result=None,
                error=error_msg
            )

        try:
            analyze_result = await analyze(state)
            state.update(analyze_result)  # 合并返回的 dict 到 state

            if state.get("errors"):
                return ToolOutput(
                    success=False,
                    result=None,
                    error=state['errors'][-1]  # 直接返回原始错误，不再包装
                )

            gap = state["gap_analysis"]
            return ToolOutput(
                success=True,
                result={
                    "overall_match_score": gap.overall_match_score,
                    "matched_skills": [sm.skill for sm in gap.skill_matches if sm.status == "matched"],
                    "missing_skills": [sm.skill for sm in gap.skill_matches if sm.status == "missing"],
                    "strengths": gap.strengths,
                    "critical_gaps": gap.critical_gaps,
                    "minor_gaps": gap.minor_gaps,
                }
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                result=None,
                error=f"分析差距时出错: {str(e)}"
            )
