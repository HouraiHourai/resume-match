"""岗位质量评估工具。"""
from resume_match.services.jd_quality import assess_quality
from resume_match.tools.core.base import Tool, ToolInput, ToolOutput


class AssessJDQualityTool(Tool):
    @property
    def name(self) -> str:
        return "assess_jd_quality"

    @property
    def description(self) -> str:
        return ("依据 JD 原文评估岗位可信度线索、要求合理性与是否值得继续分析。"
                "输出 high/low/uncertain、原文依据和待核实问题。"
                "提取完成后、差距分析与改写前必须评估；不提供外部真实性核验。")

    @property
    def input_schema(self) -> type[ToolInput]:
        return ToolInput

    async def execute(self, input: ToolInput, state: dict) -> ToolOutput:
        updates = await assess_quality(state)
        if updates.get("errors"):
            return ToolOutput(success=False, result=None, error=updates["errors"][-1])
        state.update(updates)
        return ToolOutput(success=True, result=state["jd_quality"].model_dump())
