"""工具注册中心。"""
from resume_match.tools.base import Tool
from resume_match.tools.extract_jd import ExtractJDTool
from resume_match.tools.check_company import CheckCompanyTool
from resume_match.tools.analyze_gap import AnalyzeGapTool
from resume_match.tools.generate_advice import GenerateAdviceTool


class ToolRegistry:
    """工具注册中心"""

    def __init__(self):
        self.tools: dict[str, Tool] = {}
        self._register_builtin_tools()

    def _register_builtin_tools(self):
        """注册内置工具"""
        builtin = [
            ExtractJDTool(),
            CheckCompanyTool(),
            AnalyzeGapTool(),
            GenerateAdviceTool(),
        ]
        for tool in builtin:
            self.register(tool)

    def register(self, tool: Tool):
        """注册一个工具"""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """获取工具"""
        return self.tools.get(name)

    def list_tools(self) -> list[dict]:
        """列出所有工具"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
            }
            for tool in self.tools.values()
        ]

    def get_tool_descriptions_for_llm(self) -> str:
        """生成工具描述（给 LLM 看）"""
        descriptions = []
        for tool in self.tools.values():
            descriptions.append(f"""
Tool: {tool.name}
Description: {tool.description}
""".strip())
        return "\n\n".join(descriptions)


# 全局单例
registry = ToolRegistry()
