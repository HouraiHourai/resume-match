"""提取和结构化 JD 的工具。"""
from resume_match.tools.base import Tool, ToolInput, ToolOutput
from resume_match.services.jd_fetcher import fetch_jd
from resume_match.services.jd_extractor import extract
from resume_match.state import JDRequirements


def _is_invalid_jd(jd: JDRequirements) -> bool:
    """判断 JD 是否无效。

    无效 JD 特征：
    1. role_title 为空、占位符（如"未提供"）或明显非法
    2. 核心字段（hard_skills, key_responsibilities）全部为空
    3. 其他重要字段（domain, years_of_experience, education）全部缺失
    """
    # 检查 role_title
    title = jd.role_title.strip() if jd.role_title else ""
    invalid_titles = ["未提供", "未知", "无", "N/A", "null", "None", ""]

    # role_title 包含明显的错误标记
    if any(marker in title for marker in ["非法", "无效", "不存在", "已下架"]):
        return True

    # role_title 为空或占位符
    if not title or title in invalid_titles:
        return True

    # 核心字段检查：hard_skills 和 key_responsibilities 都为空
    has_skills = bool(jd.hard_skills and len(jd.hard_skills) > 0)
    has_responsibilities = bool(jd.key_responsibilities and len(jd.key_responsibilities) > 0)

    #   没有技能要求 → 不知道招什么样的人
    #   没有职责描述 → 不知道这个岗位要做什么
    #   双核心字段都为空 → 判定无效
    if not has_skills and not has_responsibilities:
        return True

    # 如果 hard_skills 只有 1-2 个且都是非常通用的词（可能是 LLM 瞎猜的）
    if has_skills and len(jd.hard_skills) <= 2:
        generic_skills = ["agent", "ai", "llm", "后端", "前端", "开发", "工程师"]
        if all(any(g in skill.lower() for g in generic_skills) for skill in jd.hard_skills):
            # 进一步检查：如果其他字段也都为空，判定为无效
            if not jd.domain and not jd.years_of_experience and not jd.education:
                return True

    return False


class ExtractJDInput(ToolInput):
    """提取 JD 的输入（从 state 中读取）"""
    pass

# 统一工具接口
class ExtractJDTool(Tool):
    """提取并结构化 JD 内容"""

    @property
    def name(self) -> str:
        return "extract_jd"

    @property
    def description(self) -> str:
        return """提取并结构化 JD 内容。
输入：用户提供的 JD URL 或文本（从 state.jd_input 读取）
输出：结构化的 JD 信息（公司、岗位、技能要求、薪资等）

使用场景：分析任务的第一步，必须先提取 JD 信息才能进行后续分析。"""

    @property
    def input_schema(self) -> type[ToolInput]:
        return ExtractJDInput

    async def execute(self, input: ExtractJDInput, state: dict) -> ToolOutput:
        try:
            # 1. fetch JD
            fetch_result = await fetch_jd(state)
            state.update(fetch_result)  # 合并返回的 dict 到 state

            if state.get("errors"):
                # 直接返回详细错误，不添加"JD 获取失败"前缀
                return ToolOutput(
                    success=False,
                    result=None,
                    error=state["errors"][-1]  # 返回详细错误信息
                )

            # 2. extract structured info
            extract_result = await extract(state)
            state.update(extract_result)  # 合并返回的 dict 到 state

            if state.get("errors"):
                # 直接返回详细错误，不添加"JD 结构提取失败"前缀
                return ToolOutput(
                    success=False,
                    result=None,
                    error=state["errors"][-1]  # 返回详细错误信息
                )

            jd = state["jd_structured"]

            # 验证 JD 有效性：检查是否为无效/空 JD
            is_invalid = _is_invalid_jd(jd)
            if is_invalid:
                # 判断输入是 URL 还是文本
                jd_input = state.get("jd_input", "")
                is_url = jd_input.startswith(("http://", "https://"))

                if is_url:
                    error_msg = (
                        "JD 内容无效或为空，无法进行匹配分析。\n"
                        "可能原因：\n"
                        "1. URL 参数不完整或职位ID非法（如 jobUnionId= 后无值）\n"
                        "2. 职位已下架或不存在\n"
                        "3. 抓取到的是错误页面\n\n"
                        "请检查 URL 是否正确，或直接粘贴完整的 JD 文本。"
                    )
                else:
                    error_msg = (
                        "JD 文本无效或内容不完整，无法进行匹配分析。\n"
                        "可能原因：\n"
                        "1. 文本内容过短或缺少关键信息（职位名称、技能要求等）\n"
                        "2. 文本格式不规范或包含大量无关内容\n"
                        "3. 文本是错误页面或提示信息\n\n"
                        "请粘贴完整的职位描述文本，或使用职位详情页 URL。"
                    )

                # 将错误信息添加到 state，并标记为失败
                state["errors"] = state.get("errors", []) + [error_msg]
                return ToolOutput(
                    success=False,
                    result=None,
                    error=error_msg
                )

            return ToolOutput(
                success=True,
                result={
                    "company": jd.company,
                    "title": jd.role_title,
                    "location": jd.location,
                    "seniority": jd.seniority,
                    "hard_skills": jd.hard_skills,
                    "nice_to_have": jd.nice_to_have,
                    "salary_range": jd.salary_range,
                    "years_of_experience": jd.years_of_experience,
                }
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                result=None,
                error=f"提取 JD 时出错: {str(e)}"
            )
