"""Agent 推理节点 - ReAct 核心。"""
from resume_match.state import AgentState
from resume_match.config import get_deepseek_model
from resume_match.tools.registry import registry
from pydantic import BaseModel, Field


class AgentThought(BaseModel):
    """Agent 的推理结果"""
    thought: str = Field(description="当前的思考")
    action: str = Field(description="下一步要执行的 action: 'extract_jd' / 'check_company' / 'analyze_gap' / 'generate_advice' / 'finish'")
    action_input: dict = Field(default_factory=dict, description="action 的输入参数")
    reasoning: str = Field(description="为什么选择这个 action")


async def agent_think(state: AgentState) -> dict:
    """只返回本轮增量；add reducer 字段绝不回传旧列表。"""
    current_step = state.get("current_step", 0)
    if current_step >= state.get("max_steps", 10):
        return {"is_finished": True, "errors": ["达到最大步数限制"]}
    try:
        model = get_deepseek_model()
        structured_model = model.with_structured_output(AgentThought, method="function_calling")
        response = await structured_model.ainvoke(_build_agent_prompt(state))
        return {
            "current_step": current_step + 1,
            "next_action": response.action,
            "action_input": response.action_input,
            "is_finished": response.action == "finish",
            "react_history": [{
                "type": "thought", "step": current_step,
                "thought": response.thought, "action": response.action,
                "reasoning": response.reasoning,
            }],
        }
    except Exception as e:
        return {"errors": [f"Agent 推理失败: {e}"], "is_finished": True}


def _build_agent_prompt(state: AgentState) -> str:
    """构建 Agent 推理的 prompt"""

    # 获取工具描述
    tool_descriptions = registry.get_tool_descriptions_for_llm()

    # 获取历史记录
    history = state.get("react_history", [])
    history_text = []
    for h in history[-10:]:  # 显示最近 10 步
        if h['type'] == 'thought':
            history_text.append(f"Step {h['step']}: [思考] 行动={h.get('action')}")
        elif h['type'] == 'observation':
            success = "✅" if h.get('success') else "❌"
            result_preview = str(h.get('result') if h.get('success') else h.get('error', ''))[:100]
            history_text.append(f"Step {h['step']}: [结果{success}] {result_preview}")

    history_str = "\n".join(history_text) if history_text else "（无）"

    # 检查是否有连续失败的情况
    recent_failures = []
    for h in history[-5:]:  # 查看最近5步
        if h['type'] == 'observation' and not h.get('success'):
            recent_failures.append(h.get('action'))

    # 如果同一个工具连续失败3次，应该结束
    has_repeated_failures = len(recent_failures) >= 3 and len(set(recent_failures[-3:])) == 1

    # 获取已有信息
    has_jd = bool(state.get("jd_structured"))
    has_gap = bool(state.get("gap_analysis"))
    has_advice = bool(state.get("advice"))
    company_name = state.get("jd_structured").company if has_jd else None
    has_company_risk = bool(state.get("company_risk"))

    failure_warning = ""
    if has_repeated_failures:
        failure_warning = "\n⚠️ 警告：同一个工具连续失败多次，应该返回 'finish' 停止执行，并在 thought 中说明失败原因。"

    prompt = f"""你是一个 Job Description 分析 Agent。

任务：帮助用户分析岗位 JD 与简历的匹配度，并给出投递建议。

可用工具：
{tool_descriptions}

当前状态：
- JD 输入：{state.get("jd_input", "未提供")[:100]}...
- 简历：{"已提供" if state.get("resume_raw_text") else "未提供"}
- JD 结构化：{"已完成" if has_jd else "未完成"}
{f"- 公司名称：{company_name}" if company_name else ""}
- 公司风险检查：{"已完成" if has_company_risk else "未完成（请调用 check_company 工具）"}
- 差距分析：{"已完成" if has_gap else "未完成"}
- 建议生成：{"已完成" if has_advice else "未完成"}

历史操作：
{history_str}{failure_warning}

推理规则：
1. 如果 JD 未结构化，先调用 extract_jd
2. 如果 extract_jd 失败（如列表页 URL、网站防护、URL 无效、JD 内容无效等），立即返回 'finish' 并在 thought 中说明失败原因，不要继续调用其他工具
3. 如果 JD 已结构化且有公司名称，调用 check_company 检查公司可信度（工具内部对知名公司有快速路径，无需 Agent 判断）
4. 如果公司风险检查已完成或没有公司名称，调用 analyze_gap 进行差距分析
5. 如果差距分析已完成，调用 generate_advice 生成最终建议
6. 如果建议已生成，返回 finish
7. 如果任何工具连续失败多次，返回 finish 停止执行

请基于当前状态，决定下一步要执行的 action。

返回格式（JSON）：
{{
  "thought": "当前的思考",
  "action": "下一步的 action（extract_jd/check_company/analyze_gap/generate_advice/finish）",
  "action_input": {{}},
  "reasoning": "为什么选择这个 action"
}}
"""

    return prompt
