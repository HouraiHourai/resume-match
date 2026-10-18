"""工具执行边界：隔离旧式工具的原地写入，仅提交成功产出和本轮轨迹。"""
from copy import deepcopy

from resume_match.state import AgentState
from resume_match.tools.registry import registry

# 每个工具能提交的业务字段，不允许工具覆盖 reducer/循环控制字段。
_OUTPUT_FIELDS = {
    "extract_jd": ("jd_raw_text", "company_name", "jd_structured"),
    "analyze_gap": ("gap_analysis",),
    "generate_advice": ("advice", "similar_histories"),
    "check_company": (),
}


async def execute_tool(state: AgentState) -> dict:
    action = state.get("next_action")
    if not action or action == "finish":
        return {}
    tool = registry.get(action)
    if not tool:
        return {"errors": [f"工具 '{action}' 不存在"], "is_finished": True}

    # 工具沿用局部 state.update，但不能修改调用者的嵌套列表、参数或模型。
    working = deepcopy(dict(state))
    working["errors"] = []
    working["react_history"] = []
    action_input = deepcopy(state.get("action_input") or {})
    try:
        # 参数名兼容
        if action == "check_company" and "company" in action_input and "company_name" not in action_input:
            action_input["company_name"] = action_input.pop("company")

        output = await tool.execute(tool.input_schema(**action_input), working) # 用参数构造 schema 对象，执行工具
        success, result, error = output.success, output.result, output.error
    except Exception as exc:
        success, result, error = False, None, f"执行工具 '{action}' 失败: {exc}"

    updates = {"react_history": [{
        "type": "observation", "step": state.get("current_step", 0),
        "action": action, "success": success,
        "result": result if success else None,
        "error": error if not success else None,
    }]}
    if success:
        updates.update({key: working[key] for key in _OUTPUT_FIELDS.get(action, ()) if key in working})
        updates["tool_results"] = {**state.get("tool_results", {}), action: result}
        if action == "check_company":
            updates["company_risk"] = result
    else:
        # 历史错误只供审计，不参与本轮工具是否成功的判断。
        updates["errors"] = [error or f"工具 '{action}' 失败"]
    return updates
