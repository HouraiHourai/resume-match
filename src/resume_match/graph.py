"""ResumeMatch 主图: ReAct 循环架构。

Agent 自主推理并动态调用工具：
- extract_jd: 提取 JD 结构
- check_company: 检查公司风险（仅对不知名公司）
- analyze_gap: 分析差距
- generate_advice: 生成建议
"""
from langgraph.graph import StateGraph, START, END

from resume_match.state import AgentState
from resume_match.nodes.agent import agent_think
from resume_match.nodes.tool_executor import execute_tool


def should_continue(state: AgentState) -> str:
    """判断是否继续循环"""
    if state.get("is_finished"):
        return "end"
    return "continue"


def build_graph():
    """构建 ReAct 风格的 graph"""
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("agent_think", agent_think)  # Agent 推理
    builder.add_node("execute_tool", execute_tool)  # 执行工具

    # 添加边
    builder.add_edge(START, "agent_think")

    # 条件边：agent_think → execute_tool 或 END
    builder.add_conditional_edges(
        "agent_think",
        should_continue,
        {
            "continue": "execute_tool",
            "end": END,
        }
    )

    # execute_tool → agent_think（循环）
    builder.add_edge("execute_tool", "agent_think")

    return builder.compile()


# 全局实例, CLI 和后续 web server 都复用它
graph = build_graph()
