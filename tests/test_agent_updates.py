from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest
from langgraph.graph import StateGraph, START, END

from resume_match.state import AgentState
from resume_match.nodes import agent, tool_executor
from resume_match.tools.base import ToolInput, ToolOutput


@pytest.mark.asyncio
async def test_agent_returns_only_new_history_without_mutating_input(monkeypatch):
    model = MagicMock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=agent.AgentThought(
        thought="extract", action="extract_jd", reasoning="need jd",
    ))
    monkeypatch.setattr(agent, "get_deepseek_model", lambda: model)
    state = {"errors": ["old"], "react_history": [], "current_step": 0}
    original = deepcopy(state)
    updates = await agent.agent_think(state)
    assert state == original
    assert "errors" not in updates
    assert len(updates["react_history"]) == 1
    assert updates["current_step"] == 1


@pytest.mark.asyncio
async def test_step_limit_does_not_repeat_old_errors():
    updates = await agent.agent_think({"current_step": 10, "errors": ["old"]})
    assert updates["errors"] == ["达到最大步数限制"]


@pytest.mark.asyncio
async def test_executor_isolates_mutations_and_ignores_previous_errors(monkeypatch):
    async def execute(input_obj, local):
        assert local["errors"] == []
        local["jd_raw_text"] = "new text"
        local["action_input"]["nested"].append("mutation")
        local["is_finished"] = True  # 工具不能覆盖图控制字段
        return ToolOutput(success=True, result={"title": "ok"})

    tool = MagicMock(input_schema=ToolInput)
    tool.execute = execute
    monkeypatch.setattr(tool_executor.registry, "get", lambda action: tool)
    state = {"next_action": "extract_jd", "errors": ["old failure"],
             "action_input": {"nested": []}, "tool_results": {"old": 1}}
    original = deepcopy(state)
    updates = await tool_executor.execute_tool(state)
    assert state == original
    assert updates["jd_raw_text"] == "new text"
    assert "errors" not in updates
    assert "is_finished" not in updates
    assert updates["tool_results"]["old"] == 1


@pytest.mark.asyncio
async def test_real_graph_reducer_keeps_each_error_once(monkeypatch):
    tool = MagicMock(input_schema=ToolInput)
    tool.execute = AsyncMock(return_value=ToolOutput(success=False, result=None, error="new failure"))
    monkeypatch.setattr(tool_executor.registry, "get", lambda action: tool)
    builder = StateGraph(AgentState)
    builder.add_node("execute", tool_executor.execute_tool)
    builder.add_edge(START, "execute")
    builder.add_edge("execute", END)
    result = await builder.compile().ainvoke({
        "next_action": "extract_jd", "errors": ["old failure"],
        "react_history": [{"type": "thought", "step": 0}],
    })
    assert result["errors"] == ["old failure", "new failure"]
    assert len(result["react_history"]) == 2

