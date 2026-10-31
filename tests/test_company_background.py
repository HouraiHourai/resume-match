"""公司检查输出具体背景，并在模型不可用时诚实降级。"""
from copy import deepcopy
from unittest.mock import AsyncMock

import pytest

from resume_match.nodes import tool_executor
from resume_match.state import JDRequirements
from resume_match.tools.check_company import CheckCompanyInput, CheckCompanyTool, CompanyCheckResult


@pytest.mark.asyncio
@pytest.mark.parametrize("company,known", [("字节跳动", True), ("合成示例公司", False)])
async def test_company_check_returns_individual_analysis(monkeypatch, company, known):
    analysis = CompanyCheckResult(
        is_reliable=known, company_type="互联网企业" if known else "背景信息有限",
        reasoning="公司业务涉及内容平台。" if known else "缺少足够的公司背景信息，无法判断主营业务。",
    )
    tool = CheckCompanyTool()
    invoke = AsyncMock(return_value=analysis)
    monkeypatch.setattr(tool, "_check_company_with_llm", invoke)

    output = await tool.execute(CheckCompanyInput(company_name=company), {})
    invoke.assert_awaited_once_with(company)
    assert output.success
    assert output.result["company_name"] == company
    assert output.result["reasoning"] == analysis.reasoning
    assert output.result["company_type"] == analysis.company_type
    assert output.result["verification_status"] == "unverified"


@pytest.mark.asyncio
@pytest.mark.parametrize("company", ["字节跳动", "bytedance"])
async def test_known_company_model_failure_reports_unavailable_analysis(monkeypatch, company):
    tool = CheckCompanyTool()
    monkeypatch.setattr(tool, "_check_company_with_llm", AsyncMock(side_effect=RuntimeError("timeout")))

    output = await tool.execute(CheckCompanyInput(company_name=company), {})
    assert output.success
    assert output.result["company_name"] == company
    assert output.result["verification_status"] == "unverified"
    assert "分析暂时不可用" in output.result["reasoning"]
    assert "放心投递" not in output.result["message"]


@pytest.mark.asyncio
async def test_unknown_company_model_failure_does_not_invent_a_result(monkeypatch):
    tool = CheckCompanyTool()
    monkeypatch.setattr(tool, "_check_company_with_llm", AsyncMock(side_effect=RuntimeError("timeout")))

    output = await tool.execute(CheckCompanyInput(company_name="合成示例公司"), {})
    assert not output.success
    assert output.result is None
    assert "timeout" in output.error


@pytest.mark.asyncio
@pytest.mark.parametrize("action_input", [{}, {"company": "旧公司"}, {"company_name": "旧公司"}])
async def test_executor_checks_company_from_current_jd(monkeypatch, action_input):
    tool = CheckCompanyTool()
    invoke = AsyncMock(return_value=CompanyCheckResult(
        is_reliable=True, company_type="互联网企业", reasoning="当前公司的背景分析。"))
    monkeypatch.setattr(tool, "_check_company_with_llm", invoke)
    monkeypatch.setattr(tool_executor.registry, "get", lambda name: tool)
    state = {
        "next_action": "check_company", "action_input": action_input,
        "jd_structured": JDRequirements(
            role_title="开发工程师", seniority="junior", company="字节跳动",
            hard_skills=[], key_responsibilities=[],
        ),
    }
    original = deepcopy(state)
    updates = await tool_executor.execute_tool(state)
    invoke.assert_awaited_once_with("字节跳动")
    assert updates["company_risk"]["company_name"] == "字节跳动"
    assert state == original
