"""岗位质量证据约束、工具前置条件与真实图分支的离线回归。"""
from copy import deepcopy
from unittest.mock import AsyncMock, MagicMock

import pytest

from resume_match.graph import build_graph
from resume_match.nodes import agent, tool_executor
from resume_match.services import jd_quality, adviser
from resume_match.state import JDQuality, JDRequirements, ResumeAdvice, ApplicationAdvice, GapAnalysis, ScoreBreakdown
from resume_match.tools import assess_jd_quality, generate_advice, analyze_gap, check_company
from resume_match.tools.core.base import ToolInput, ToolOutput


RAW = "Python 工程师。负责开发业务接口，要求熟悉 Python 和 SQL。入职前需缴纳培训费。"


def jd():
    return JDRequirements(role_title="Python 工程师", seniority="junior",
                          hard_skills=["Python", "SQL"], key_responsibilities=["开发业务接口"])


def quality(level):
    return JDQuality(level=level, reason="依据岗位职责及招聘条件进行初步判断",
                     positive_signals=[{"point": "职责明确", "evidence": "负责开发业务接口"}],
                     risk_signals=([{"point": "入职收费", "evidence": "入职前需缴纳培训费"}]
                                   if level == "low" else []),
                     verification_questions=["请确认招聘条件。"])


def patch_quality_model(monkeypatch, result):
    monkeypatch.setattr(jd_quality, "get_deepseek_model", MagicMock())
    invoke = AsyncMock(return_value=result)
    monkeypatch.setattr(jd_quality, "ainvoke_structured_with_retry", invoke)
    return invoke


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["high", "low", "uncertain"])
async def test_quality_preserves_supported_assessment(monkeypatch, level):
    expected = quality(level)
    invoke = patch_quality_model(monkeypatch, expected)
    state = {"jd_raw_text": RAW, "jd_structured": jd()}
    original = deepcopy(state)
    result = await jd_quality.assess_quality(state)
    assert result["jd_quality"] == expected
    assert state == original
    prompt = invoke.await_args.args[1]
    assert RAW in prompt
    assert "缺少薪资" in prompt and "不要用企业知名度判断好坏" in prompt


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["high", "low"])
@pytest.mark.parametrize("quote", ["", "原文未出现的虚构条款"])
async def test_unsupported_evidence_cannot_pass_or_reject_job(monkeypatch, level, quote):
    result = quality(level)
    signals = result.positive_signals if level == "high" else result.risk_signals
    signals[0].evidence = quote
    patch_quality_model(monkeypatch, result)
    updates = await jd_quality.assess_quality({"jd_raw_text": RAW, "jd_structured": jd()})
    assert updates["jd_quality"].level == "uncertain"
    assert updates["jd_quality"].verification_questions
    assert "虚构条款" not in updates["jd_quality"].model_dump_json()


@pytest.mark.asyncio
async def test_model_error_is_not_a_successful_quality_assessment(monkeypatch):
    invoke = patch_quality_model(monkeypatch, quality("high"))
    invoke.side_effect = RuntimeError("timeout")
    result = await jd_quality.assess_quality({"jd_raw_text": RAW, "jd_structured": jd()})
    assert "jd_quality" not in result
    assert "timeout" in result["errors"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("state", [{}, {"jd_raw_text": RAW}, {"jd_structured": jd()}])
async def test_quality_requires_original_and_extracted_jd(monkeypatch, state):
    invoke = patch_quality_model(monkeypatch, quality("high"))
    assert (await jd_quality.assess_quality(state))["errors"]
    invoke.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("module,tool_cls,service", [
    (generate_advice, generate_advice.GenerateAdviceTool, "advise"),
    (analyze_gap, analyze_gap.AnalyzeGapTool, "analyze"),
])
async def test_direct_tool_cannot_bypass_quality_gate(monkeypatch, module, tool_cls, service):
    expensive = AsyncMock()
    monkeypatch.setattr(module, service, expensive)
    state = {"jd_structured": jd(), "gap_analysis": object()}
    result = await tool_cls().execute(ToolInput(), state)
    assert result.success is False
    expensive.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["high", "low", "uncertain", "failure"])
@pytest.mark.parametrize("company_check_order", [None, "before_gap", "after_gap"])
async def test_real_graph_generates_advice_for_all_quality_levels(monkeypatch, level, company_check_order):
    async def extract(input_obj, state):
        position = jd().model_copy(update={"company": "字节跳动" if company_check_order else None})
        state.update(jd_raw_text=RAW, jd_structured=position)
        return ToolOutput(success=True, result={"title": "Python 工程师"})

    extracted = MagicMock(input_schema=ToolInput, execute=AsyncMock(side_effect=extract))
    assessed = AsyncMock(return_value=({"errors": ["质量服务不可用"]} if level == "failure"
                                      else {"jd_quality": quality(level)}))
    monkeypatch.setattr(assess_jd_quality, "assess_quality", assessed)

    analyzed = AsyncMock(return_value={"gap_analysis": GapAnalysis(
            overall_match_score=70, score_breakdown=ScoreBreakdown(
                tech_match=30, role_fit=15, project_relevance=20, internship_match=5, bonus=0),
            skill_matches=[], strengths=["Python"], critical_gaps=[])})
    monkeypatch.setattr(analyze_gap, "analyze", analyzed)
    advice_model = MagicMock()
    advised = AsyncMock(return_value=ResumeAdvice(
        should_apply="worth_trying", should_apply_reason="职责相关，需核实招聘条件",
        bullet_rewrites=[{"original": "开发接口", "rewritten": "使用 Python 开发业务接口", "rationale": "突出相关技能"}],
        interview_prep_focus=["Python 接口开发"], questions_to_ask_interviewer=["请确认招聘条件。"],
    ))
    advice_model.with_structured_output.return_value.ainvoke = advised
    monkeypatch.setattr(adviser, "get_deepseek_model", lambda: advice_model)
    monkeypatch.setattr(adviser, "find_similar", lambda *a, **k: [])
    company_tool = check_company.CheckCompanyTool()
    company_analysis = AsyncMock(return_value=check_company.CompanyCheckResult(
        is_reliable=True, company_type="互联网企业", reasoning="公司业务涉及内容平台和企业服务。"))
    monkeypatch.setattr(company_tool, "_check_company_with_llm", company_analysis)
    tools = {"extract_jd": extracted, "assess_jd_quality": assess_jd_quality.AssessJDQualityTool(),
             "check_company": company_tool,
             "analyze_gap": analyze_gap.AnalyzeGapTool(), "generate_advice": generate_advice.GenerateAdviceTool()}
    monkeypatch.setattr(tool_executor.registry, "get", tools.get)
    model = MagicMock()
    actions = ["extract_jd", "assess_jd_quality", "analyze_gap", "generate_advice"]
    if company_check_order:
        actions.insert(2 if company_check_order == "before_gap" else 3, "check_company")
    model.with_structured_output.return_value.ainvoke = AsyncMock(side_effect=[
        agent.AgentThought(thought="选择下一步", action=action, reasoning="基于当前观察")
        for action in actions
    ])
    monkeypatch.setattr(agent, "get_deepseek_model", lambda: model)
    result = await build_graph().ainvoke({"jd_input": RAW, "resume_raw_text": "熟悉 Python"})
    assert result["is_finished"]
    if level != "failure":
        analyzed.assert_awaited_once()
        advised.assert_awaited_once()
        expected_schema = ResumeAdvice if level == "high" else ApplicationAdvice
        advice_model.with_structured_output.assert_called_once_with(expected_schema, method="function_calling")
        assert result["advice"] and not result.get("stop_reason")
        assert result["gap_analysis"].overall_match_score == 70
        assert result["advice"].interview_prep_focus
        assert bool(result["advice"].bullet_rewrites) == (level == "high")
        assert bool(result["rewrite_skipped_reason"]) == (level != "high")
        assert result["tool_results"]["generate_advice"]["rewrite_skipped_reason"] == result["rewrite_skipped_reason"]
        if company_check_order:
            company_analysis.assert_awaited_once_with("字节跳动")
            assert result["company_risk"]["company_name"] == "字节跳动"
            assert result["company_risk"]["reasoning"] == "公司业务涉及内容平台和企业服务。"
        else:
            company_analysis.assert_not_awaited()
    else:
        company_analysis.assert_not_awaited()
        analyzed.assert_not_awaited()
        advised.assert_not_awaited()
        assert result.get("advice") is None and result["stop_reason"]
    assert bool(result.get("errors")) == (level == "failure")
    observations = [entry for entry in result["react_history"] if entry["type"] == "observation"]
    assert [entry["action"] for entry in observations][:2] == ["extract_jd", "assess_jd_quality"]


@pytest.mark.asyncio
async def test_agent_rejects_rewrite_without_quality(monkeypatch):
    model = MagicMock()
    model.with_structured_output.return_value.ainvoke = AsyncMock(return_value=agent.AgentThought(
        thought="跳过质量检查", action="generate_advice", reasoning="错误动作"))
    monkeypatch.setattr(agent, "get_deepseek_model", lambda: model)
    result = await agent.agent_think({"jd_structured": jd(), "resume_raw_text": "Python"})
    assert result["is_finished"] and result["errors"]
    assert result.get("next_action") != "generate_advice"


def test_agent_sees_complete_quality_evidence_and_can_choose_check_order():
    report = quality("high")
    report.reason = "很长的质量判断" * 50 + "结尾关键依据"
    position = jd().model_copy(update={"company": "某普通企业"})
    state = {"jd_structured": position, "jd_quality": report, "resume_raw_text": "Python"}
    assert "结尾关键依据" in agent._build_agent_prompt(state)
    assert set(agent._available_actions(state)) == {"check_company", "analyze_gap"}
    state["gap_analysis"] = GapAnalysis(
        overall_match_score=70,
        score_breakdown=ScoreBreakdown(tech_match=30, role_fit=15, project_relevance=20, internship_match=5, bonus=0),
        skill_matches=[], strengths=[], critical_gaps=[],
    )
    assert agent._available_actions(state) == ["check_company"]
    state["company_risk"] = {"company_name": "某普通企业", "reasoning": "背景信息有限。"}
    assert agent._available_actions(state) == ["generate_advice"]
