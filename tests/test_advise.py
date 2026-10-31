"""投递建议服务 (advise) 单元测试。mock LLM, 只验证装填逻辑与错误处理。"""
from unittest.mock import AsyncMock, MagicMock

import pytest

from resume_match.services import adviser as advise_module
from resume_match.state import ApplicationAdvice, ResumeAdvice, BulletRewrite, GapAnalysis, JDRequirements, JDQuality


def _fake_advice() -> ResumeAdvice:
    return ResumeAdvice(
        should_apply="worth_trying",
        should_apply_reason="匹配度 72, 强项能补缺口。",
        bullet_rewrites=[
            BulletRewrite(
                original="做过一个 Agent 项目",
                rewritten="独立设计并实现基于 LangGraph 的 4 节点 Agent 工作流",
                rationale="突出 JD 要求的 LangGraph 关键词。",
            )
        ],
        interview_prep_focus=["LangGraph checkpointer", "Docker 部署"],
        questions_to_ask_interviewer=["团队现在 Agent 的可观测性怎么做?"],
    )


def _fake_gap() -> GapAnalysis:
    return GapAnalysis(
        overall_match_score=72,
        score_breakdown={'tech_match': 25, 'role_fit': 15, 'project_relevance': 20, 'internship_match': 7, 'bonus': 5},
        skill_matches=[],
        strengths=["LangGraph 实战"],
        critical_gaps=["缺 Docker"],
    )


def _fake_jd() -> JDRequirements:
    return JDRequirements(
        role_title="AI Agent 工程师",
        seniority="mid",
        hard_skills=["Python", "Docker"],
        key_responsibilities=["搭 Agent"],
    )


def _patch_model(monkeypatch, *, ainvoke_return=None, ainvoke_raises=None):
    structured_model = MagicMock()
    if ainvoke_raises is not None:
        structured_model.ainvoke = AsyncMock(side_effect=ainvoke_raises)
    else:
        structured_model.ainvoke = AsyncMock(return_value=ainvoke_return)
    model = MagicMock()
    model.with_structured_output.return_value = structured_model
    monkeypatch.setattr(advise_module, "get_deepseek_model", lambda: model)
    # 桩掉 RAG 检索, 避免单测加载 torch / 连 Chroma
    monkeypatch.setattr(advise_module, "find_similar", lambda *a, **k: [])
    return structured_model


@pytest.mark.asyncio
async def test_advise_success(monkeypatch):
    fake = _fake_advice()
    sm = _patch_model(monkeypatch, ainvoke_return=fake)

    state = {
        "jd_structured": _fake_jd(),
        "gap_analysis": _fake_gap(),
        "jd_quality": JDQuality(level="high", reason="职责明确、要求合理"),
        "resume_raw_text": "我的简历..." * 20,
    }
    result = await advise_module.advise(state)

    assert result["advice"] is fake
    assert result["advice"].should_apply == "worth_trying"
    assert result["advice"].bullet_rewrites
    assert result["rewrite_skipped_reason"] is None
    sm.ainvoke.assert_awaited_once()


@pytest.mark.asyncio
async def test_advise_missing_prerequisites(monkeypatch):
    _patch_model(monkeypatch, ainvoke_return=_fake_advice())

    # 缺 gap_analysis
    result = await advise_module.advise({"jd_structured": _fake_jd()})

    assert "advice" not in result
    assert result["errors"] == ["advise: missing prerequisites (gap_analysis or jd_structured)"]


@pytest.mark.asyncio
async def test_advise_llm_error(monkeypatch):
    _patch_model(monkeypatch, ainvoke_raises=RuntimeError("boom"))

    state = {
        "jd_structured": _fake_jd(),
        "gap_analysis": _fake_gap(),
        "jd_quality": JDQuality(level="high", reason="职责明确、要求合理"),
        "resume_raw_text": "x" * 100,
    }
    result = await advise_module.advise(state)

    assert "advice" not in result
    assert "advise failed" in result["errors"][0]
    assert "boom" in result["errors"][0]


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["low", "uncertain"])
async def test_advice_only_schema_excludes_rewrites_and_preserves_suggestions(monkeypatch, level):
    # 即使返回值错误地带了改写，服务也不能把它写入报告。
    fake = _fake_advice()
    sm = _patch_model(monkeypatch, ainvoke_return=fake)
    quality = JDQuality(level=level, reason="入职收费条件需核实",
                        verification_questions=["是否必须先缴纳培训费？"])
    result = await advise_module.advise({
        "jd_structured": _fake_jd(), "gap_analysis": _fake_gap(),
        "jd_quality": quality, "resume_raw_text": "做过一个 Agent 项目",
    })
    model = advise_module.get_deepseek_model()
    model.with_structured_output.assert_called_once_with(ApplicationAdvice, method="function_calling")
    assert "bullet_rewrites" not in ApplicationAdvice.model_json_schema()["properties"]
    prompt = sm.ainvoke.await_args.args[0]
    assert "Skip all resume rewriting" in prompt
    assert "Suggest 3-5 bullet_rewrites" not in prompt
    assert quality.reason in prompt and quality.verification_questions[0] in prompt
    advice = result["advice"]
    assert advice.bullet_rewrites == []
    assert advice.should_apply_reason == fake.should_apply_reason
    assert advice.interview_prep_focus == fake.interview_prep_focus
    assert advice.questions_to_ask_interviewer == fake.questions_to_ask_interviewer
    assert result["rewrite_skipped_reason"]
    assert fake.bullet_rewrites  # 未原地修改模型返回值


@pytest.mark.asyncio
async def test_direct_service_cannot_generate_advice_before_quality_check(monkeypatch):
    sm = _patch_model(monkeypatch, ainvoke_return=_fake_advice())
    result = await advise_module.advise({"jd_structured": _fake_jd(), "gap_analysis": _fake_gap()})
    assert "advice" not in result
    assert result["errors"] == ["advise: missing prerequisite (jd_quality)"]
    sm.ainvoke.assert_not_awaited()
