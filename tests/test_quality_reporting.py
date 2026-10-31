"""低质量岗位继续生成建议，仅跳过改写，结果通过各入口和历史记录返回。"""
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from resume_match import cli
from resume_match.api import routes
from resume_match.api.server import app
from resume_match.history import storage
from resume_match.state import JDQuality, JDRequirements, GapAnalysis, ResumeAdvice


@pytest.fixture
def client(monkeypatch, tmp_path):
    storage.set_storage(storage.HistoryStorage(db_path=tmp_path / "history.db"))
    monkeypatch.setattr(routes, "_resume_text_from_upload", AsyncMock(return_value="合成简历"))
    with TestClient(app) as test_client:
        yield test_client
    storage.set_storage(None)


def _advice_only_state(level):
    return {
        "jd_structured": JDRequirements(
            role_title="Python 工程师", seniority="junior",
            hard_skills=["Python"], key_responsibilities=["开发业务接口"],
        ),
        "jd_quality": JDQuality(
            level=level,
            reason="入职前要求付费，建议先核实招聘条件。",
            positive_signals=[],
            risk_signals=[{"point": "可能存在入职收费", "evidence": "录用后需缴纳培训费"}],
            verification_questions=["是否必须先缴纳费用才能入职？"],
        ),
        "stop_reason": None,
        "rewrite_skipped_reason": "岗位质量较低或信息待核实，已跳过简历改写。",
        "gap_analysis": GapAnalysis(
            overall_match_score=70,
            score_breakdown={"tech_match": 30, "role_fit": 15, "project_relevance": 20,
                             "internship_match": 5, "bonus": 0},
            skill_matches=[], strengths=["熟悉 Python"], critical_gaps=[],
        ),
        "advice": ResumeAdvice(
            should_apply="not_recommended", should_apply_reason="请先核实入职收费条件。",
            bullet_rewrites=[], interview_prep_focus=["Python 接口设计"],
            questions_to_ask_interviewer=["是否必须先缴纳费用才能入职？"],
        ),
        "errors": [],
        "is_finished": True,
    }


def _events_for(state, action="assess_jd_quality", tool_output=None):
    async def events(*args, **kwargs):
        yield {
            "event": "on_chain_start", "name": "execute_tool",
            "metadata": {"langgraph_node": "execute_tool"},
            "data": {"input": {"next_action": action}},
        }
        yield {
            "event": "on_chain_end", "name": "execute_tool",
            "metadata": {"langgraph_node": "execute_tool"},
            "data": {"output": tool_output or {}},
        }
        yield {
            "event": "on_chain_end", "name": "LangGraph",
            "metadata": {}, "data": {"output": state},
        }
    return events


def _request(client, streamed):
    return client.post(
        "/api/analyze/stream" if streamed else "/api/analyze",
        data={"jd_text": "Python 工程师，开发业务接口；录用后需缴纳培训费"},
        files={"resume_file": ("resume.pdf", b"synthetic", "application/pdf")},
    )


def _final_report(response, streamed):
    if not streamed:
        return response.json()
    final_event = next(part for part in response.text.split("\n\n") if part.startswith("event: final_result\n"))
    return json.loads(final_event.split("\ndata: ", 1)[1])


@pytest.mark.parametrize("level", ["low", "uncertain"])
@pytest.mark.parametrize("streamed", [False, True])
def test_advice_without_rewrites_is_reported_and_saved(client, monkeypatch, level, streamed):
    state = _advice_only_state(level)
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=state)
    graph.astream_events = _events_for(state)
    monkeypatch.setattr(routes, "graph", graph)

    response = _request(client, streamed)
    assert response.status_code == 200
    report = _final_report(response, streamed)
    assert report["jd_quality"] == state["jd_quality"].model_dump()
    assert report["stop_reason"] == state["stop_reason"]
    assert report["rewrite_skipped_reason"] == state["rewrite_skipped_reason"]
    assert report["errors"] == []
    assert report["gap"]["overall_match_score"] == 70
    assert report["advice"] == state["advice"].model_dump()
    assert report["advice"]["bullet_rewrites"] == []
    assert isinstance(report["history_id"], int)
    if streamed:
        assert 'event: node_start\ndata: {"node": "quality"}' in response.text
        assert 'event: node_end\ndata: {"node": "quality"}' in response.text
        assert "event: error\n" not in response.text

    history = client.get(f"/api/history/{report['history_id']}").json()
    assert history["full_result"]["jd_quality"] == report["jd_quality"]
    assert history["full_result"]["stop_reason"] == state["stop_reason"]
    assert history["full_result"]["rewrite_skipped_reason"] == state["rewrite_skipped_reason"]
    assert history["full_result"]["advice"] == report["advice"]
    assert history["match_score"] == 70
    assert history["should_apply"] == "not_recommended"


@pytest.mark.parametrize("streamed", [False, True])
def test_failed_analysis_does_not_create_history(client, monkeypatch, streamed):
    state = _advice_only_state("uncertain")
    state["errors"] = ["质量评估服务失败"]
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=state)
    graph.astream_events = _events_for(state)
    monkeypatch.setattr(routes, "graph", graph)

    response = _request(client, streamed)
    report = _final_report(response, streamed)
    assert report["history_id"] is None
    assert report["errors"] == state["errors"]
    assert client.get("/api/history").json() == []


def test_stream_reports_errors_from_tools_without_progress_step(client, monkeypatch):
    state = _advice_only_state("uncertain")
    state["errors"] = ["公司检查失败"]
    graph = MagicMock()
    graph.astream_events = _events_for(state, "check_company", {"errors": state["errors"]})
    monkeypatch.setattr(routes, "graph", graph)

    response = _request(client, True)
    assert response.text.count("event: error\n") == 1
    assert "公司检查失败" in response.text
    assert _final_report(response, True)["history_id"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("level", ["low", "uncertain"])
async def test_cli_includes_advice_and_rewrite_skip_reason(monkeypatch, level):
    state = _advice_only_state(level)
    graph = MagicMock()
    graph.ainvoke = AsyncMock(return_value=state)
    monkeypatch.setattr(cli, "graph", graph)
    monkeypatch.setattr(cli, "extract_pdf_text", lambda path: "合成简历")

    report = await cli.run("合成 JD", Path("unused.pdf"))
    assert report["jd_quality"] == state["jd_quality"].model_dump()
    assert report["stop_reason"] == state["stop_reason"]
    assert report["advice"] == state["advice"].model_dump()
    assert report["rewrite_skipped_reason"] == state["rewrite_skipped_reason"]
    assert report["errors"] == []
    markdown = cli.render_markdown(report)
    assert "岗位质量评估" in markdown
    assert "录用后需缴纳培训费" in markdown
    assert "是否必须先缴纳费用才能入职？" in markdown
    assert state["rewrite_skipped_reason"] in markdown
    assert "总体匹配度" in markdown
    assert "Python 接口设计" in markdown
    assert "### 简历改写建议" not in markdown
    assert "运行错误" not in markdown
