"""API 路由定义。复用现有的 PDF 解析适配器和 graph, 不重写核心逻辑。"""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from resume_match.graph import graph
from resume_match.adapters.pdf_parser import extract_pdf_text
from resume_match.api.schemas import AnalyzeResponse, HistoryListItem, HistoryDetail
from resume_match.history.models import HistoryEntry
from resume_match.history.storage import get_storage
from resume_match.history.summary import generate_summary

router = APIRouter()

# ReAct 图里的节点名（用于调试）
_NODE_NAMES = {"agent_think", "execute_tool"}

# 工具名称到前端步骤的映射（ReAct 架构下，工具 = 前端的步骤）
_TOOL_TO_STEP = {
    "extract_jd": "extract",
    "check_company": None,  # 不显示在前端进度条
    "analyze_gap": "analyze",
    "generate_advice": "advise",
}


async def _resume_text_from_upload(resume_file: UploadFile) -> str:
    """把上传的 PDF 存到临时文件、解析成文本、随即删除(不持久化, 隐私红线 3.4)。"""
    filename = resume_file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail={"type": "invalid_file", "message": "resume_file 必须是 PDF"})
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(await resume_file.read())
        tmp.flush()
        tmp.close()
        try:
            return extract_pdf_text(Path(tmp.name))
        except Exception as e:
            raise HTTPException(status_code=400, detail={"type": "pdf_parse_error", "message": str(e)})
    finally:
        os.unlink(tmp.name)  # 处理完立即删掉, 不留用户简历


def _assemble_report(final_state: dict) -> dict:
    """把图的最终 state 组装成对外报告(和 CLI / sample_output.json 一致)。"""
    return {
        "jd": final_state["jd_structured"].model_dump() if final_state.get("jd_structured") else None,
        "gap": final_state["gap_analysis"].model_dump() if final_state.get("gap_analysis") else None,
        "advice": final_state["advice"].model_dump() if final_state.get("advice") else None,
        "company_risk": final_state.get("company_risk"),
        "errors": final_state.get("errors", []),
        "similar_histories": [s.model_dump() for s in final_state.get("similar_histories", [])],
    }


def _sse(event_type: str, data: dict) -> str:
    """格式化一条 SSE 消息: event: <type>\\n data: <json>\\n\\n。"""
    return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _save_history(final_state: dict, report: dict, jd_text: str, resume_filename: str):
    """分析完成后存档到 SQLite。返回 history_id; 若 JD 未成功提取则不存, 返回 None。"""
    jd_obj = final_state.get("jd_structured")
    if not jd_obj:
        return None
    entry = HistoryEntry(
        created_at=datetime.now().isoformat(),
        jd_text=jd_text,
        jd_summary=generate_summary(jd_obj),
        resume_filename=resume_filename,
        match_score=(report.get("gap") or {}).get("overall_match_score"),
        should_apply=(report.get("advice") or {}).get("should_apply"),
        full_result=report,
    )
    return get_storage().create_entry(entry)


@router.get("/health")
async def health() -> dict:
    """健康检查。"""
    return {"status": "ok"}

# 用户提交 JD 和简历，等整张图跑完，一次性返回完整结果。
@router.post("/api/analyze", response_model=AnalyzeResponse)
async def analyze_endpoint(
    jd_text: str = Form(..., description="JD 原文(前端先自行抓取)"),
    resume_file: UploadFile = File(..., description="简历 PDF"),
) -> AnalyzeResponse:
    """一次性分析: 跑完整张图, 返回和 CLI 相同形状的 JSON。"""
    if len(jd_text.strip()) < 1:
        raise HTTPException(status_code=400, detail={"type": "invalid_input", "message": "jd_text 不能为空"})

    resume_filename = resume_file.filename or "resume.pdf"
    resume_text = await _resume_text_from_upload(resume_file)   # 存临时文件 → 解析 PDF → 删临时文件
    initial_state = {"jd_input": jd_text, "resume_raw_text": resume_text}

    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception as e:  # 意外崩溃 → 500
        raise HTTPException(status_code=500, detail={"type": "internal_error", "message": str(e)})

    report = _assemble_report(final_state)
    history_id = _save_history(final_state, report, jd_text, resume_filename)

    # 图正常跑完即 200; 节点级的软错误(如某节点失败)随 errors 数组透出, 和 CLI 行为一致。
    return AnalyzeResponse(
        jd=final_state.get("jd_structured"),
        gap=final_state.get("gap_analysis"),
        advice=final_state.get("advice"),
        company_risk=final_state.get("company_risk"),
        errors=final_state.get("errors", []),
        history_id=history_id,
        similar_histories=final_state.get("similar_histories", []),
    )

# 流式分析，用 SSE 推进度。
@router.post("/api/analyze/stream")
async def analyze_stream_endpoint(
    jd_text: str = Form(..., description="JD 原文"),
    resume_file: UploadFile = File(..., description="简历 PDF"),
) -> StreamingResponse:
    """流式分析: 用 astream_events 实时推送每个节点的 start/end, 最后推 final_result。

    事件类型: node_start / node_end / final_result / error。
    """
    # 保证输入非空
    if len(jd_text.strip()) < 1:
        raise HTTPException(status_code=400, detail={"type": "invalid_input", "message": "jd_text 不能为空"})

    # PDF 解析放在开始流式之前, 这样解析错误能以 400 立刻返回, 而不是塞进事件流。
    resume_filename = resume_file.filename or "resume.pdf"
    resume_text = await _resume_text_from_upload(resume_file)
    initial_state = {"jd_input": jd_text, "resume_raw_text": resume_text}

    async def event_generator():
        final_output: dict = {}
        current_tool: str | None = None
        fetch_completed = False
        has_errors = False

        try:
            yield _sse("node_start", {"node": "fetch_jd"})

            # 将工具调用事件映射为前端进度步骤。
            async for ev in graph.astream_events(initial_state, version="v2"):
                kind = ev["event"]
                name = ev.get("name")
                data = ev.get("data", {})

                if name == "execute_tool":
                    if kind == "on_chain_start":
                        input_state = data.get("input", {})
                        tool_name = input_state.get("next_action")

                        # fetch_jd 不是独立节点，它在 extract_jd 工具内部被调用。astream_events 捕获不到它。
                        # 当 Agent 调用第一个工具（extract_jd）时，标记"抓取"完成
                        if not fetch_completed and tool_name == "extract_jd":
                            yield _sse("node_end", {"node": "fetch_jd"})
                            fetch_completed = True

                        if tool_name and tool_name in _TOOL_TO_STEP:
                            step_name = _TOOL_TO_STEP[tool_name]
                            if step_name:  # check_company 为 None，不发送事件
                                current_tool = step_name
                                yield _sse("node_start", {"node": step_name})

                    elif kind == "on_chain_end":
                        output_state = data.get("output", {})

                        if current_tool and output_state.get("errors"):
                            has_errors = True
                            yield _sse("error", {"message": output_state["errors"][-1]})

                        if current_tool:
                            yield _sse("node_end", {"node": current_tool})
                            current_tool = None

                # 最外层图(没有 langgraph_node 元数据)的 on_chain_end 带着最终 state
                meta = ev.get("metadata", {}) or {}
                if kind == "on_chain_end" and not meta.get("langgraph_node"):
                    output = data.get("output")
                    if isinstance(output, dict):
                        final_output = output

            report = _assemble_report(final_output)
            # 仅存档含分析结果且未收到工具错误事件的报告。
            has_valid_data = report.get("gap") or report.get("advice")
            if has_valid_data and not has_errors:
                report["history_id"] = _save_history(final_output, report, jd_text, resume_filename)
            else:
                report["history_id"] = None
            yield _sse("final_result", report)
        except Exception as e:
            yield _sse("error", {"type": "internal_error", "message": str(e)})

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------- History ----------------

@router.get("/api/history", response_model=list[HistoryListItem])
async def list_history() -> list[HistoryListItem]:
    """历史列表(倒序, 不含 full_result)。"""
    entries = get_storage().list_entries()
    return [
        HistoryListItem(
            id=e.id,
            created_at=e.created_at,
            jd_summary=e.jd_summary,
            match_score=e.match_score,
            should_apply=e.should_apply,
        )
        for e in entries
    ]


@router.get("/api/history/{entry_id}", response_model=HistoryDetail)
async def get_history(entry_id: int) -> HistoryDetail:
    """单条历史完整数据; 不存在 → 404。"""
    entry = get_storage().get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail={"type": "not_found", "message": f"history {entry_id} 不存在"})
    return HistoryDetail(
        id=entry.id,
        created_at=entry.created_at,
        jd_text=entry.jd_text,
        jd_summary=entry.jd_summary,
        resume_filename=entry.resume_filename,
        match_score=entry.match_score,
        should_apply=entry.should_apply,
        full_result=entry.full_result,
    )


@router.delete("/api/history/{entry_id}")
async def delete_history(entry_id: int) -> dict:
    """删除一条历史; 不存在 → 404。"""
    if not get_storage().delete_entry(entry_id):
        raise HTTPException(status_code=404, detail={"type": "not_found", "message": f"history {entry_id} 不存在"})
    return {"deleted": True}
