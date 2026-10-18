"""匹配分析服务：对比 JD 需求与简历原文，输出差距分析。

关键决策:
- 输入用 state['jd_structured'] 而不是原文: 结构化后字段清晰，模型对比时更聚焦
- 简历用 raw_text: 简历格式多样，让模型直接读全文去找证据更灵活
- 使用 function_calling 方法进行结构化输出
"""
from resume_match.config import get_deepseek_model
from resume_match.services.structured_llm import ainvoke_structured_with_retry
from resume_match.prompts.templates import ANALYZE_PROMPT
from resume_match.state import AgentState, GapAnalysis


async def analyze(state: AgentState) -> dict:
    """对比 JD 需求与简历，输出差距分析。"""
    if not state.get("jd_structured") or not state.get("resume_raw_text"):
        return {"errors": ["analyze: missing prerequisites (jd_structured or resume_raw_text)"]}

    model = get_deepseek_model()
    structured_model = model.with_structured_output(GapAnalysis, method="function_calling")

    prompt = ANALYZE_PROMPT.format(
        jd=state["jd_structured"].model_dump_json(indent=2),
        resume=state["resume_raw_text"],
    )
    try:
        result = await ainvoke_structured_with_retry(
            structured_model,
            prompt,
            fallback_model="qwen",
            schema_class=GapAnalysis,
        )
        return {"gap_analysis": result}
    except Exception as e:
        return {"errors": [f"analyze failed: {e}"]}
