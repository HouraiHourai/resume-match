"""投递建议服务：基于 gap 分析与相似历史投递生成建议。

关键决策:
- 三个输入全给 (jd_structured + gap_analysis + resume_raw_text): advise 是综合决策服务
- RAG: LLM 调用前先检索相似历史投递，注入 prompt，让建议参考用户历史模式
- 检索失败/为空都不阻断 advise，降级为"无历史上下文"继续
- 检索结果也塞进 state['similar_histories']，给前端展示
- 使用 function_calling 方法进行结构化输出
"""
from resume_match.config import get_deepseek_model
from resume_match.services.structured_llm import ainvoke_structured_with_retry
from resume_match.prompts.templates import ADVISE_PROMPT
from resume_match.rag.retriever import find_similar, format_similar_histories_for_prompt
from resume_match.state import AgentState, ApplicationAdvice, ResumeAdvice


async def advise(state: AgentState) -> dict:
    """生成建议；只有 high 使用包含改写字段的模型输出结构。"""
    if not state.get("gap_analysis") or not state.get("jd_structured"):
        return {"errors": ["advise: missing prerequisites (gap_analysis or jd_structured)"]}
    quality = state.get("jd_quality")
    if not quality:
        return {"errors": ["advise: missing prerequisite (jd_quality)"]}
    allow_rewrites = quality.level == "high"
    schema = ResumeAdvice if allow_rewrites else ApplicationAdvice
    rewrite_instructions = (
        "Suggest 3-5 bullet_rewrites using relevant resume bullets and JD keywords. "
        "Only rephrase real experience; do not invent achievements. Include a rationale for each rewrite."
        if allow_rewrites else
        "Skip all resume rewriting. Do not produce bullet_rewrites or rewritten resume text in any field. "
        "Continue generating application advice, preparation topics and questions to verify the job conditions."
    )
    rewrite_skipped_reason = None if allow_rewrites else (
        f"{'岗位质量较低' if quality.level == 'low' else '岗位信息待核实'}，已跳过简历改写。"
    )

    # RAG 检索（失败降级为空，不阻断）
    try:
        current_jd_text = state.get("jd_input", "")
        similar = find_similar(
            state["jd_structured"],
            top_k=3,
            current_jd_text=current_jd_text
        )
    except Exception:
        similar = []

    model = get_deepseek_model()
    structured_model = model.with_structured_output(schema, method="function_calling")

    prompt = ADVISE_PROMPT.format(
        jd=state["jd_structured"].model_dump_json(indent=2),
        gap=state["gap_analysis"].model_dump_json(indent=2),
        resume=state.get("resume_raw_text", ""),
        similar_histories_context=format_similar_histories_for_prompt(similar),
        jd_quality=quality.model_dump_json(indent=2),
        rewrite_instructions=rewrite_instructions,
    )
    try:
        result = await ainvoke_structured_with_retry(
            structured_model,
            prompt,
            fallback_model="qwen",
            schema_class=schema,
        )
        if not allow_rewrites:
            # 即使模型带回额外字段，也只保留建议结构中的字段，改写固定为空。
            advice_only = ApplicationAdvice.model_validate(result.model_dump())
            result = ResumeAdvice(**advice_only.model_dump(), bullet_rewrites=[])
        return {"advice": result, "similar_histories": similar,
                "rewrite_skipped_reason": rewrite_skipped_reason}
    except Exception as e:
        return {"errors": [f"advise failed: {e}"], "similar_histories": similar}
