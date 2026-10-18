"""JD 结构化服务：从 JD 原文抽取结构化需求。

关键决策:
- 使用 with_structured_output：让 LangChain 把 Pydantic schema 转成 function-calling 格式
- 模型输出会被强制校验成 JDRequirements 实例，不符合会自动 retry
- 错误只往 state['errors'] append，不 raise，让整张图跑完由上层决定如何呈现
"""
from resume_match.config import get_deepseek_model
from resume_match.services.structured_llm import ainvoke_structured_with_retry
from resume_match.prompts.templates import EXTRACT_JD_PROMPT
from resume_match.state import AgentState, JDRequirements


async def extract(state: AgentState) -> dict:
    """从 JD 原文抽取结构化需求。"""
    if not state.get("jd_raw_text"):
        return {"errors": ["extract: no jd_raw_text to process"]}

    model = get_deepseek_model()
    structured_model = model.with_structured_output(JDRequirements, method="function_calling")

    prompt = EXTRACT_JD_PROMPT.format(jd_text=state["jd_raw_text"])
    try:
        result = await ainvoke_structured_with_retry(
            structured_model,
            prompt,
            fallback_model="qwen",
            schema_class=JDRequirements,
        )
        # 如果用户通过 [公司: XXX] 指定了公司名，覆盖 LLM 提取的结果
        if state.get("company_name"):
            result.company = state["company_name"]
        return {"jd_structured": result}
    except Exception as e:
        return {"errors": [f"extract failed: {e}"]}
