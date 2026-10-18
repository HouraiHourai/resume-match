"""结构化 LLM 调用的有限重试封装 + 多模型自动降级。

背景:
- LLM 的 function_calling 偶发不返回有效结构，导致 with_structured_output 返回 None
- 短时高频调用可能被限流
- 单次调用不稳定会导致整个分析流程失败

设计:
- 有限次重试（默认 3 次），失败原因（None 或异常）都触发重试
- 指数退避等待（0.5s, 1.0s, 2.0s...），仅在还有下一次尝试时等待
- 多模型降级: 主模型（deepseek-v4-flash）重试失败后，自动切换到备用模型（qwen3.8-max）
- 全部失败后 raise，由调用方转成 state['errors']

多模型降级的价值:
- 主模型便宜但偶有不稳定，备用模型稳定但成本稍高
- 主模型失败时尝试备用模型；调用比例和费用需要实际统计
- 兼顾成本和稳定性
"""
import asyncio

from resume_match.config import get_llm


async def ainvoke_structured_with_retry(
    structured_model,
    prompt: str,
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    fallback_model: str | None = "qwen",
    schema_class=None,
):
    """调用 structured_model.ainvoke(prompt)，对 None 结果和异常做有限次重试+降级。

    Args:
        structured_model: 主模型的 with_structured_output 包装后的实例
        prompt: 输入 prompt
        attempts: 每个模型的最大尝试次数（包含首次调用）
        base_delay: 首次重试等待秒数；第 i 次失败后等待 base_delay * 2**i（i 从 0 开始）
        fallback_model: 主模型失败后的降级模型名称（如 "qwen"），None 表示不降级
        schema_class: Pydantic schema（用于降级时重建 structured_model）

    流程:
        1. 用主模型重试 attempts 次
        2. 如果失败且有 fallback_model，切换到备用模型再重试 attempts 次
        3. 都失败则 raise
    """
    last_reason = "unknown error"

    # 主模型重试
    for i in range(attempts):
        try:
            result = await structured_model.ainvoke(prompt)
            if result is not None:
                return result
            last_reason = "model returned no structured output (possibly truncated)"
        except Exception as e:
            last_reason = str(e)
        if i < attempts - 1:
            await asyncio.sleep(base_delay * (2 ** i))

    primary_reason = last_reason
    # 如果主模型失败且配置了降级，切换到备用模型
    if fallback_model and schema_class:
        print(f"[Fallback] Primary model failed after {attempts} attempts, switching to {fallback_model}")
        try:
            fallback_llm = get_llm(fallback_model)
            fallback_structured = fallback_llm.with_structured_output(schema_class, method="function_calling")

            for i in range(attempts):
                try:
                    result = await fallback_structured.ainvoke(prompt)
                    if result is not None:
                        print(f"[Fallback] Success with {fallback_model}")
                        return result
                    last_reason = f"{fallback_model} returned no structured output"
                except Exception as e:
                    last_reason = f"{fallback_model} error: {str(e)}"
                if i < attempts - 1:
                    await asyncio.sleep(base_delay * (2 ** i))
        except Exception as e:
            last_reason = f"fallback to {fallback_model} failed: {str(e)}"

    # 所有尝试都失败
    raise RuntimeError(
        f"after {attempts} attempts on primary model and fallback: "
        f"primary: {primary_reason}; last failure: {last_reason}"
    )
