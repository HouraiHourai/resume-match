"""配置管理。所有 API key / model 从环境变量读，不硬编码。

多模型支持:
- 主模型: DeepSeek V4 Flash (deepseek-v4-flash)
- 备用模型: 通义千问 3.8 Max (qwen3.8-max)
- 配置驱动: 通过 MODEL_CONFIGS 统一管理，添加新模型只需加配置
"""
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# 多模型配置字典
MODEL_CONFIGS = {
    "deepseek": {
        "model": "deepseek-v4-flash",  # DeepSeek V4 Flash (最新版本)
        "api_key_env": "DEEPSEEK_API_KEY",
        "base_url": "https://api.deepseek.com",
        "max_tokens": 8192,
    },
    "qwen": {
        "model": "qwen3.8-max",  # Qwen 3.8 Max (最新版本)
        "api_key_env": "DASHSCOPE_API_KEY",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens": 8000,
    },
}

# 默认模型（可通过环境变量 DEFAULT_MODEL 修改）
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "deepseek")


def get_llm(model_name: str | None = None, temperature: float = 0.3) -> ChatOpenAI:
    """返回配置好的 LLM 客户端（支持多模型）。

    Args:
        model_name: 模型名称 (deepseek/qwen)，None 时用 DEFAULT_MODEL
        temperature: 温度参数，0.3 适合抽取/分析任务（稳定、可复现）

    Raises:
        ValueError: 未知的模型名称
        RuntimeError: 对应的 API key 未设置
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in MODEL_CONFIGS:
        available = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Unknown model: {model_name}. Available: {available}")

    config = MODEL_CONFIGS[model_name]

    # API Key 检查
    api_key = os.getenv(config["api_key_env"])
    if not api_key:
        raise RuntimeError(
            f"{config['api_key_env']} not set. Add it to .env file.\n"
            f"Example: {config['api_key_env']}=your_api_key_here"
        )

    # DeepSeek V4 需要显式禁用思考模式，否则与 function_calling 冲突
    extra_body = None
    if model_name == "deepseek":
        extra_body = {"thinking": {"type": "disabled"}}
    elif model_name == "qwen":
        # DashScope thinking mode rejects required/object tool_choice used
        # by structured function calling. Verified against qwen3.8-max.
        extra_body = {"enable_thinking": False}

    return ChatOpenAI(
        model=config["model"],
        api_key=api_key,
        base_url=config["base_url"],
        temperature=temperature,
        max_tokens=config["max_tokens"],
        extra_body=extra_body,
    )


def get_deepseek_model(temperature: float = 0.3, max_tokens: int = 8192) -> ChatOpenAI:
    """返回配置好的 DeepSeek 客户端（向后兼容）。

    新代码建议直接用 get_llm()，更灵活。
    """
    return get_llm("deepseek", temperature)
