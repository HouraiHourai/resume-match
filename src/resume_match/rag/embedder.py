"""向量嵌入模型：BGE-M3 (通过 SiliconFlow API)。

选型: Pro/BAAI/bge-m3
- 支持中英文多语言，适合 JD 中英混杂场景
- 通过 SiliconFlow API 调用，无需本地部署
- 用于 RAG 系统检索相似历史记录
"""
import os
import hashlib
import json
from pathlib import Path
from threading import RLock
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from typing import Optional
from resume_match.rag.embedding_cache import EmbeddingCache

# 加载 .env 文件
env_path = Path(__file__).parent.parent.parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)

_model = None
_model_lock = RLock()
_cache = EmbeddingCache(
    max_entries=int(os.getenv("EMBEDDING_CACHE_MAX_ENTRIES", "256")),
    ttl_seconds=float(os.getenv("EMBEDDING_CACHE_TTL_SECONDS", "3600")),
)

def _get_model():
    """懒加载嵌入模型（首次调用时初始化）。"""
    global _model
    with _model_lock:
        if _model is None:
            _model = OpenAIEmbeddings(
                model="Pro/BAAI/bge-m3",
                base_url=os.getenv("SILICONFLOW_BASE_URL") or "https://api.siliconflow.cn/v1",
                api_key=os.getenv("SILICONFLOW_API_KEY"),
                # BGE 接口接收原始文本，不能发送 OpenAI tokenizer 的 token ID。
                check_embedding_ctx_length=False,
            )
    return _model


def embed_text(text: str) -> list[float]:
    """精确输入复用BGE向量；LRU/TTL有界缓存，失败不缓存。"""
    model = _get_model()
    return _cache.get_or_compute(_cache_key(model, text), lambda: model.embed_query(text))


def _cache_key(model, text: str) -> str:
    # 缓存按客户端实际配置隔离；凭据和 JD 文本仅以摘要参与缓存键计算。
    credential = getattr(model, "openai_api_key", None)
    if hasattr(credential, "get_secret_value"):
        credential = credential.get_secret_value()
    config = {
        "schema": "exact-embedding-input-v1", "model": model.model,
        "base_url": str(getattr(model, "openai_api_base", "")),
        "dimensions": getattr(model, "dimensions", None),
        "check_embedding_ctx_length": getattr(model, "check_embedding_ctx_length", None),
        "account": hashlib.sha256(str(credential).encode()).hexdigest(),
        "model_kwargs": getattr(model, "model_kwargs", {}),
        "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def clear_embedding_cache():
    _cache.clear()


def embedding_cache_info():
    return _cache.info()


def build_embedding_text(
    role_title: str,
    hard_skills: list[str],
    domain: Optional[str],
    seniority: str,
) -> str:
    """生成用于向量嵌入的结构化文本。

    使用结构化字段而非原文，提高检索准确性。
    """
    return "\n".join([
        role_title,
        "Skills required: " + ", ".join(hard_skills),
        "Domain: " + (domain or "general"),
        "Seniority: " + seniority,
    ])


def build_embedding_text_from_dict(jd: dict) -> str:
    """从 full_result['jd'] 这样的 dict 生成 embedding 文本。"""
    return build_embedding_text(
        role_title=jd.get("role_title", ""),
        hard_skills=jd.get("hard_skills", []) or [],
        domain=jd.get("domain"),
        seniority=jd.get("seniority", "unspecified"),
    )
