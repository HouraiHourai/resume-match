"""使用真实 LangChain/OpenAI 客户端 + HTTP MockTransport 验证最终请求体。"""
import json

import httpx
from langchain_openai import OpenAIEmbeddings

from resume_match.rag import embedder


def test_bge_request_uses_original_strings(monkeypatch):
    captured = []

    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "object": "list", "model": "Pro/BAAI/bge-m3",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.2, 0.8]}],
            "usage": {"prompt_tokens": 2, "total_tokens": 2},
        })

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        def factory(**kwargs):
            return OpenAIEmbeddings(**kwargs, http_client=client, max_retries=0)

        monkeypatch.setenv("SILICONFLOW_API_KEY", "test-only")
        monkeypatch.delenv("SILICONFLOW_BASE_URL", raising=False)
        monkeypatch.setattr(embedder, "_model", None)
        monkeypatch.setattr(embedder, "OpenAIEmbeddings", factory)
        text = "Python / LangGraph 岗位要求"
        assert embedder.embed_text(text) == [0.2, 0.8]
        assert captured[0]["input"] == [text]
        assert captured[0]["model"] == "Pro/BAAI/bge-m3"
