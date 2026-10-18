"""离线回归默认禁止真实 HTTP；MockTransport 仍可验证请求序列化。"""
import os

import httpx
import pytest


@pytest.fixture(autouse=True)
def offline_http(monkeypatch):
    if os.getenv("RUN_E2E") == "1":
        return

    def blocked(*args, **kwargs):
        raise RuntimeError("External HTTP disabled in offline tests")

    async def async_blocked(*args, **kwargs):
        raise RuntimeError("External HTTP disabled in offline tests")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", async_blocked)
    # 普通节点测试不应尝试构造真实备用模型；专门的降级测试可自行覆盖。
    from resume_match.services import structured_llm
    monkeypatch.setattr(structured_llm, "get_llm", blocked)

