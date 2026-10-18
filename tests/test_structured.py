"""ainvoke_structured_with_retry 的单元测试。base_delay=0 保持测试快。"""
from unittest.mock import AsyncMock, MagicMock, call

import pytest

from resume_match.services.structured_llm import ainvoke_structured_with_retry
from resume_match.services import structured_llm


@pytest.mark.asyncio
async def test_succeeds_first_try():
    sm = MagicMock()
    sm.ainvoke = AsyncMock(return_value="ok")
    result = await ainvoke_structured_with_retry(sm, "p", base_delay=0)
    assert result == "ok"
    assert sm.ainvoke.await_count == 1


@pytest.mark.asyncio
async def test_retries_on_none_then_succeeds():
    sm = MagicMock()
    sm.ainvoke = AsyncMock(side_effect=[None, "ok"])   # 第一次 None, 第二次成功
    result = await ainvoke_structured_with_retry(sm, "p", attempts=3, base_delay=0)
    assert result == "ok"
    assert sm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_retries_on_exception_then_succeeds():
    sm = MagicMock()
    sm.ainvoke = AsyncMock(side_effect=[RuntimeError("transient"), "ok"])
    result = await ainvoke_structured_with_retry(sm, "p", attempts=3, base_delay=0)
    assert result == "ok"
    assert sm.ainvoke.await_count == 2


@pytest.mark.asyncio
async def test_exhausts_attempts_and_raises():
    sm = MagicMock()
    sm.ainvoke = AsyncMock(return_value=None)   # 一直 None
    with pytest.raises(RuntimeError, match="after 3 attempts"):
        await ainvoke_structured_with_retry(sm, "p", attempts=3, base_delay=0)
    assert sm.ainvoke.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [None, RuntimeError("transient")])
async def test_primary_exponential_delays_and_no_sleep_after_exhaustion(monkeypatch, failure):
    sleep = AsyncMock()
    monkeypatch.setattr(structured_llm.asyncio, "sleep", sleep)
    sm = MagicMock()
    sm.ainvoke = AsyncMock(side_effect=[failure] * 5)
    with pytest.raises(RuntimeError):
        await ainvoke_structured_with_retry(sm, "p", attempts=5, fallback_model=None)
    assert sm.ainvoke.await_count == 5
    assert sleep.await_args_list == [call(.5), call(1.0), call(2.0), call(4.0)]


@pytest.mark.asyncio
async def test_fallback_restarts_exponential_delay_and_stops_on_success(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(structured_llm.asyncio, "sleep", sleep)
    primary = MagicMock()
    primary.ainvoke = AsyncMock(return_value=None)
    fallback = MagicMock()
    fallback.ainvoke = AsyncMock(side_effect=[None, None, None, "ok"])
    llm = MagicMock()
    llm.with_structured_output.return_value = fallback
    monkeypatch.setattr(structured_llm, "get_llm", lambda name: llm)
    result = await ainvoke_structured_with_retry(
        primary, "p", attempts=5, base_delay=.25, schema_class=object,
    )
    assert result == "ok"
    assert fallback.ainvoke.await_count == 4
    assert sleep.await_args_list == [
        call(.25), call(.5), call(1.0), call(2.0),
        call(.25), call(.5), call(1.0),
    ]


@pytest.mark.asyncio
async def test_first_attempt_success_does_not_sleep(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(structured_llm.asyncio, "sleep", sleep)
    sm = MagicMock()
    sm.ainvoke = AsyncMock(return_value="ok")
    assert await ainvoke_structured_with_retry(sm, "p") == "ok"
    sleep.assert_not_awaited()