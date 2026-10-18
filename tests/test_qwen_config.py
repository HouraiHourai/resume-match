def test_qwen_structured_client_disables_thinking(monkeypatch):
    from resume_match.config import get_llm
    monkeypatch.setenv('DASHSCOPE_API_KEY', 'test-not-real')
    model = get_llm('qwen')
    assert model.extra_body == {'enable_thinking': False}
