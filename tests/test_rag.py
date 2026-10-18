"""离线检索回归：真实临时 Chroma/BM25 + 固定向量，无旧的注入参数。"""
from unittest.mock import MagicMock

import pytest

from resume_match.rag.vector_store import VectorStore
from resume_match.rag.bm25_store import BM25Store
from resume_match.rag import retriever, coarse_ranker
from resume_match.rag.fine_ranker import fine_rank, reciprocal_rank_scores
from resume_match.state import JDRequirements


def _jd(company=None):
    return JDRequirements(
        role_title="AI Agent 工程师", company=company, seniority="intern",
        hard_skills=["Python", "LangGraph"], key_responsibilities=["搭 Agent"],
    )


def _meta(hid, company=None):
    return {
        "history_id": hid, "jd_summary": f"岗位{hid}", "match_score": 80,
        "should_apply": "worth_trying", "created_at": "2026-09-17",
        "company": company or "", "role_title": "AI Agent 工程师",
        "hard_skills": "Python,LangGraph", "seniority": "intern",
    }


@pytest.fixture
def stores(tmp_path, monkeypatch):
    vector = VectorStore(persist_dir=tmp_path / "chroma")
    bm25 = BM25Store(index_path=tmp_path / "bm25.pkl")
    for module in (retriever, coarse_ranker):
        monkeypatch.setattr(module, "get_vector_store", lambda: vector)
        monkeypatch.setattr(module, "get_bm25_store", lambda: bm25)
    monkeypatch.setattr(coarse_ranker, "embed_text", lambda text: [1.0, 0.0])
    return vector, bm25


def test_add_query_delete(stores):
    store, _ = stores
    store.add("1", "A", _meta(1), [1.0, 0.0])
    store.add("2", "B", _meta(2), [0.0, 1.0])
    assert store.query([1.0, 0.0], top_k=2)["ids"][0][0] == "1"
    store.delete("2")
    assert store.existing_ids() == {"1"}


def test_retriever_threshold_and_none_company(stores):
    store, _ = stores
    store.add("1", "A", _meta(1), [1.0, 0.0])
    store.add("2", "B", _meta(2), [0.0, 1.0])
    result = retriever.find_similar(_jd())
    assert [r.history_id for r in result] == [1]


def test_retriever_excludes_self(stores):
    store, _ = stores
    store.add("1", "A", _meta(1), [1.0, 0.0])
    assert retriever.find_similar(_jd(), exclude_id=1) == []


def test_retriever_empty_store_returns_empty(stores):
    assert retriever.find_similar(_jd()) == []


def test_known_same_company_and_title_are_filtered(stores):
    store, _ = stores
    store.add("1", "A", _meta(1, "ACME"), [1.0, 0.0])
    assert retriever.find_similar(_jd("acme")) == []


@pytest.mark.parametrize("vector_failure", [False, True])
def test_bm25_only_survives_empty_or_failed_vector(stores, monkeypatch, vector_failure):
    vector, bm25 = stores
    if vector_failure:
        monkeypatch.setattr(vector, "count", MagicMock(side_effect=RuntimeError("offline")))
    monkeypatch.setattr(bm25, "count", lambda: 1)
    monkeypatch.setattr(bm25, "query", lambda *a, **k: [
        {"history_id": 1, "bm25_score": 3.0, "metadata": _meta(1)},
    ])
    result = retriever.find_similar(_jd())
    assert [r.history_id for r in result] == [1]
    assert result[0].match_score == 80


def test_vector_survives_bm25_failure(stores, monkeypatch):
    vector, bm25 = stores
    vector.add("1", "A", _meta(1), [1.0, 0.0])
    monkeypatch.setattr(bm25, "query", MagicMock(side_effect=RuntimeError("offline")))
    monkeypatch.setattr(bm25, "count", lambda: 1)
    assert retriever.find_similar(_jd())[0].history_id == 1


def _candidate(hid, vector, bm25):
    return {"history_id": hid, "vector_similarity": vector, "bm25_score": bm25,
            "metadata": _meta(hid), "hard_skills": ["Python", "LangGraph"],
            "seniority": "intern", "company": None, "domain": None}


def test_bm25_score_changes_order_and_can_win_top_one():
    candidates = [_candidate(1, .9, 0), _candidate(2, 0, 10)]
    # 单路第1名贡献相同，按ID稳定打破并列。
    assert fine_rank(_jd(), candidates, top_k=1)[0].history_id == 1
    candidates[0]["bm25_score"] = 10
    candidates[1]["vector_similarity"] = .9
    candidates[1]["bm25_score"] = 20
    assert fine_rank(_jd(), candidates, top_k=1)[0].history_id == 2
    # 只由 BM25 召回、技能与资历相关，也能单独进入结果。
    assert fine_rank(_jd(), [_candidate(2, 0, 20)], top_k=1)[0].history_id == 2


def test_rrf_bounded_and_ties_are_equal():
    candidates = [_candidate(1, .8, 5), _candidate(2, .8, 5)]
    assert reciprocal_rank_scores(candidates) == {1: 1.0, 2: 1.0}
    assert fine_rank(_jd(), candidates, top_k=0) == []


def test_bm25_only_candidate_beats_semantically_similar_wrong_skills():
    candidates = [_candidate(i, .95, 0) for i in (1, 2, 3)]
    for candidate in candidates:
        candidate["hard_skills"] = ["Photoshop"]
    candidates.append(_candidate(4, 0, 20))
    assert fine_rank(_jd(), candidates, top_k=3)[0].history_id == 4


def test_merge_deduplicates_id_and_keeps_both_scores(stores, monkeypatch):
    vector, bm25 = stores
    vector.add("1", "A", _meta(1), [1.0, 0.0])
    monkeypatch.setattr(bm25, "count", lambda: 1)
    monkeypatch.setattr(bm25, "query", lambda *a, **k: [
        {"history_id": 1, "bm25_score": 3, "metadata": _meta(1)},
    ])
    result = coarse_ranker.coarse_rank(_jd())
    assert len(result) == 1
    assert result[0]["bm25_score"] == 3
    assert result[0]["vector_similarity"] > .99
