from unittest.mock import MagicMock

import pytest

from resume_match.history.models import HistoryEntry
from resume_match.history.storage import HistoryStorage
from resume_match.rag.bm25_store import BM25Store, _tokenize


def entry(title="Agent", skills=None):
    return HistoryEntry(
        created_at="2026-09-17", jd_text=title, jd_summary=title,
        resume_filename="test.pdf", match_score=80, should_apply="worth_trying",
        full_result={"jd": {
            "role_title": title, "hard_skills": skills or [title],
            "seniority": "intern", "company": None,
        }},
    )


def test_split_slash_preserves_technical_punctuation():
    assert _tokenize("C/C++，LangChain/LangGraph；C# Node.js") == [
        "c", "c++", "langchain", "langgraph", "c#", "node.js",
    ]


def test_normal_create_delete_updates_bm25_even_when_vector_fails(tmp_path):
    bm25 = BM25Store(tmp_path / "bm25.pkl")
    vector = MagicMock()
    vector.index_entry.side_effect = RuntimeError("embedding unavailable")
    storage = HistoryStorage(tmp_path / "history.db", vector_store=vector, bm25_store=bm25)
    target = storage.create_entry(entry("C++", ["C/C++"]))
    for title in ["Java", "Rust", "Go"]:
        storage.create_entry(entry(title))
    result = bm25.query("C++")
    assert [r["history_id"] for r in result] == [target]
    assert result[0]["metadata"]["match_score"] == 80
    assert result[0]["metadata"]["created_at"] == "2026-09-17"
    vector.remove.side_effect = RuntimeError("vector offline")
    assert storage.delete_entry(target)
    assert target not in bm25.doc_ids
    assert bm25.query("C++") == []
    assert target not in BM25Store(tmp_path / "bm25.pkl").doc_ids


def test_full_rebuild_removes_zombies_and_updates_tokenization(tmp_path):
    storage = HistoryStorage(tmp_path / "history.db")
    bm25 = BM25Store(tmp_path / "bm25.pkl")
    old = entry("deleted")
    old.id = 999
    bm25.add(old)
    hid = storage.create_entry(entry("LangChain/LangGraph"))
    assert bm25.replace_all(storage.iter_entries()) == 1
    assert bm25.doc_ids == [hid]
    assert "langgraph" in bm25.corpus[0]
    storage.delete_entry(hid)
    assert bm25.replace_all(storage.iter_entries()) == 0
    assert BM25Store(tmp_path / "bm25.pkl").count() == 0


def test_rebuild_failure_does_not_replace_old_snapshot(tmp_path, monkeypatch):
    bm25 = BM25Store(tmp_path / "bm25.pkl")
    old = entry("old")
    old.id = 1
    bm25.add(old)
    fresh = entry("new")
    fresh.id = 2
    monkeypatch.setattr(bm25, "_save", MagicMock(side_effect=OSError("disk full")))
    with pytest.raises(OSError):
        bm25.replace_all([fresh])
    assert bm25.doc_ids == [1]
    assert BM25Store(tmp_path / "bm25.pkl").doc_ids == [1]


def test_default_storage_repairs_old_index_on_initialization(tmp_path, monkeypatch):
    from resume_match.history import storage as storage_module
    from resume_match.rag import bm25_store, vector_store

    db_path = tmp_path / "history.db"
    standalone = HistoryStorage(db_path)
    hid = standalone.create_entry(entry("LangChain/LangGraph"))
    bm25 = BM25Store(tmp_path / "bm25.pkl")
    zombie = entry("deleted")
    zombie.id = 999
    bm25.add(zombie)
    monkeypatch.setenv("SQLITE_PATH", str(db_path))
    monkeypatch.setattr(storage_module, "_storage", None)
    monkeypatch.setattr(bm25_store, "get_bm25_store", lambda: bm25)
    monkeypatch.setattr(vector_store, "get_vector_store", lambda: MagicMock())
    storage = storage_module.get_storage()
    assert storage.bm25_store.doc_ids == [hid]
    assert "langgraph" in storage.bm25_store.corpus[0]


def test_atomic_save_failure_keeps_readable_disk_snapshot(tmp_path, monkeypatch):
    from resume_match.rag import bm25_store

    index = BM25Store(tmp_path / "bm25.pkl")
    old = entry("old")
    old.id = 1
    index.add(old)
    monkeypatch.setattr(bm25_store.os, "replace", MagicMock(side_effect=OSError("disk full")))
    fresh = entry("new")
    fresh.id = 2
    with pytest.raises(OSError):
        index.replace_all([fresh])
    assert BM25Store(tmp_path / "bm25.pkl").doc_ids == [1]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bm25.pkl"]
