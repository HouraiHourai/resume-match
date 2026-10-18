"""从 SQLite 全量重建 BM25。运行前停止应用，避免跨进程写索引。"""
from resume_match.history.storage import HistoryStorage
from resume_match.rag.bm25_store import get_bm25_store


def rebuild_bm25_index():
    storage = HistoryStorage()
    count = get_bm25_store().replace_all(storage.iter_entries())
    print(f"Rebuilt BM25 index: {count} entries")
    return count


if __name__ == "__main__":
    rebuild_bm25_index()
