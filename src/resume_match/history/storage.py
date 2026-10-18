"""SQLite 存储层。

用 Python 自带的 sqlite3, 不引 SQLAlchemy —— 单用户、零配置、嵌入式, ORM 是杀鸡用牛刀。
DB 文件默认放 ~/.resume_match/history.db(用户数据, 不进 git 也不进 Docker 镜像)。
"""
import json
import logging
import os
import sqlite3
from pathlib import Path
from typing import Optional

from resume_match.history.models import HistoryEntry

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS history_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at TEXT NOT NULL,
    jd_text TEXT NOT NULL,
    jd_summary TEXT NOT NULL,
    resume_filename TEXT NOT NULL,
    match_score INTEGER,
    should_apply TEXT,
    full_result_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_history_created_at ON history_entries(created_at DESC);
"""


def _default_db_path() -> Path:
    # 可用 SQLITE_PATH 覆盖(见 README Configuration), 默认 ~/.resume_match/history.db。
    env = os.getenv("SQLITE_PATH")
    return Path(env) if env else Path.home() / ".resume_match" / "history.db"


class HistoryStorage:
    """历史记录的 SQLite 存储。首次实例化自动建表(无需手动 migration)。

    vector_store/bm25_store 是可注入的派生索引，各自失败不阻止另一套更新。
    不传索引时为纯 SQLite，便于离线测试；只有默认 get_storage 才同时挂载两路索引。
    """

    def __init__(self, db_path: Optional[Path] = None, vector_store=None, bm25_store=None):
        self.db_path = Path(db_path) if db_path else _default_db_path()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_store = vector_store
        self.bm25_store = bm25_store
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row   # 让查询结果能按列名取
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def create_entry(self, entry: HistoryEntry) -> int:
        """插入一条记录, 返回自增 id。"""
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO history_entries
                   (created_at, jd_text, jd_summary, resume_filename,
                    match_score, should_apply, full_result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.created_at,
                    entry.jd_text,
                    entry.jd_summary,
                    entry.resume_filename,
                    entry.match_score,
                    entry.should_apply,
                    json.dumps(entry.full_result, ensure_ascii=False),
                ),
            )
            new_id = cur.lastrowid

        entry.id = new_id
        # 各索引独立更新；一个失败不影响另一个。SQLite 是事实源。
        if self.vector_store is not None:
            entry.id = new_id
            try:
                self.vector_store.index_entry(entry)
            except Exception:
                # 向量化失败不该让"存历史"整体失败(SQLite 已写成功)。
                logger.warning("Vector indexing failed for history %s", new_id, exc_info=True)
        if self.bm25_store is not None:
            try:
                self.bm25_store.add(entry)
            except Exception:
                logger.warning("BM25 indexing failed for history %s", new_id, exc_info=True)
        return new_id

    def list_entries(self, limit: int = 50) -> list[HistoryEntry]:
        """列表, 按 created_at 倒序。不加载 full_result_json(列表不需要, 省体积)。"""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT id, created_at, jd_text, jd_summary, resume_filename,
                          match_score, should_apply
                   FROM history_entries
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                (limit,),
            ).fetchall()
        return [
            HistoryEntry(
                id=r["id"],
                created_at=r["created_at"],
                jd_text=r["jd_text"],
                jd_summary=r["jd_summary"],
                resume_filename=r["resume_filename"],
                match_score=r["match_score"],
                should_apply=r["should_apply"],
                full_result={},   # 列表不带完整结果
            )
            for r in rows
        ]

    def get_entry(self, entry_id: int) -> Optional[HistoryEntry]:
        """按 id 取单条完整记录; 不存在返回 None。"""
        with self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM history_entries WHERE id = ?", (entry_id,)
            ).fetchone()
        if r is None:
            return None
        return HistoryEntry(
            id=r["id"],
            created_at=r["created_at"],
            jd_text=r["jd_text"],
            jd_summary=r["jd_summary"],
            resume_filename=r["resume_filename"],
            match_score=r["match_score"],
            should_apply=r["should_apply"],
            full_result=json.loads(r["full_result_json"]),
        )

    def delete_entry(self, entry_id: int) -> bool:
        """删除一条; 返回是否真的删到(id 不存在则 False)。同步删向量, 防止"僵尸检索"。"""
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM history_entries WHERE id = ?", (entry_id,))
            deleted = cur.rowcount > 0
        if deleted and self.vector_store is not None:
            try:
                self.vector_store.remove(entry_id)
            except Exception:
                logger.warning("Vector deletion failed for history %s", entry_id, exc_info=True)
        # 即使 SQL 记录已不存在，重复删除也尝试清理 BM25 残留。
        if self.bm25_store is not None:
            try:
                self.bm25_store.remove(entry_id)
            except Exception:
                logger.warning("BM25 deletion failed for history %s", entry_id, exc_info=True)
        return deleted

    def iter_entries(self):
        """有界批次读取全部完整记录，供本地索引重建，无 1000 条截断。"""
        conn = self._connect()
        try:
            cursor = conn.execute("SELECT * FROM history_entries ORDER BY id")
            while rows := cursor.fetchmany(500):
                for r in rows:
                    yield HistoryEntry(
                        id=r["id"], created_at=r["created_at"], jd_text=r["jd_text"],
                        jd_summary=r["jd_summary"], resume_filename=r["resume_filename"],
                        match_score=r["match_score"], should_apply=r["should_apply"],
                        full_result=json.loads(r["full_result_json"]),
                    )
        finally:
            conn.close()



# ---- 进程级单例(API 用); 测试可用 set_storage 注入临时库 ----
_storage: Optional[HistoryStorage] = None


def get_storage() -> HistoryStorage:
    global _storage
    if _storage is None:
        # 默认存储挂载两套索引；测试可以使用不带索引的实例。
        from resume_match.rag.vector_store import get_vector_store
        from resume_match.rag.bm25_store import get_bm25_store
        storage = HistoryStorage(vector_store=get_vector_store(), bm25_store=get_bm25_store())
        # 初次启动从 SQLite 重建本地 BM25，修复旧版本遗漏/删除的记录。
        storage.bm25_store.replace_all(storage.iter_entries())
        _storage = storage
    return _storage


def set_storage(storage: HistoryStorage) -> None:
    global _storage
    _storage = storage


def reindex_missing(storage: HistoryStorage) -> int:
    """平滑升级: 给 SQLite 里有、但向量库里没有的历史补索引。返回补了多少条。

    场景: 项目早期版本还没引入向量库, 或手动删了 ~/.resume_match/chroma/ 后重启。
    """
    if storage.vector_store is None:
        return 0
    existing = storage.vector_store.existing_ids()
    count = 0
    for item in storage.list_entries(limit=10_000):
        if str(item.id) in existing:
            continue
        full = storage.get_entry(item.id)   # 需要 full_result 才能生成 embedding
        # 理论不会发生，预防性编程
        if full is None:
            continue
        storage.vector_store.index_entry(full)
        count += 1
    return count
