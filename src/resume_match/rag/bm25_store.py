"""BM25 关键词索引。

设计目标:
- 精确匹配: 对技术栈关键词（Python, FastAPI, LangChain）敏感
- 补充向量检索: 向量擅长语义，BM25 擅长精确词匹配
- 持久化: 索引保存到磁盘，避免每次重建

BM25 原理:
- 基于词频（TF）和逆文档频率（IDF）
- 对重要关键词（如技能名称）给予高权重
- 短文档匹配度不会因为长度被稀释
"""
import os
import pickle
import tempfile
from threading import RLock
from functools import wraps
from pathlib import Path
from typing import Optional, List
from rank_bm25 import BM25Okapi

from resume_match.history.models import HistoryEntry


def _default_index_path() -> Path:
    """BM25 索引默认路径（和 Chroma 同目录）"""
    env = os.getenv("BM25_INDEX_PATH")
    if env:
        return Path(env)
    return Path.home() / ".resume_match" / "bm25_index.pkl"


def _tokenize(text: str) -> List[str]:
    """简单分词: 按空白、逗号、分号和斜杠分割（保留 +/#，支持 C++、C#），转小写。

    对于中文混合英文的技术栈，这个简单策略足够:
    - "Python, FastAPI, MySQL" → ["python", "fastapi", "mysql"]
    - "LLM应用开发" → ["llm应用开发"]（整体作为一个词）
    """
    if not text:
        return []

    # 按分隔符切分
    for sep in [',', ';', '\n', '、', '/', '／', '，', '；']:
        text = text.replace(sep, ' ')

    # 分词并转小写
    tokens = [t.strip().lower() for t in text.split() if t.strip()]
    return tokens


def _locked(method):
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return wrapper


class BM25Store:
    """BM25 索引存储。

    索引内容:
    - corpus: 文档列表（每个文档是分词后的 token 列表）
    - doc_ids: 文档 ID 列表（history_id）
    - metadata: 元数据列表（用于返回结果）
    """

    def __init__(self, index_path: Optional[Path] = None):
        self._lock = RLock()
        self.index_path = Path(index_path) if index_path else _default_index_path()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        self.corpus: List[List[str]] = []
        self.doc_ids: List[int] = []
        self.metadata: List[dict] = []
        self.bm25: Optional[BM25Okapi] = None

        # 尝试加载已有索引
        self._load()

    def _load(self) -> None:
        """从磁盘加载索引"""
        if not self.index_path.exists():
            return

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.corpus = data['corpus']
                self.doc_ids = data['doc_ids']
                self.metadata = data['metadata']

                # 重建 BM25 索引
                if self.corpus:
                    self.bm25 = BM25Okapi(self.corpus)
        except Exception:
            # 索引文件损坏，重新开始
            pass

    def _save(self) -> None:
        """保存索引到磁盘"""
        data = {
            'corpus': self.corpus,
            'doc_ids': self.doc_ids,
            'metadata': self.metadata,
        }
        # 同目录原子替换，避免中途写坏旧索引；锁只覆盖单进程。
        temp_name = None
        try:
            with tempfile.NamedTemporaryFile(dir=self.index_path.parent, delete=False) as f:
                temp_name = f.name
                pickle.dump(data, f)
            os.replace(temp_name, self.index_path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)


    def build_document(self, entry: HistoryEntry) -> str:
        """从历史记录构建 BM25 文档。

        策略: 拼接多个字段，技能列表权重最高（重复3次）
        """
        jd = (entry.full_result or {}).get("jd")
        if not jd:
            return ""

        parts = []

        # 岗位名称（重复2次，提升权重）
        role_title = jd.get("role_title", "")
        if role_title:
            parts.extend([role_title, role_title])

        # 技能列表（重复3次，最高权重）
        hard_skills = jd.get("hard_skills", []) or []
        if hard_skills:
            skills_text = " ".join(hard_skills)
            parts.extend([skills_text, skills_text, skills_text])

        # 领域
        domain = jd.get("domain")
        if domain:
            parts.append(domain)

        # 资历
        seniority = jd.get("seniority", "")
        if seniority:
            parts.append(seniority)

        return " ".join(parts)

    @_locked
    def add(self, entry: HistoryEntry) -> None:
        """添加一条历史记录到索引"""
        if not entry.id:
            return

        jd = (entry.full_result or {}).get("jd")
        if not jd:
            return

        # 构建文档
        doc_text = self.build_document(entry)
        if not doc_text:
            return

        # 分词
        tokens = _tokenize(doc_text)
        if not tokens:
            return

        # 如果已存在，先删除
        if entry.id in self.doc_ids:
            idx = self.doc_ids.index(entry.id)
            del self.corpus[idx]
            del self.doc_ids[idx]
            del self.metadata[idx]

        self._append_entry(entry, tokens)

        # 重建 BM25 索引
        self.bm25 = BM25Okapi(self.corpus)

        # 持久化
        self._save()

    def _append_entry(self, entry: HistoryEntry, tokens: list[str]) -> None:
        jd = entry.full_result["jd"]
        self.corpus.append(tokens)
        self.doc_ids.append(entry.id)
        self.metadata.append({
            "history_id": entry.id,
            "jd_summary": entry.jd_summary,
            "role_title": jd.get("role_title", ""),
            "hard_skills": jd.get("hard_skills") or [],
            "domain": jd.get("domain"),
            "seniority": jd.get("seniority"),
            "company": jd.get("company"),
            "match_score": entry.match_score,
            "should_apply": entry.should_apply,
            "created_at": entry.created_at,
        })

    @_locked
    def replace_all(self, entries) -> int:
        """从事实源完整重建；清除陈旧 ID、旧 tokenizer 和旧元数据。

        内存构建成功后一次切换并原子写盘；不会逐条重建/写盘。
        """
        prepared = {}
        for entry in entries:
            tokens = _tokenize(self.build_document(entry))
            if entry.id is not None and tokens:
                prepared[entry.id] = (entry, tokens)
        old = (self.corpus, self.doc_ids, self.metadata, self.bm25)
        self.corpus, self.doc_ids, self.metadata = [], [], []
        try:
            for entry, tokens in prepared.values():
                self._append_entry(entry, tokens)
            self.bm25 = BM25Okapi(self.corpus) if self.corpus else None
            self._save()
        except Exception:
            self.corpus, self.doc_ids, self.metadata, self.bm25 = old
            raise
        return len(self.doc_ids)

    @_locked
    def query(self, query_text: str, top_k: int = 20) -> List[dict]:
        """BM25 检索。

        Args:
            query_text: 查询文本
            top_k: 返回前 k 个结果

        Returns:
            结果列表，每个包含 history_id, metadata, bm25_score
        """
        if top_k <= 0 or not self.bm25 or not self.corpus:
            return []

        # 分词
        query_tokens = _tokenize(query_text)
        if not query_tokens:
            return []

        # BM25 检索
        scores = self.bm25.get_scores(query_tokens)

        # 排序并返回 top-k
        results = []
        for i, score in enumerate(scores):
            if score > 0:  # 只返回有匹配的
                results.append({
                    'history_id': self.doc_ids[i],
                    'metadata': self.metadata[i],
                    'bm25_score': float(score),
                })

        # 按分数排序
        results.sort(key=lambda x: x['bm25_score'], reverse=True)

        return results[:top_k]

    @_locked
    def remove(self, history_id: int) -> None:
        """从索引中删除一条记录"""
        if history_id not in self.doc_ids:
            return

        idx = self.doc_ids.index(history_id)
        del self.corpus[idx]
        del self.doc_ids[idx]
        del self.metadata[idx]

        # 重建 BM25 索引
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)
        else:
            self.bm25 = None

        # 持久化
        self._save()

    @_locked
    def count(self) -> int:
        """返回索引中的文档数量"""
        return len(self.corpus)


# ---- 进程级单例 ----
_store: Optional[BM25Store] = None


def get_bm25_store() -> BM25Store:
    global _store
    if _store is None:
        _store = BM25Store()
    return _store


def set_bm25_store(store: Optional[BM25Store]) -> None:
    global _store
    _store = store
