"""独立双路召回，按历史 ID 合并；保留原始分数供精排做 RRF。"""
import logging

from resume_match.state import JDRequirements
from resume_match.rag.embedder import embed_text, build_embedding_text
from resume_match.rag.vector_store import get_vector_store
from resume_match.rag.bm25_store import get_bm25_store

logger = logging.getLogger(__name__)
COARSE_THRESHOLD = 0.45


def _candidate(history_id: int, meta: dict) -> dict:
    skills = meta.get("hard_skills") or []
    if isinstance(skills, str):
        skills = [s.strip() for s in skills.split(",") if s.strip()]
    return {
        "history_id": history_id, "metadata": meta,
        "vector_similarity": 0.0, "bm25_score": 0.0,
        "hard_skills": skills, "domain": meta.get("domain"),
        "seniority": meta.get("seniority"), "company": meta.get("company"),
    }


def coarse_rank(jd_structured: JDRequirements, top_k: int = 20, *,
                vector_store=None, bm25_store=None, embedder=None,
                strict: bool = False) -> list[dict]:
    """每路最多 top_k 条；单路故障不阻止另一路，最多返回 2*top_k 条。"""
    if top_k <= 0:
        return []
    candidates: dict[int, dict] = {}
    try:
        vector_store = vector_store if vector_store is not None else get_vector_store()
        count = vector_store.count()
        if count:
            query = build_embedding_text(
                jd_structured.role_title, jd_structured.hard_skills,
                jd_structured.domain, jd_structured.seniority,
            )
            result = vector_store.query((embedder or embed_text)(query), top_k=min(top_k, count))
            for id_, distance, meta in zip(
                result.get("ids", [[]])[0],
                result.get("distances", [[]])[0],
                result.get("metadatas", [[]])[0],
            ):
                similarity = 1.0 - float(distance)
                if similarity < COARSE_THRESHOLD:
                    continue
                item = _candidate(int(id_), meta or {})
                item["vector_similarity"] = similarity
                candidates[item["history_id"]] = item
    except Exception:
        if strict:
            raise
        logger.warning("Vector recall failed; continuing with BM25", exc_info=True)

    try:
        bm25_store = bm25_store if bm25_store is not None else get_bm25_store()
        if bm25_store.count():
            query = " ".join([
                jd_structured.role_title,
                " ".join(jd_structured.hard_skills),
                " ".join(jd_structured.hard_skills),
                jd_structured.domain or "",
            ])
            for result in bm25_store.query(query, top_k=top_k):
                history_id = int(result["history_id"])
                if history_id not in candidates:
                    candidates[history_id] = _candidate(history_id, result["metadata"])
                candidates[history_id]["bm25_score"] = float(result["bm25_score"])
    except Exception:
        if strict:
            raise
        logger.warning("BM25 recall failed; keeping vector candidates", exc_info=True)
    return list(candidates.values())
