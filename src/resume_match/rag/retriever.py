"""按数据规模选择候选预算，过滤重复后做混合排序。"""
import logging
from typing import Optional

from resume_match.state import JDRequirements, SimilarHistory
from resume_match.rag.vector_store import get_vector_store
from resume_match.rag.bm25_store import get_bm25_store
from resume_match.rag.coarse_ranker import coarse_rank
from resume_match.rag.fine_ranker import fine_rank

logger = logging.getLogger(__name__)


def find_similar(
    jd_structured: JDRequirements,
    top_k: int = 3,
    exclude_id: Optional[int] = None,
    current_jd_text: Optional[str] = None,
) -> list[SimilarHistory]:
    """每路预算：小库全部，否则 35%（15–80）。正文参数保留兼容，未用于去重。"""
    if top_k <= 0:
        return []
    counts = []
    for factory in (get_vector_store, get_bm25_store):
        try:
            counts.append(factory().count())
        except Exception:
            logger.warning("Cannot count recall store", exc_info=True)
    total_count = max(counts, default=0)
    if not total_count:
        return []
    coarse_k = total_count if total_count <= 15 else max(15, min(80, int(total_count * 0.35)))
    candidates = coarse_rank(jd_structured, top_k=coarse_k)
    filtered = filter_candidates(jd_structured, candidates, exclude_id=exclude_id)
    return fine_rank(jd_structured, filtered, top_k=top_k)


def filter_candidates(jd_structured: JDRequirements, candidates: list[dict],
                      exclude_id: Optional[int] = None) -> list[dict]:
    """共享生产去重策略，供独立评测复用，不读取任何存储。"""
    company = (jd_structured.company or "").strip().casefold()
    title = jd_structured.role_title.strip().casefold()
    filtered = []
    for candidate in candidates:
        if candidate["history_id"] == exclude_id:
            continue
        meta = candidate["metadata"]
        # 未知公司不等于同一家公司；不能把两个 None 当成相同实体。
        same_company = bool(company) and (meta.get("company") or "").strip().casefold() == company
        same_position = (meta.get("role_title") or "").strip().casefold() == title
        if candidate["vector_similarity"] > 0.95 and same_company and same_position:
            continue
        filtered.append(candidate)
    return filtered


def format_similar_histories_for_prompt(similar: list[SimilarHistory]) -> str:
    if not similar:
        return "(No usable historical context is available for this request.)"
    lines = []
    for i, s in enumerate(similar, 1):
        score = s.match_score if s.match_score is not None else "?"
        lines.append(
            f"{i}. {s.jd_summary} — match_score={score}, should_apply={s.should_apply or '?'}, "
            f"ranking_score={s.similarity_score:.3f}, analyzed {s.created_at}"
        )
    return "\n".join(lines)
