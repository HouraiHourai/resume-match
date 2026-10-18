"""精排：45% 归一化双路 RRF + 35% 技能 + 10% 资历 + 5% 领域 + 5% 公司。

RRF 按各路原始分数生成排名，避免直接相加不同尺度的 BM25/向量分数。
阈值是待评测的启发式参数，融合分不是概率或余弦相似度。
"""
from resume_match.state import JDRequirements, SimilarHistory

FINE_THRESHOLD = 0.42
RRF_K = 60


def reciprocal_rank_scores(candidates: list[dict]) -> dict[int, float]:
    """固定双路归一化：单路第 1 名=0.5，两路均第 1 名=1。

    同分使用相同竞争排名；排序再用 history_id 保证确定性。
    不按命中路数归一化，避免单路故障时弱证据突然得到双倍权重。
    """
    scores = {c["history_id"]: 0.0 for c in candidates}
    for field in ("vector_similarity", "bm25_score"):
        ranked = sorted(
            (c for c in candidates if c.get(field, 0) > 0),
            key=lambda c: (-c[field], c["history_id"]),
        )
        rank = 0
        previous = None
        for position, candidate in enumerate(ranked, 1):
            if candidate[field] != previous:
                rank = position
                previous = candidate[field]
            scores[candidate["history_id"]] += (RRF_K + 1) / (2 * (RRF_K + rank))
    return scores


def compute_skill_overlap(query_skills: list[str], candidate_skills: list[str]) -> float:
    """计算技能 Jaccard 相似度。

    Args:
        query_skills: 查询 JD 的技能列表
        candidate_skills: 候选历史 JD 的技能列表

    Returns:
        Jaccard 相似度 (0-1)
    """
    if not query_skills or not candidate_skills:
        return 0.0

    query_set = set(s.lower().strip() for s in query_skills if s)
    candidate_set = set(s.lower().strip() for s in candidate_skills if s)

    if not query_set or not candidate_set:
        return 0.0

    intersection = len(query_set & candidate_set)
    union = len(query_set | candidate_set)

    return intersection / union if union > 0 else 0.0


def compute_seniority_score(query_seniority: str, candidate_seniority: str) -> float:
    """计算资历匹配度。

    资历分级: intern(0) < junior(1) < mid(2) < senior(3) < lead(4)
    差距越小分数越高，每差 1 级扣 0.25 分。

    Args:
        query_seniority: 查询 JD 的资历要求
        candidate_seniority: 候选历史 JD 的资历

    Returns:
        资历匹配分数 (0-1)
    """
    seniority_map = {
        "intern": 0,
        "junior": 1,
        "mid": 2,
        "senior": 3,
        "lead": 4,
        "unspecified": 2,  # 默认中级
    }

    query_level = seniority_map.get(query_seniority.lower() if query_seniority else "unspecified", 2)
    candidate_level = seniority_map.get(candidate_seniority.lower() if candidate_seniority else "unspecified", 2)

    # 差距越小分数越高
    diff = abs(query_level - candidate_level)
    return max(0.0, 1.0 - diff * 0.25)  # 差 1 级扣 0.25


def compute_domain_score(query_domain: str, candidate_domain: str) -> float:
    """计算领域匹配度（宽松匹配策略）。

    完全匹配 → 1.0
    包含关系或相关领域 → 0.7（例如 "AI Application" 和 "短视频/AIGC"）
    不匹配或未指定 → 0.5（给中等分，不严格惩罚）

    Args:
        query_domain: 查询 JD 的领域
        candidate_domain: 候选历史 JD 的领域

    Returns:
        领域匹配分数 (0.5-1.0)
    """
    if not query_domain or not candidate_domain:
        return 0.5  # 未指定时给中等分

    query_lower = query_domain.lower()
    candidate_lower = candidate_domain.lower()

    # 完全匹配
    if query_lower == candidate_lower:
        return 1.0

    # 包含关系（AI 相关领域宽松匹配）
    ai_keywords = ['ai', 'agent', 'llm', 'nlp', 'machine learning', 'aigc']
    query_is_ai = any(kw in query_lower for kw in ai_keywords)
    candidate_is_ai = any(kw in candidate_lower for kw in ai_keywords)

    if query_is_ai and candidate_is_ai:
        return 0.7  # 都是 AI 相关，给较高分

    # 其他领域：不严格惩罚，给中等偏低分
    return 0.5


def compute_company_score(query_company: str, candidate_company: str) -> float:
    """计算公司匹配度。

    同一公司的历史岗位应该被优先推荐，因为:
    1. 用户可能对该公司感兴趣
    2. 可以对比不同岗位的要求
    3. 历史投递经验更有参考价值

    Args:
        query_company: 查询 JD 的公司名称
        candidate_company: 候选历史 JD 的公司名称

    Returns:
        公司匹配分数 (0-1)
    """
    if not query_company or not candidate_company:
        return 0.0  # 公司信息缺失，不给分

    # 规范化公司名称（去除常见后缀）
    query_normalized = query_company.replace("有限公司", "").replace("股份", "").replace("集团", "").strip().lower()
    candidate_normalized = candidate_company.replace("有限公司", "").replace("股份", "").replace("集团", "").strip().lower()

    if not query_normalized or not candidate_normalized:
        return 0.0

    # 完全匹配
    if query_normalized == candidate_normalized:
        return 1.0

    # 包含关系（处理子公司、部门等情况）
    if query_normalized in candidate_normalized or candidate_normalized in query_normalized:
        return 0.8

    # 不匹配
    return 0.0


def fine_rank(
    query_jd: JDRequirements,
    candidates: list[dict],
    top_k: int = 3,
) -> list[SimilarHistory]:
    """精排: 多因子加权打分。

    Args:
        query_jd: 当前查询的 JD 结构化数据
        candidates: 粗排返回的候选列表
        top_k: 返回的最终结果数量

    Returns:
        精排后的 SimilarHistory 列表
    """
    if top_k <= 0:
        return []
    retrieval_scores = reciprocal_rank_scores(candidates)
    scored_candidates = []

    for c in candidates:
        # 计算各项得分
        retrieval_score = retrieval_scores[c['history_id']]
        skill_score = compute_skill_overlap(query_jd.hard_skills, c['hard_skills'])
        company_score = compute_company_score(query_jd.company, c.get('company'))
        seniority_score = compute_seniority_score(query_jd.seniority, c.get('seniority', ''))
        domain_score = compute_domain_score(query_jd.domain, c.get('domain'))

        # 召回相关性使用 RRF，BM25-only 候选同样获得检索分。
        final_score = (
            0.45 * retrieval_score +
            0.35 * skill_score +
            0.05 * company_score +
            0.10 * seniority_score +
            0.05 * domain_score
        )

        scored_candidates.append({
            'candidate': c,
            'final_score': final_score,
            'debug_scores': {  # 用于调试和结果解释
                'retrieval_rrf': round(retrieval_score, 3),
                'skill': round(skill_score, 3),
                'company': round(company_score, 3),
                'seniority': round(seniority_score, 3),
                'domain': round(domain_score, 3),
            }
        })

    # 排序
    scored_candidates.sort(key=lambda x: (-x['final_score'], x['candidate']['history_id']))

    # 精排阈值过滤 + 转换为 SimilarHistory
    results = []
    for item in scored_candidates:
        if item['final_score'] < FINE_THRESHOLD:
            continue

        c = item['candidate']
        meta = c['metadata']

        results.append(SimilarHistory(
            history_id=c['history_id'],
            jd_summary=meta.get('jd_summary', ''),
            match_score=meta.get('match_score') if meta.get('match_score', -1) != -1 else None,
            should_apply=meta.get('should_apply') or None,
            similarity_score=round(item['final_score'], 3),  # 用融合分数替代纯向量分数
            created_at=meta.get('created_at', ''),
        ))

        if len(results) >= top_k:
            break

    return results
