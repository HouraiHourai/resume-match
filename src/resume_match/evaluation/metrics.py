"""计算宏平均检索指标，并单独统计无相关候选查询的误召回率。"""
import math
import statistics


def score_query(ranked_ids: list[str], relevance: dict[str, int], k: int) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("duplicate retrieved IDs")
    if set(ranked_ids) - set(relevance):
        raise ValueError("retrieved unknown or excluded document")
    ranked = ranked_ids[:k]
    positives = {id_ for id_, grade in relevance.items() if grade > 0}
    hits = sum(id_ in positives for id_ in ranked)
    dcg = sum((2 ** relevance[id_] - 1) / math.log2(i + 2) for i, id_ in enumerate(ranked))
    ideal = sum((2 ** grade - 1) / math.log2(i + 2)
                for i, grade in enumerate(sorted(relevance.values(), reverse=True)[:k]))
    rr = next((1 / i for i, id_ in enumerate(ranked, 1) if id_ in positives), 0.0)
    return {
        "has_relevant": bool(positives), "relevant_count": len(positives),
        "returned_count": len(ranked), "hit": float(hits > 0) if positives else None,
        "precision": hits / k,  # 分母固定k，不因少返回而虚高
        "recall": hits / len(positives) if positives else None,
        "mrr": rr if positives else None,
        "ndcg": dcg / ideal if ideal else None,
        "empty": float(not ranked),
        "false_positive": float(bool(ranked)) if not positives else None,
    }


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * p
    lower = math.floor(position)
    upper = math.ceil(position)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(rows: list[dict]) -> dict:
    result = {"queries": len(rows), "positive_queries": sum(r["metrics"]["has_relevant"] for r in rows)}
    for key in ("hit", "precision", "recall", "mrr", "ndcg", "empty", "false_positive"):
        values = [r["metrics"][key] for r in rows if r["metrics"][key] is not None]
        result[key] = statistics.mean(values) if values else None
    # 每条query重复测量后的中位数，用这些query级耗时汇总。
    times = [statistics.median(r["latency_ms"]) for r in rows]
    result.update(latency_mean_ms=statistics.mean(times) if times else None,
                  latency_p50_ms=percentile(times, .5), latency_p95_ms=percentile(times, .95))
    return result
