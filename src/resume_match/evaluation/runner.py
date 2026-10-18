"""使用独立临时索引评测检索方法，支持本地哈希向量和 SiliconFlow 嵌入。"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import platform
import statistics
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from resume_match.evaluation.dataset import Dataset, dataset_digest, load_dataset
from resume_match.evaluation.metrics import score_query, summarize
from resume_match.history.models import HistoryEntry
from resume_match.rag.bm25_store import BM25Store, _tokenize
from resume_match.rag.coarse_ranker import COARSE_THRESHOLD, coarse_rank
from resume_match.rag.embedder import build_embedding_text_from_dict
from resume_match.rag.fine_ranker import FINE_THRESHOLD, RRF_K, fine_rank, reciprocal_rank_scores
from resume_match.rag.retriever import filter_candidates
from resume_match.rag.vector_store import VectorStore

METHODS = ("vector", "bm25", "vector_rules", "rrf_only", "hybrid")


class EmptyStore:
    def count(self):
        return 0


def smoke_embedding(text: str) -> list[float]:
    """生成确定性词袋哈希向量，用于验证评测流程。"""
    result = [0.0] * 128
    for token in _tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        result[int.from_bytes(digest[:4], "big") % len(result)] += 1
    norm = math.sqrt(sum(x * x for x in result))
    return [x / norm for x in result] if norm else result


def prepare_embeddings(texts: list[str], backend: str) -> tuple[dict, dict]:
    unique = list(dict.fromkeys(texts))
    vectors, durations = {}, []
    client = None
    if backend == "siliconflow":
        from resume_match.rag.embedder import _get_model
        client = _get_model()
        encode = client.embed_query
    elif backend == "smoke":
        encode = smoke_embedding
    else:
        raise ValueError("unknown embedding backend")
    started = time.perf_counter()
    dimension = None
    for text in unique:
        t0 = time.perf_counter()
        vector = encode(text)
        durations.append((time.perf_counter() - t0) * 1000)
        if not vector or not all(math.isfinite(v) for v in vector) or not any(vector):
            raise ValueError("embedding returned empty, zero or non-finite vector")
        dimension = dimension or len(vector)
        if len(vector) != dimension:
            raise ValueError("inconsistent embedding dimensions")
        vectors[text] = vector
    # 不记录token费用估算，更不打印key或完整服务地址。
    return vectors, {
        "backend": backend, "model": client.model if client else "deterministic-hash-smoke-v1",
        "dimension": dimension, "unique_texts": len(unique),
        "logical_embedding_calls": len(unique) if client else 0,
        "sdk_retries_may_add_http_calls": bool(client),
        "total_precompute_ms": (time.perf_counter() - started) * 1000,
        "per_text_ms": durations,
        "note": "Embeddings precomputed once for fair ranking; NOT included in retrieval latency. No token/cost accounting.",
    }


def _retrieve(method, jd, vector, bm25, embed, coarse_k, k, excluded):
    candidates = coarse_rank(
        jd, top_k=coarse_k,
        vector_store=EmptyStore() if method == "bm25" else vector,
        bm25_store=EmptyStore() if method in ("vector", "vector_rules") else bm25,
        embedder=embed, strict=True,
    )
    candidates = filter_candidates(jd, [c for c in candidates if c["history_id"] not in excluded])
    if method in ("hybrid", "vector_rules"):
        selected = fine_rank(jd, candidates, top_k=k)
        results = [(r.history_id, r.similarity_score) for r in selected]
    else:
        if method == "rrf_only":
            scores = reciprocal_rank_scores(candidates)
        else:
            field = "vector_similarity" if method == "vector" else "bm25_score"
            scores = {c["history_id"]: c[field] for c in candidates}
        ordered = sorted(scores, key=lambda id_: (-scores[id_], id_))[:k]
        results = [(id_, scores[id_]) for id_ in ordered]
    return results, candidates


def run_evaluation(data: Dataset, *, backend="smoke", split="test", k=3, repeats=3) -> dict:
    if k < 1 or repeats < 1:
        raise ValueError("k and repeats must be positive")
    if data.label_source == "unreviewed":
        raise ValueError("unreviewed placeholder labels cannot be evaluated; import human annotations first")
    queries = [q for q in data.queries if q.split == split]
    if not queries:
        raise ValueError(f"no queries in split={split}")
    documents = data.documents
    to_int = {d.id: i for i, d in enumerate(documents, 1)}
    to_id = {i: id_ for id_, i in to_int.items()}
    def text_for(record):
        return build_embedding_text_from_dict(record.jd.model_dump())
    vectors, embedding_info = prepare_embeddings([text_for(d) for d in documents + queries], backend)
    rows = []
    coarse_k = len(documents) if len(documents) <= 15 else max(15, min(80, int(len(documents) * .35)))
    # 所有索引只活在专属临时目录，不调用get_storage或默认索引单例。
    with tempfile.TemporaryDirectory(prefix="resume-match-eval-") as directory:
        root = Path(directory)
        vector = VectorStore(root / "chroma")
        bm25 = BM25Store(root / "bm25.pkl")
        try:
            entries = []
            t0 = time.perf_counter()
            for d in documents:
                jd = d.jd.model_dump()
                entry = HistoryEntry(id=to_int[d.id], created_at="evaluation-fixture",
                    jd_text=d.text, jd_summary=d.jd.role_title, resume_filename="no-resume",
                    match_score=None, should_apply=None, full_result={"jd": jd})
                entries.append(entry)
                vector.add(str(entry.id), text_for(d), {
                    "history_id": entry.id, "jd_summary": d.jd.role_title,
                    "role_title": d.jd.role_title, "company": d.jd.company or "",
                    "hard_skills": ",".join(d.jd.hard_skills), "domain": d.jd.domain or "",
                    "seniority": d.jd.seniority, "created_at": "evaluation-fixture",
                }, vectors[text_for(d)])
            bm25.replace_all(entries)
            index_build_ms = (time.perf_counter() - t0) * 1000
            # 热身排除首次索引查询开销；每次运行均记录明确的热缓存条件。
            _retrieve("hybrid", queries[0].jd, vector, bm25, vectors.__getitem__, coarse_k, k, set())
            for qi, q in enumerate(queries):
                excluded = {to_int[id_] for id_ in q.excluded_ids}
                method_rows = {}
                for repeat in range(repeats):
                    shift = (qi + repeat) % len(METHODS)
                    for method in METHODS[shift:] + METHODS[:shift]:
                        t0 = time.perf_counter()
                        results, candidates = _retrieve(method, q.jd, vector, bm25,
                            vectors.__getitem__, coarse_k, k, excluded)
                        elapsed = (time.perf_counter() - t0) * 1000
                        ids = [to_id[id_] for id_, _ in results]
                        if method not in method_rows:
                            method_rows[method] = {
                                "query_id": q.id, "split": q.split, "method": method,
                                "retrieved_ids": ids, "scores": [score for _, score in results],
                                "metrics": score_query(ids, q.relevance, k), "latency_ms": [],
                                "candidates": [{
                                    "id": to_id[c["history_id"]],
                                    "vector_similarity": c["vector_similarity"], "bm25_score": c["bm25_score"],
                                } for c in candidates],
                            }
                        elif method_rows[method]["retrieved_ids"] != ids:
                            raise RuntimeError(f"non-deterministic ranking for {q.id}/{method}")
                        method_rows[method]["latency_ms"].append(elapsed)
                rows.extend(method_rows.values())
        finally:
            # Chroma has no public close API. Release only this run's system,
            # including failed runs, before Windows deletes the temporary files.
            from chromadb.api.client import SharedSystemClient
            for identifier, system in list(SharedSystemClient._identifier_to_system.items()):
                if str(root / "chroma") == system.settings.persist_directory:
                    system.stop()
                    SharedSystemClient._identifier_to_system.pop(identifier, None)
    summary = {method: summarize([r for r in rows if r["method"] == method]) for method in METHODS}
    human = data.label_source == "human" and all(q.human_reviewed for q in queries) and bool(data.reviewer.strip())
    eligible = data.provenance == "real" and human and backend == "siliconflow" and split == "test"
    code_root = Path(__file__).resolve().parents[1]
    fingerprints = {str(p.relative_to(code_root)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ("rag", "evaluation") for p in (code_root / folder).glob("*.py")}
    return {
        "run_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": data.name,
        "dataset_sha256": dataset_digest(data), "provenance": data.provenance,
        "label_source": data.label_source, "human_reviewed": human, "resume_claim_candidate": eligible,
        "claim_gate_note": "Gate checks declarations, not truth. Verify source rights, labels, split independence and experiment design before publishing.",
        "documents": len(documents), "queries": len(queries), "split": split, "k": k, "repeats": repeats,
        "embedding": embedding_info, "index_build_ms": index_build_ms,
        "protocol": {
            "coarse_k_per_route": coarse_k, "vector_threshold": COARSE_THRESHOLD,
            "fine_threshold": FINE_THRESHOLD, "rrf_k": RRF_K,
            "weights": {"rrf": .45, "skill": .35, "seniority": .10, "domain": .05, "company": .05},
            "latency_scope": "warm retrieval/ranking with precomputed embeddings; excludes remote embedding, index build, JD extraction, Agent, generation, SSE",
            "metrics": "macro mean; hit/recall/MRR@k/nDCG@k over positive queries; precision uses fixed k over all queries; negatives reported separately",
            "methods": {
                "vector": "production vector recall threshold + common exclusions, raw cosine ordering",
                "bm25": "production positive-score BM25 recall + common exclusions, raw BM25 ordering",
                "vector_rules": "vector candidates only + same production fine_rank (BM25 ablation)",
                "rrf_only": "two recall routes + common exclusions + RRF only, no business fine threshold",
                "hybrid": "production dual recall + common exclusions + fine_rank",
            },
        },
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
            "packages": {p: importlib.metadata.version(p) for p in ("chromadb", "rank-bm25", "langchain-openai")}},
        "source_sha256": fingerprints, "summary": summary, "rows": rows,
    }


def write_report(result: dict, data: Dataset, out: Path):
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "dataset.snapshot.json").write_text(data.model_dump_json(indent=2), encoding="utf-8")
    with (out / "per_query.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "method", "retrieved_ids", "hit", "precision", "recall", "mrr_at_k", "ndcg_at_k", "median_ms"])
        for row in result["rows"]:
            m = row["metrics"]
            writer.writerow([row["query_id"], row["method"], "|".join(row["retrieved_ids"]),
                m["hit"], m["precision"], m["recall"], m["mrr"], m["ndcg"], statistics.median(row["latency_ms"])])
    lines = ["# RAG 检索评测报告", "",
        f"数据：{result['dataset']}；来源：{result['provenance']}；标签：{result['label_source']}；人工复核：{result['human_reviewed']}。",
        f"Embedding：{result['embedding']['model']}；split={result['split']}；{result['documents']}条候选，{result['queries']}条查询，k={result['k']}。", "",
        "评测范围为岗位检索与排序，不包含线上可用性、生成质量和端到端请求时延。",
        "当前实验未同时满足真实来源、人工复核、远程嵌入和 test 划分条件；结果仅用于流程验证或探索性分析。" if not result["resume_claim_candidate"]
        else "数据声明满足真实来源、人工复核、远程嵌入和 test 划分条件；数据独立性、代表性和来源仍需单独核验。", "",
        "| 方法 | Hit@k | Precision@k | Recall@k | MRR@k | nDCG@k | 空结果率 | 排序P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    def fmt(value):
        return "N/A" if value is None else f"{value:.4f}"
    for method, m in result["summary"].items():
        lines.append("| " + method + " | " + " | ".join(fmt(m[x]) for x in
            ("hit", "precision", "recall", "mrr", "ndcg", "empty", "latency_p95_ms")) + " |")
    v, h = result["summary"]["vector"]["hit"], result["summary"]["hybrid"]["hit"]
    if v is not None and h is not None:
        lines += ["", f"混合检索相对纯向量检索的 Hit@k 差值：{(h-v)*100:+.2f} 个百分点。"]
    lines += ["", "## 计量边界", "",
        "- Hit/Recall/MRR/nDCG 仅对存在相关候选的查询取宏平均；无相关候选查询的误召回率见 result.json。",
        "- MRR 截断于前 k 个结果。Precision 使用固定分母 k，并对全部查询取宏平均。",
        "- 时延覆盖预计算嵌入后的热检索与排序，排除远程嵌入、建索引、JD 提取、Agent、生成和 SSE。各查询重复测量取中位数，再计算查询级 P95。",
        "- vector_rules 使用向量召回和业务规则精排，用于分析 BM25 的贡献；rrf_only 使用双路召回与 RRF，用于分析业务规则的贡献。",
        "- 标注应在检索结果评阅前完成；dev 用于调参，test 保持冻结。使用测试结果调参后，需重新准备独立测试集。",
        "- 逐条结果、候选原始分数、运行环境和源码指纹见 result.json；完整输入见 dataset.snapshot.json。",
        "", "## 结果适用范围", "",
        "合成数据和哈希向量用于评测流程验证，其指标不代表真实岗位检索质量。",
        "结果解释需结合候选规模、查询数量、标注来源、嵌入后端和数据划分。自动检查仅核验数据声明，不验证样本代表性或测试集独立性。",
    ]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="必须为尚不存在的新目录")
    parser.add_argument("--backend", choices=["smoke", "siliconflow"], default="smoke")
    parser.add_argument("--split", choices=["dev", "test"], default="test")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output directory already exists; choose a new run name")
    data = load_dataset(args.dataset)
    print(f"Dataset validated: {len(data.documents)} documents / {len(data.queries)} queries; backend={args.backend}", flush=True)
    if args.backend == "siliconflow":
        print("Sending this dataset's structured JD text to configured SiliconFlow API (may incur cost). No personal history DB is read.", flush=True)
    result = run_evaluation(data, backend=args.backend, split=args.split, k=args.k, repeats=args.repeats)
    write_report(result, data, args.output)
    print(f"Report: {(args.output / 'report.md').resolve()}")
    print(f"Evaluation declaration checks passed: {result['resume_claim_candidate']}")


if __name__ == "__main__":
    main()
