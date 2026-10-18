import copy
import math
from pathlib import Path

import pytest
from pydantic import ValidationError

from resume_match.evaluation.annotation import create_draft, export_annotations, import_annotations, write_new
from resume_match.evaluation.dataset import Dataset, load_dataset
from resume_match.evaluation.metrics import percentile, score_query, summarize
from resume_match.evaluation.runner import METHODS, run_evaluation, write_report


@pytest.fixture
def data():
    return load_dataset(Path(__file__).resolve().parents[1] / "evaluation/synthetic_jd_pilot.json")


def test_metrics_hand_calculated_and_fixed_precision():
    m = score_query(["c", "a"], {"a": 2, "b": 1, "c": 0}, 3)
    assert m["hit"] == 1
    assert m["precision"] == pytest.approx(1 / 3)
    assert m["recall"] == .5
    assert m["mrr"] == .5
    assert m["ndcg"] == pytest.approx((3 / math.log2(3)) / (3 + 1 / math.log2(3)))


def test_negative_queries_are_separate_from_positive_macro():
    positive = score_query(["a"], {"a": 1}, 3)
    negative = score_query(["b"], {"b": 0}, 3)
    assert negative["recall"] is None and negative["false_positive"] == 1
    result = summarize([{"metrics": positive, "latency_ms": [1, 100, 2]},
                        {"metrics": negative, "latency_ms": [3, 4, 5]}])
    assert result["hit"] == 1
    assert result["precision"] == pytest.approx(1 / 6)
    assert result["latency_p95_ms"] == pytest.approx(3.9)
    assert score_query([], {"b": 0}, 3)["false_positive"] == 0
    assert percentile([], .95) is None


@pytest.mark.parametrize("ids,k", [(["a", "a"], 3), (["unknown"], 3), ([], 0)])
def test_invalid_rankings_rejected(ids, k):
    with pytest.raises(ValueError):
        score_query(ids, {"a": 2}, k)


@pytest.mark.parametrize("fault", ["missing_grade", "raw_leak", "embedding_leak", "duplicate_id",
                                  "blank_source", "duplicate_query_embedding", "invalid_exclusion"])
def test_dataset_rejects_invalid_or_leaky_data(data, fault):
    raw = data.model_dump()
    q, d = raw["queries"][0], raw["documents"][0]
    if fault == "missing_grade":
        q["relevance"].pop(d["id"])
    elif fault == "raw_leak":
        q["text"] = d["text"]
    elif fault == "embedding_leak":
        q["jd"] = copy.deepcopy(d["jd"])
    elif fault == "duplicate_id":
        raw["documents"][1]["id"] = d["id"]
    elif fault == "blank_source":
        q["source"] = " "
    elif fault == "duplicate_query_embedding":
        raw["queries"][1]["jd"] = copy.deepcopy(q["jd"])
    else:
        q["excluded_ids"] = ["missing"]
    with pytest.raises(ValidationError):
        Dataset.model_validate(raw)


def test_annotation_roundtrip_preserves_synthetic_provenance(data, tmp_path):
    sheet = export_annotations(data)
    assert all(p["grade"] is None and "rank" not in p for p in sheet["pairs"])
    sheet["reviewer"] = "test-reviewer"
    for pair in sheet["pairs"]:
        pair.update(grade=1, reason="测试标注，不作为真实证据")
    reviewed = import_annotations(data, sheet)
    assert reviewed.provenance == "synthetic"
    assert reviewed.label_source == "human"
    assert all(q.human_reviewed and q.relevance_reasons for q in reviewed.queries)
    write_new(tmp_path / "reviewed.json", reviewed.model_dump())
    with pytest.raises(FileExistsError):
        write_new(tmp_path / "reviewed.json", {})


@pytest.mark.parametrize("fault", ["digest", "reviewer", "blank_grade", "bool_grade", "reason", "missing", "duplicate"])
def test_bad_annotations_cannot_be_imported(data, fault):
    sheet = export_annotations(data)
    sheet["reviewer"] = "test"
    for pair in sheet["pairs"]:
        pair.update(grade=0, reason="测试")
    if fault == "digest":
        sheet["dataset_sha256"] = "wrong"
    elif fault == "reviewer":
        sheet["reviewer"] = ""
    elif fault in ("blank_grade", "bool_grade"):
        sheet["pairs"][0]["grade"] = None if fault == "blank_grade" else True
    elif fault == "reason":
        sheet["pairs"][0]["reason"] = " "
    elif fault == "missing":
        sheet["pairs"].pop()
    else:
        sheet["pairs"].append(sheet["pairs"][0])
    with pytest.raises(ValueError):
        import_annotations(data, sheet)


def test_draft_labels_cannot_be_evaluated(data):
    draft = create_draft(data.model_dump())
    with pytest.raises(ValueError, match="unreviewed"):
        run_evaluation(draft)


@pytest.mark.parametrize("labels", ["synthetic_author", "model_assisted"])
def test_smoke_integration_never_opens_personal_stores(data, monkeypatch, tmp_path, labels):
    from resume_match.rag import coarse_ranker, retriever, embedder
    from resume_match.history import storage

    def forbidden(*args, **kwargs):
        raise AssertionError("personal store or real embedding must not be accessed")

    for module in (coarse_ranker, retriever):
        monkeypatch.setattr(module, "get_vector_store", forbidden)
        monkeypatch.setattr(module, "get_bm25_store", forbidden)
    monkeypatch.setattr(storage, "get_storage", forbidden)
    monkeypatch.setattr(embedder, "_get_model", forbidden)
    if labels == "model_assisted":
        data = Dataset.model_validate({**data.model_dump(), "provenance": "real", "label_source": labels})
    result = run_evaluation(data, backend="smoke", split="test", repeats=2)
    assert result["queries"] == 24 and result["documents"] == 32
    assert result["resume_claim_candidate"] is False
    assert result["embedding"]["logical_embedding_calls"] == 0
    assert set(result["summary"]) == set(METHODS)
    assert all(m["positive_queries"] == 22 for m in result["summary"].values())
    assert len(result["rows"]) == 24 * 5
    assert all(len(r["latency_ms"]) == 2 for r in result["rows"])
    write_report(result, data, tmp_path / "report")
    assert (tmp_path / "report/report.md").exists()
    with pytest.raises(FileExistsError):
        write_report(result, data, tmp_path / "report")


def test_strict_eval_does_not_hide_route_failure():
    from resume_match.rag.coarse_ranker import coarse_rank
    from resume_match.evaluation.runner import EmptyStore

    class Broken:
        def count(self):
            raise RuntimeError("broken index")

    with pytest.raises(RuntimeError, match="broken index"):
        coarse_rank(None, vector_store=Broken(), bm25_store=EmptyStore(), strict=True)
