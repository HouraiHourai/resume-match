"""离线人工标注：draft生成待标注数据，export盲标，import校验并写新文件。"""
import argparse
import copy
import json
from pathlib import Path

from resume_match.evaluation.dataset import Dataset, dataset_digest, load_dataset


def create_draft(raw: dict) -> Dataset:
    data = copy.deepcopy(raw)
    data.update(label_source="unreviewed", reviewer="")
    ids = [d["id"] for d in data["documents"]]
    for query in data["queries"]:
        query.update(relevance={id_: 0 for id_ in ids if id_ not in query.get("excluded_ids", [])},
                     rationale="未标注：0仅为占位符，禁止用于效果评测。",
                     human_reviewed=False, relevance_reasons={})
    return Dataset.model_validate(data)


def export_annotations(data: Dataset) -> dict:
    # 固定ID顺序，不暴露模型排名或原草稿标签；使用原文与结构化字段一起判断。
    pairs = []
    for query in sorted(data.queries, key=lambda q: q.id):
        for doc in sorted(data.documents, key=lambda d: d.id):
            if doc.id in query.excluded_ids:
                continue
            pairs.append({"query_id": query.id, "document_id": doc.id,
                          "query_text": query.text, "document_text": doc.text,
                          "query_jd": query.jd.model_dump(), "document_jd": doc.jd.model_dump(),
                          "grade": None, "reason": ""})
    return {"schema_version": 1, "dataset_sha256": dataset_digest(data), "reviewer": "",
            "rubric": "2=职责、关键技能及资历高度相关；1=部分相关、可参考；0=不相关。判断相似历史岗位，不判断录用率。",
            "pairs": pairs}


def import_annotations(data: Dataset, annotation: dict) -> Dataset:
    if annotation.get("schema_version") != 1 or annotation.get("dataset_sha256") != dataset_digest(data):
        raise ValueError("annotation schema/dataset digest mismatch; export again after changing data")
    reviewer = annotation.get("reviewer", "")
    if not isinstance(reviewer, str) or not reviewer.strip():
        raise ValueError("reviewer identity/alias is required")
    expected = {(q.id, id_) for q in data.queries for id_ in q.relevance}
    judgments = {}
    for pair in annotation.get("pairs", []):
        key = (pair["query_id"], pair["document_id"])
        if key not in expected or key in judgments:
            raise ValueError("unknown or duplicate annotation pair")
        grade, reason = pair.get("grade"), pair.get("reason")
        if type(grade) is not int or grade not in (0, 1, 2):
            raise ValueError(f"{key}: grade must be integer 0/1/2; blanks are not negative labels")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError(f"{key}: a short reason is required")
        judgments[key] = grade, reason.strip()
    if set(judgments) != expected:
        raise ValueError("missing annotation pairs; review every non-excluded document")
    result = data.model_dump()
    result.update(label_source="human", reviewer=reviewer.strip())
    # 不允许通过人工标注把合成来源改成真实来源。
    for query in result["queries"]:
        query["relevance"] = {id_: judgments[query["id"], id_][0] for id_ in query["relevance"]}
        query["relevance_reasons"] = {id_: judgments[query["id"], id_][1] for id_ in query["relevance"]}
        query.update(human_reviewed=True, rationale="人工逐对标注，依据见relevance_reasons。")
    return Dataset.model_validate(result)


def write_new(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    draft = commands.add_parser("draft", help="原始JD集合转待标注数据，占位0不能用于评测")
    draft.add_argument("--input", type=Path, required=True)
    draft.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export", help="导出不含已有标签与排名的盲标JSON")
    export.add_argument("--dataset", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    ingest = commands.add_parser("import", help="导入人工填写的JSON并生成新数据集")
    ingest.add_argument("--dataset", type=Path, required=True)
    ingest.add_argument("--annotations", type=Path, required=True)
    ingest.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "draft":
        result = create_draft(json.loads(args.input.read_text(encoding="utf-8"))).model_dump()
    elif args.command == "export":
        result = export_annotations(load_dataset(args.dataset))
    else:
        result = import_annotations(load_dataset(args.dataset),
            json.loads(args.annotations.read_text(encoding="utf-8"))).model_dump()
    write_new(args.output, result)
    print(f"Written: {args.output.resolve()}")


if __name__ == "__main__":
    main()
