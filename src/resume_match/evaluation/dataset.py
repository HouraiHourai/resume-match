"""评测数据及人工标注的显式来源；严格拒绝重复、自匹配和不完整标签。"""
import hashlib
import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from resume_match.state import JDRequirements


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(min_length=1)
    source: str = Field(min_length=1, description="真实来源或明确synthetic标记，勿填私人简历")
    text: str = Field(min_length=1)
    jd: JDRequirements

    @field_validator("id", "source", "text")
    @classmethod
    def not_blank(cls, value):
        if not value.strip():
            raise ValueError("must not be blank")
        return value


class Query(Record):
    split: Literal["dev", "test"]
    # 所有非排除候选必须给0/1/2分，避免把未标注候选暗中当不相关。
    relevance: dict[str, Literal[0, 1, 2]]
    excluded_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    human_reviewed: bool = False
    relevance_reasons: dict[str, str] = Field(default_factory=dict)


class Dataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    name: str
    provenance: Literal["synthetic", "real"]
    label_source: Literal["unreviewed", "synthetic_author", "model_assisted", "human"]
    reviewer: str = ""
    description: str
    documents: list[Record] = Field(min_length=3)
    queries: list[Query] = Field(min_length=1)

    @model_validator(mode="after")
    def check_integrity(self):
        ids = [d.id for d in self.documents]
        qids = [q.id for q in self.queries]
        if len(set(ids)) != len(ids) or len(set(qids)) != len(qids):
            raise ValueError("document/query IDs must be unique within their collection")
        doc_ids = set(ids)
        texts = {normalized(d.text) for d in self.documents}
        if len(texts) != len(self.documents):
            raise ValueError("duplicate document text")
        if len({normalized(q.text) for q in self.queries}) != len(self.queries):
            raise ValueError("duplicate query text; do not inflate sample counts")
        if len({embedding_key(q.jd) for q in self.queries}) != len(self.queries):
            raise ValueError("duplicate query embedding inputs; possible dev/test leakage")
        for q in self.queries:
            excluded = set(q.excluded_ids)
            if not excluded <= doc_ids or len(excluded) != len(q.excluded_ids):
                raise ValueError(f"{q.id}: invalid excluded IDs")
            if set(q.relevance) != doc_ids - excluded:
                raise ValueError(f"{q.id}: relevance must judge every non-excluded document exactly once")
            if q.relevance_reasons and (set(q.relevance_reasons) != set(q.relevance)
                                       or not all(r.strip() for r in q.relevance_reasons.values())):
                raise ValueError(f"{q.id}: reasons must cover every judgment and not be blank")
            for d in self.documents:
                if d.id in excluded:
                    continue
                if normalized(q.text) == normalized(d.text):
                    raise ValueError(f"{q.id}: exact query/document leakage; exclude {d.id}")
                # 模型实际看到的是结构化字段，也要排除仅改原文的重复。
                if embedding_key(q.jd) == embedding_key(d.jd):
                    raise ValueError(f"{q.id}: identical embedding input; exclude {d.id}")
        if self.label_source == "human" and not self.reviewer.strip():
            raise ValueError("human labels require a reviewer identity/alias")
        return self


def normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def embedding_key(jd: JDRequirements) -> str:
    from resume_match.rag.embedder import build_embedding_text_from_dict
    return normalized(build_embedding_text_from_dict(jd.model_dump()))


def load_dataset(path: Path) -> Dataset:
    return Dataset.model_validate_json(path.read_text(encoding="utf-8"))


def dataset_digest(data: Dataset) -> str:
    raw = json.dumps(data.model_dump(), ensure_ascii=False, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()
