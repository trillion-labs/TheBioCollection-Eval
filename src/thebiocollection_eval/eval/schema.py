"""Shared record schema and JSONL helpers for TheBioCollection-Eval."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class EvalRecord:
    id: str
    domain: str
    task: str
    instruction: str
    input: str
    target: Any
    metadata: dict[str, Any]
    task_group: str = "prediction"
    target_format: str = "text"
    metric: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "EvalRecord":
        metadata = row.get("metadata") or {}
        if isinstance(metadata, str):
            try:
                parsed_metadata = json.loads(metadata)
            except json.JSONDecodeError:
                parsed_metadata = {}
            metadata = parsed_metadata if isinstance(parsed_metadata, dict) else {}
        return cls(
            id=str(row["id"]),
            domain=str(row["domain"]),
            task=str(row["task"]),
            task_group=str(row.get("task_group", "prediction")),
            instruction=str(row.get("instruction", "")),
            input=str(row.get("input", "")),
            target=row.get("target"),
            target_format=str(row.get("target_format", "text")),
            metric=str(row["metric"]),
            metadata=dict(metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "domain": self.domain,
            "task": self.task,
            "task_group": self.task_group,
            "instruction": self.instruction,
            "input": self.input,
            "target": self.target,
            "target_format": self.target_format,
            "metric": self.metric,
            "metadata": self.metadata,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
    return records


def write_jsonl(path: Path, records: Iterable[dict[str, Any] | EvalRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            row = record.to_dict() if isinstance(record, EvalRecord) else record
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_records(path: Path) -> list[EvalRecord]:
    return [EvalRecord.from_dict(row) for row in read_jsonl(path)]


def load_predictions(
    path: Path,
    gold: list[EvalRecord],
    *,
    allow_extra: bool = False,
) -> dict[str, str]:
    rows = read_jsonl(path)
    gold_ids = {record.id for record in gold}
    has_explicit_id = any("id" in row and str(row.get("id") or "") for row in rows)
    if not has_explicit_id and len(rows) != len(gold):
        raise ValueError(
            f"{path} has {len(rows)} id-less prediction rows, but gold has {len(gold)} records."
        )

    predictions: dict[str, str] = {}
    for index, row in enumerate(rows):
        prediction = row.get("prediction", row.get("response", row.get("output")))
        if prediction is None:
            raise ValueError(f"{path} row {index + 1} has no prediction/response/output field.")
        record_id = str(row["id"]) if row.get("id") else gold[index].id
        if record_id not in gold_ids:
            if allow_extra and has_explicit_id:
                continue
            raise ValueError(
                f"{path} contains unknown prediction id: {record_id}. "
                "This usually means the predictions were generated for an older or different "
                "test.jsonl. Regenerate predictions, or merge fresh shards against the current "
                "test file before evaluation."
            )
        if record_id in predictions:
            raise ValueError(f"{path} contains duplicate prediction id: {record_id}")
        predictions[record_id] = str(prediction)

    missing = sorted(gold_ids - set(predictions))
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise ValueError(f"{path} is missing {len(missing)} prediction ids: {preview}{suffix}")
    return predictions
