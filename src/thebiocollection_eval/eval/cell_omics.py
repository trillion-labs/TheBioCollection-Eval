"""Cell/pathway metrics used by TheBioCollection-Eval."""

from __future__ import annotations

import json
import re
from statistics import mean
from typing import Any

from thebiocollection_eval.eval.schema import EvalRecord


TASK_CELL_MARKER = "cellmarker_celltype_recognition"
TASK_HALLMARK_PROGRAM = "hallmark_program_recognition"
TASK_PERTURBENCH_REPLOGLE = "perturbench_replogle_k562"

METRIC_DEG_DIRECTION_JSON = "perturbation_deg_direction_json"

_GENE_RE = re.compile(r"^[A-Z0-9_.-]+$")


def perturbation_deg_direction_json_metrics(
    pairs: list[tuple[EvalRecord, str]]
) -> dict[str, Any]:
    per_record: list[dict[str, float]] = []
    for record, prediction in pairs:
        gold = normalize_deg_gold_label(record.metadata.get("cell_omics_gold_label"))
        candidate_genes = [
            normalize_gene(gene)
            for gene in record.metadata.get("candidate_genes", [])
            if normalize_gene(gene)
        ]
        if not candidate_genes:
            candidate_genes = sorted(gold["up"] | gold["down"])
        pred = parse_deg_direction_json_prediction(prediction)
        per_record.append(single_deg_direction_metrics(gold, pred, candidate_genes))

    return {
        "direction_accuracy": _mean_metric(per_record, "direction_accuracy"),
        "macro_f1": _mean_metric(per_record, "macro_f1"),
        "candidate_coverage": _mean_metric(per_record, "candidate_coverage"),
    }


def parse_deg_direction_json_prediction(prediction: str) -> dict[str, set[str]]:
    text = str(prediction or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    try:
        parsed = json.loads(text)
    except Exception:
        parsed = {}
    if isinstance(parsed, dict):
        return normalize_deg_gold_label(parsed)
    return normalize_deg_gold_label({})


def normalize_deg_gold_label(value: Any) -> dict[str, set[str]]:
    gold = value if isinstance(value, dict) else {}
    return {
        "up": {gene for gene in (normalize_gene(gene) for gene in gold.get("up", [])) if gene},
        "down": {gene for gene in (normalize_gene(gene) for gene in gold.get("down", [])) if gene},
    }


def normalize_gene(value: Any) -> str:
    gene = str(value or "").strip().upper()
    return gene if _GENE_RE.match(gene) else ""


def single_deg_direction_metrics(
    gold: dict[str, set[str]], prediction: dict[str, set[str]], candidate_genes: list[str]
) -> dict[str, float]:
    candidate_set = set(candidate_genes)
    pred_up = prediction["up"] & candidate_set
    pred_down = prediction["down"] & candidate_set
    predicted = pred_up | pred_down

    correct = 0
    for gene in candidate_set:
        if gene in gold["up"] and gene in pred_up:
            correct += 1
        elif gene in gold["down"] and gene in pred_down:
            correct += 1

    up_f1 = _label_f1(gold["up"] & candidate_set, pred_up)
    down_f1 = _label_f1(gold["down"] & candidate_set, pred_down)
    return {
        "direction_accuracy": _safe_div(correct, len(candidate_set)),
        "macro_f1": mean([up_f1, down_f1]),
        "candidate_coverage": _safe_div(len(predicted), len(candidate_set)),
    }


def _label_f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    tp = len(gold & pred)
    precision = _safe_div(tp, len(pred))
    recall = _safe_div(tp, len(gold))
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean_metric(rows: list[dict[str, float]], key: str) -> float:
    values = [row[key] for row in rows if key in row]
    return mean(values) if values else 0.0


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0
