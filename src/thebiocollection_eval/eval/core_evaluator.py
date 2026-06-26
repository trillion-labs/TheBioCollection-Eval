"""Unified evaluator for TheBioCollection-Eval records."""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Any

from thebiocollection_eval.eval.cell_omics import perturbation_deg_direction_json_metrics
from thebiocollection_eval.eval.metrics import (
    binder_generation_metrics,
    interpro_overlap_metrics,
    label_exact_metrics,
    mollang_recognition_metrics,
    mol_instructions_forward_synthesis_metrics,
    protein_design_metrics,
    smiles_generation_metrics,
    span_localization_json_with_subsequence_metrics,
)
from thebiocollection_eval.eval.prediction_postprocess import postprocess_prediction
from thebiocollection_eval.eval.protein_deep_metrics import (
    ProteinDeepMetricConfig,
    ProteinDeepMetricScorer,
    validate_required_assets_for_metrics,
)
from thebiocollection_eval.eval.schema import EvalRecord, load_predictions, load_records


DEEP_PROTEIN_METRICS = {"protein_design", "binder_generation"}


def evaluate_core_records(
    records: list[EvalRecord],
    *,
    predictions: dict[str, str] | None = None,
    use_ground_truth: bool = False,
    protein_deep_metrics_config: ProteinDeepMetricConfig | None = None,
    protein_deep_metric_scorer: ProteinDeepMetricScorer | None = None,
) -> list[dict[str, Any]]:
    if predictions is None and use_ground_truth:
        predictions = {record.id: str(record.target) for record in records}
    deep_metric_names = {
        record.metric for record in records if record.metric in DEEP_PROTEIN_METRICS
    }
    if deep_metric_names:
        if protein_deep_metric_scorer is None:
            if protein_deep_metrics_config is None:
                raise RuntimeError(
                    "Protein design/binder evaluation requires deep protein metrics. "
                    "Pass a ProteinDeepMetricConfig with ESM-C and ESMFold2-Fast assets."
                )
            validate_required_assets_for_metrics(
                protein_deep_metrics_config,
                deep_metric_names,
            )
            protein_deep_metric_scorer = ProteinDeepMetricScorer(protein_deep_metrics_config)

    grouped: dict[str, list[EvalRecord]] = defaultdict(list)
    for record in records:
        grouped[record.task].append(record)

    results: list[dict[str, Any]] = []
    for task in sorted(grouped):
        gold = grouped[task]
        row: dict[str, Any] = {
            "domain": gold[0].domain,
            "task": task,
            "task_group": gold[0].task_group,
            "metric": gold[0].metric,
            "count": len(gold),
            "evaluation_status": "complete",
        }

        if predictions is not None:
            raw_pairs = [(record, predictions.get(record.id, "")) for record in gold]
            pairs = [
                (record, postprocess_prediction(record, prediction))
                for record, prediction in raw_pairs
            ]
            row["missing_predictions"] = sum(1 for _, pred in raw_pairs if pred == "")
            row["postprocessed_predictions"] = sum(
                1
                for (_, raw_prediction), (_, prediction) in zip(raw_pairs, pairs)
                if raw_prediction != prediction
            )
            row.update(evaluate_pairs(pairs, protein_deep_metric_scorer=protein_deep_metric_scorer))

        results.append(row)
    return results


def evaluate_pairs(
    pairs: list[tuple[EvalRecord, str]],
    *,
    protein_deep_metric_scorer: ProteinDeepMetricScorer | None = None,
) -> dict[str, Any]:
    if not pairs:
        return {}
    metric = pairs[0][0].metric
    if metric == "canonical_smiles_exact":
        return smiles_generation_metrics(pairs)
    if metric == "mol_instructions_forward_synthesis":
        return mol_instructions_forward_synthesis_metrics(pairs)
    if metric == "mollang_recognition_exact":
        return mollang_recognition_metrics(pairs)
    if metric == "protein_design":
        if protein_deep_metric_scorer is None:
            raise RuntimeError("protein_design requires deep protein metrics.")
        return protein_design_metrics(pairs, protein_deep_metric_scorer)
    if metric == "binder_generation":
        if protein_deep_metric_scorer is None:
            raise RuntimeError("binder_generation requires deep protein metrics.")
        return binder_generation_metrics(pairs, protein_deep_metric_scorer)
    if metric == "interpro_overlap":
        return interpro_overlap_metrics(pairs)
    if metric == "span_localization_json_with_subsequence":
        return span_localization_json_with_subsequence_metrics(pairs)
    if metric == "perturbation_deg_direction_json":
        return perturbation_deg_direction_json_metrics(pairs)
    if metric == "label_exact":
        return label_exact_metrics(pairs)
    raise ValueError(f"Unsupported metric: {metric}")


def evaluate_core_file(
    test_file: Path,
    *,
    predictions_file: Path | None = None,
    output_dir: Path,
    use_ground_truth: bool = False,
    protein_deep_metrics_config: ProteinDeepMetricConfig | None = None,
    task_filter: set[str] | None = None,
    record_id_regex: str | None = None,
) -> list[dict[str, Any]]:
    records = load_records(test_file)
    if task_filter:
        records = [record for record in records if record.task in task_filter]
    if record_id_regex:
        pattern = re.compile(record_id_regex)
        records = [record for record in records if pattern.search(record.id)]
    predictions = (
        load_predictions(
            predictions_file,
            records,
            allow_extra=bool(task_filter or record_id_regex),
        )
        if predictions_file
        else None
    )
    if (
        protein_deep_metrics_config is not None
        and protein_deep_metrics_config.trace_dir is None
    ):
        protein_deep_metrics_config = replace(
            protein_deep_metrics_config,
            trace_dir=output_dir / "protein_deep_metrics",
        )
    results = evaluate_core_records(
        records,
        predictions=predictions,
        use_ground_truth=use_ground_truth,
        protein_deep_metrics_config=protein_deep_metrics_config,
    )
    write_summary(results, output_dir)
    return results


def write_summary(results: list[dict[str, Any]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    fieldnames = sorted({key for row in results for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    with (output_dir / "summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# TheBioCollection-Eval Summary\n\n")
        for row in results:
            handle.write(f"## {row['domain']} / {row['task']}\n\n")
            for key in fieldnames:
                if key in {"domain", "task"} or row.get(key) is None:
                    continue
                handle.write(f"- `{key}`: {row[key]}\n")
            handle.write("\n")
