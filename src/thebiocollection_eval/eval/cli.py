"""Command-line interface for TheBioCollection-Eval scoring."""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

from thebiocollection_eval.eval.core_evaluator import (
    DEEP_PROTEIN_METRICS,
    evaluate_core_records,
    write_summary,
)
from thebiocollection_eval.eval.protein_deep_metrics import (
    DEFAULT_ESMC_MODEL,
    DEFAULT_ESMFOLD2_MODEL,
    ProteinDeepMetricConfig,
)
from thebiocollection_eval.eval.schema import EvalRecord, load_predictions, load_records


DEFAULT_DATASET = "trillionlabs/TheBioCollection-Eval"


def env_path(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


def load_hf_records(
    dataset_id: str,
    *,
    config_name: str,
    split: str,
    cache_dir: Path | None = None,
) -> list[EvalRecord]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - depends on environment.
        raise RuntimeError(
            "Loading the default Hugging Face dataset requires the `datasets` package. "
            "Install with `pip install -r requirements.txt`, or pass --test-file."
        ) from exc

    kwargs = {"split": split}
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    if config_name == "default":
        dataset = load_dataset(dataset_id, **kwargs)
    else:
        dataset = load_dataset(dataset_id, config_name, **kwargs)
    return [EvalRecord.from_dict(dict(row)) for row in dataset]


def filter_records(
    records: list[EvalRecord],
    *,
    tasks: list[str],
    record_id_regex: str | None,
) -> list[EvalRecord]:
    if tasks:
        wanted = set(tasks)
        records = [record for record in records if record.task in wanted]
    if record_id_regex:
        pattern = re.compile(record_id_regex)
        records = [record for record in records if pattern.search(record.id)]
    if not records:
        raise ValueError("No evaluation records remain after filtering.")
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate predictions for TheBioCollection-Eval.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="Hugging Face dataset id to load when --test-file is not provided.",
    )
    parser.add_argument(
        "--dataset-config",
        default="default",
        help="Dataset config to load, e.g. protein__binder_generation.",
    )
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Optional Hugging Face datasets cache directory.",
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Optional local JSONL test file. If set, --dataset is ignored.",
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        help=(
            "Prediction JSONL with one row per eval record. Each row should contain "
            "`id` and one of `prediction`, `response`, or `output`."
        ),
    )
    parser.add_argument(
        "--task",
        action="append",
        default=[],
        help="Restrict evaluation to this task. Can be repeated.",
    )
    parser.add_argument(
        "--record-id-regex",
        help="Restrict evaluation to record IDs matching this Python regex.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("eval_outputs/thebiocollection_eval"),
        help="Directory where summary.json, summary.csv, and summary.md are written.",
    )
    parser.add_argument(
        "--use-ground-truth",
        action="store_true",
        help="Score gold targets as predictions. Useful for smoke tests.",
    )
    parser.add_argument(
        "--esmc-model",
        default=os.environ.get("ESMC_MODEL", DEFAULT_ESMC_MODEL),
        help=(
            "ESM-C model id or local checkpoint path. Required for protein_design "
            "functional_consistency."
        ),
    )
    parser.add_argument(
        "--esmfold2-model",
        default=os.environ.get("ESMFOLD2_MODEL", DEFAULT_ESMFOLD2_MODEL),
        help=(
            "ESMFold2-Fast model id or local checkpoint path. Required for protein_design "
            "and binder_generation fold-confidence metrics."
        ),
    )
    parser.add_argument(
        "--protein-device",
        default=os.environ.get("PROTEIN_DEVICE", "cuda"),
        help="Device for ESM-C and ESMFold2-Fast, e.g. cuda, cuda:0, cpu, or auto.",
    )
    parser.add_argument(
        "--protein-dtype",
        default=os.environ.get("PROTEIN_DTYPE", "auto"),
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Torch dtype for ESMFold2-Fast model loading.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        type=Path,
        default=env_path("HF_CACHE_DIR"),
        help="Optional Hugging Face model cache directory for ESM-C/ESMFold2-Fast.",
    )
    parser.add_argument(
        "--esm-c-embedding-cache-dir",
        type=Path,
        default=env_path("ESM_C_EMBEDDING_CACHE_DIR"),
        help="Optional cache directory for ESM-C sequence embeddings.",
    )
    parser.add_argument(
        "--esmfold2-python",
        type=Path,
        default=env_path("ESMFOLD2_PYTHON"),
        help="Optional Python executable for an isolated ESMFold2-Fast runtime.",
    )
    parser.add_argument(
        "--esmfold2-timeout-sec",
        type=int,
        default=int(os.environ.get("ESMFOLD2_TIMEOUT_SEC", "3600")),
    )
    parser.add_argument(
        "--esmfold2-num-loops",
        type=int,
        default=int(os.environ.get("ESMFOLD2_NUM_LOOPS", "3")),
    )
    parser.add_argument(
        "--esmfold2-num-sampling-steps",
        type=int,
        default=int(os.environ.get("ESMFOLD2_NUM_SAMPLING_STEPS", "50")),
    )
    parser.add_argument(
        "--esmfold2-num-diffusion-samples",
        type=int,
        default=int(os.environ.get("ESMFOLD2_NUM_DIFFUSION_SAMPLES", "1")),
    )
    parser.add_argument(
        "--esmfold2-seed",
        type=int,
        default=int(os.environ.get("ESMFOLD2_SEED", "0")),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.use_ground_truth and args.predictions is None:
        raise SystemExit("Provide --predictions, or pass --use-ground-truth for a smoke test.")

    if args.test_file is not None:
        records = load_records(args.test_file)
    else:
        records = load_hf_records(
            args.dataset,
            config_name=args.dataset_config,
            split=args.split,
            cache_dir=args.cache_dir,
        )
    records = filter_records(
        records,
        tasks=args.task,
        record_id_regex=args.record_id_regex,
    )

    predictions = None
    if args.use_ground_truth:
        predictions = {record.id: str(record.target) for record in records}
    elif args.predictions is not None:
        predictions = load_predictions(
            args.predictions,
            records,
            allow_extra=bool(args.task or args.record_id_regex),
        )

    protein_deep_metrics_config = None
    if any(record.metric in DEEP_PROTEIN_METRICS for record in records):
        protein_deep_metrics_config = ProteinDeepMetricConfig(
            enabled=True,
            esmc_model=args.esmc_model,
            esmfold2_model=args.esmfold2_model,
            protein_device=args.protein_device,
            protein_dtype=args.protein_dtype,
            hf_cache_dir=args.hf_cache_dir,
            esm_c_embedding_cache_dir=(
                args.esm_c_embedding_cache_dir
                if args.esm_c_embedding_cache_dir is not None
                else args.output_dir / "_esm_c_embedding_cache"
            ),
            esmfold2_python=args.esmfold2_python,
            esmfold2_timeout_sec=args.esmfold2_timeout_sec,
            esmfold2_num_loops=args.esmfold2_num_loops,
            esmfold2_num_sampling_steps=args.esmfold2_num_sampling_steps,
            esmfold2_num_diffusion_samples=args.esmfold2_num_diffusion_samples,
            esmfold2_seed=args.esmfold2_seed,
            trace_dir=args.output_dir / "protein_deep_metrics",
        )

    try:
        results = evaluate_core_records(
            records,
            predictions=predictions,
            protein_deep_metrics_config=protein_deep_metrics_config,
        )
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    write_summary(results, args.output_dir)
    print(f"[done] evaluated {len(records)} records; wrote summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
