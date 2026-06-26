"""Deep-model metrics for protein generation subtasks.

The public protein-generation metrics use ESM-C sequence embeddings and
ESMFold2-Fast structure confidence. These metrics are required for protein
design and binder-generation evaluation.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from thebiocollection_eval.eval.metrics import (
    VALID_AA,
    extract_protein_sequence,
    protein_degeneracy_stats,
)
from thebiocollection_eval.eval.schema import EvalRecord


DEFAULT_ESMC_MODEL = "biohub/esmc-600m-2024-12"
DEFAULT_ESMFOLD2_MODEL = "biohub/ESMFold2-Fast"

ESMC_ARCHITECTURES = {
    "esmc_300m": {
        "d_model": 960,
        "n_heads": 15,
        "n_layers": 30,
        "repo": "models--biohub--esmc-300m-2024-12",
        "pattern": "esmc_300m*_v0.pth",
    },
    "esmc_600m": {
        "d_model": 1152,
        "n_heads": 18,
        "n_layers": 36,
        "repo": "models--biohub--esmc-600m-2024-12",
        "pattern": "esmc_600m*_v0.pth",
    },
    "esmc_6b": {
        "d_model": 2560,
        "n_heads": 40,
        "n_layers": 80,
        "repo": "models--biohub--esmc-6b-2024-12",
        "pattern": "esmc_6b*_v0.pth",
    },
}

ESMC_MODEL_ALIASES = {
    "esmc_300m": "esmc_300m",
    "esmc-300m": "esmc_300m",
    "biohub/esmc-300m-2024-12": "esmc_300m",
    "esmc_600m": "esmc_600m",
    "esmc-600m": "esmc_600m",
    "biohub/esmc-600m-2024-12": "esmc_600m",
    "esmc_6b": "esmc_6b",
    "esmc-6b": "esmc_6b",
    "biohub/esmc-6b-2024-12": "esmc_6b",
}


@dataclass(frozen=True)
class ProteinDeepMetricConfig:
    enabled: bool = True
    esmc_model: str = DEFAULT_ESMC_MODEL
    esmfold2_model: str = DEFAULT_ESMFOLD2_MODEL
    protein_device: str = "cuda"
    protein_dtype: str = "auto"
    hf_cache_dir: Path | None = None
    esm_c_embedding_cache_dir: Path | None = None
    esmfold2_python: Path | None = None
    esmfold2_timeout_sec: int = 3600
    esmfold2_num_loops: int = 3
    esmfold2_num_sampling_steps: int = 50
    esmfold2_num_diffusion_samples: int = 1
    esmfold2_seed: int = 0
    trace_dir: Path | None = None


def validate_required_assets_for_metrics(
    config: ProteinDeepMetricConfig,
    metrics: set[str],
) -> None:
    requested = metrics & {"protein_design", "binder_generation"}
    if not requested:
        return
    errors: list[str] = []
    if not config.enabled:
        errors.append("ProteinDeepMetricConfig.enabled must be True.")
    if "protein_design" in requested and not str(config.esmc_model or "").strip():
        errors.append("Missing ESM-C model; set --esmc-model or ESMC_MODEL.")
    if not str(config.esmfold2_model or "").strip():
        errors.append("Missing ESMFold2-Fast model; set --esmfold2-model or ESMFOLD2_MODEL.")
    if config.esmfold2_python is not None and not config.esmfold2_python.expanduser().is_file():
        errors.append(f"ESMFold2 Python runtime not found: {config.esmfold2_python}")
    for label, model in (
        ("ESM-C model", config.esmc_model),
        ("ESMFold2-Fast model", config.esmfold2_model),
    ):
        if _is_local_path_string(model) and not Path(model).expanduser().exists():
            errors.append(f"{label} path not found: {model}")
    if errors:
        metric_list = ", ".join(sorted(requested))
        assets = ["ESMFold2-Fast"]
        if "protein_design" in requested:
            assets.insert(0, "ESM-C")
        raise RuntimeError(
            f"Protein deep metrics are mandatory for {metric_list}. "
            f"Install/download the {' and '.join(assets)} checkpoints and pass their paths. "
            + " ".join(errors)
        )


class ProteinDeepMetricScorer:
    def __init__(self, config: ProteinDeepMetricConfig):
        if not config.enabled:
            raise ValueError("ProteinDeepMetricScorer requires enabled=True.")
        self.config = config
        self._esmc: ESMCEmbeddingClient | None = None
        self._esmfold2: ESMFold2FastClient | None = None

    def score_protein_design(self, pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
        cosine_scores: list[float] = []
        plddt_scores: list[float] = []
        ptm_scores: list[float] = []
        traces: list[dict[str, Any]] = []

        for record, prediction in pairs:
            pred_seq = extract_valid_sequence(prediction)
            gold_seq = extract_valid_sequence(str(record.target))
            trace: dict[str, Any] = {
                "id": record.id,
                "task": record.task,
                "gold_sequence_length": len(gold_seq) if gold_seq else 0,
                "prediction_sequence_length": len(pred_seq) if pred_seq else 0,
                **trace_degeneracy(pred_seq),
            }
            if not pred_seq:
                trace["skipped_reason"] = "invalid_or_missing_prediction_sequence"
                traces.append(trace)
                continue
            if gold_seq:
                start = time.perf_counter()
                cosine = self.esmc.cosine(gold_seq, pred_seq)
                trace["esm_c_gold_generated_cosine"] = cosine
                trace["esm_c_embedding_elapsed_sec"] = time.perf_counter() - start
                trace["esm_c_model_name"] = self.config.esmc_model
                cosine_scores.append(cosine)

            fold = self.esmfold2.fold_monomer(
                pred_seq,
                sample_id=safe_sample_name(record.id),
            )
            if fold.get("pLDDT") is not None:
                plddt_scores.append(float(fold["pLDDT"]))
            if fold.get("pTM") is not None:
                ptm_scores.append(float(fold["pTM"]))
            trace.update(fold)
            traces.append(trace)

        self.write_trace("mol_instructions_protein_design", traces)
        return {
            "fold_backend": "esmfold2_fast",
            "fold_model_name": self.config.esmfold2_model,
            "functional_consistency": safe_mean(cosine_scores),
            "pLDDT": safe_mean(plddt_scores),
            "pTM": safe_mean(ptm_scores),
        }

    def score_binder_generation(self, pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
        complex_plddt_scores: list[float] = []
        iptm_scores: list[float] = []
        traces: list[dict[str, Any]] = []

        for record, prediction in pairs:
            binder_seq = extract_valid_sequence(prediction)
            target_seq = extract_valid_sequence(
                str(
                    record.metadata.get("target_sequence")
                    or record.metadata.get("partner_sequence")
                    or ""
                )
            )
            target_smiles = clean_optional_string(record.metadata.get("smiles"))
            target_ccd_code = clean_optional_string(record.metadata.get("ccd_code"))
            trace: dict[str, Any] = {
                "id": record.id,
                "task": record.task,
                "target_type": record.metadata.get("target_type"),
                "target_sequence_length": len(target_seq) if target_seq else 0,
                "target_smiles_present": bool(target_smiles),
                "target_ccd_code": target_ccd_code or None,
                "binder_sequence_length": len(binder_seq) if binder_seq else 0,
                **trace_degeneracy(binder_seq),
            }
            if not binder_seq:
                trace["skipped_reason"] = "invalid_or_missing_binder_sequence"
                traces.append(trace)
                continue

            if target_seq:
                trace["complex_target_kind"] = "protein"
                fold = self.esmfold2.fold_protein_complex(
                    target_seq,
                    binder_seq,
                    sample_id=safe_sample_name(record.id),
                )
            elif target_smiles or target_ccd_code:
                trace["complex_target_kind"] = "ligand"
                fold = self.esmfold2.fold_ligand_complex(
                    binder_seq,
                    sample_id=safe_sample_name(record.id),
                    smiles=target_smiles or None,
                    ccd_code=target_ccd_code or None,
                )
            else:
                trace["skipped_reason"] = "invalid_or_missing_complex_target"
                traces.append(trace)
                continue

            if fold.get("Complex_pLDDT") is not None:
                complex_plddt_scores.append(float(fold["Complex_pLDDT"]))
            if fold.get("ipTM") is not None:
                iptm_scores.append(float(fold["ipTM"]))
            trace.update(fold)
            traces.append(trace)

        self.write_trace("prolangbench_binder_generation", traces)
        return {
            "fold_backend": "esmfold2_fast",
            "fold_model_name": self.config.esmfold2_model,
            "Complex_pLDDT": safe_mean(complex_plddt_scores),
            "ipTM": safe_mean(iptm_scores),
        }

    @property
    def esmc(self) -> "ESMCEmbeddingClient":
        if self._esmc is None:
            self._esmc = ESMCEmbeddingClient(self.config)
        return self._esmc

    @property
    def esmfold2(self) -> "ESMFold2FastClient":
        if self._esmfold2 is None:
            self._esmfold2 = ESMFold2FastClient(self.config)
        return self._esmfold2

    def write_trace(self, task: str, rows: list[dict[str, Any]]) -> None:
        if self.config.trace_dir is None:
            return
        self.config.trace_dir.mkdir(parents=True, exist_ok=True)
        path = self.config.trace_dir / f"{task}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")


class ESMCEmbeddingClient:
    def __init__(self, config: ProteinDeepMetricConfig):
        self.config = config
        self.model_name = config.esmc_model
        self.cache_dir = config.esm_c_embedding_cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import torch
            from accelerate import init_empty_weights
            from esm.models.esmc import ESMC
            from esm.tokenization import get_esmc_model_tokenizers
        except Exception as exc:
            raise RuntimeError(
                "Could not import local ESM-C dependencies. Install torch, accelerate, "
                "and the Biohub ESM package stack."
            ) from exc
        self.torch = torch
        self.device = resolve_device(torch, config.protein_device)
        checkpoint_path, architecture = resolve_local_esmc_checkpoint(
            self.model_name,
            hf_cache_dir=config.hf_cache_dir,
        )
        spec = ESMC_ARCHITECTURES[architecture]
        with init_empty_weights():
            model = ESMC(
                d_model=spec["d_model"],
                n_heads=spec["n_heads"],
                n_layers=spec["n_layers"],
                tokenizer=get_esmc_model_tokenizers(),
                use_flash_attn=False,
            ).eval()
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state_dict, strict=False, assign=True)
        model = model.to(self.device)
        if self.device != "cpu":
            model = model.to(torch.bfloat16)
        self.model = model.eval()
        self.tokenizer = model.tokenizer
        self.checkpoint_path = checkpoint_path

    def cosine(self, left_sequence: str, right_sequence: str) -> float:
        left = self.embed(left_sequence)
        right = self.embed(right_sequence)
        score = self.torch.nn.functional.cosine_similarity(left, right, dim=0).item()
        return float(score)

    def embed(self, sequence: str) -> Any:
        cache_path = self._cache_path(sequence)
        if cache_path is not None and cache_path.is_file():
            return self.torch.load(cache_path, map_location="cpu").float()
        token_ids = self.tokenizer.encode(sequence)
        sequence_tokens = self.torch.tensor(
            token_ids,
            dtype=self.torch.long,
            device=self.device,
        ).unsqueeze(0)
        with self.torch.inference_mode():
            output = self.model(sequence_tokens=sequence_tokens)
        hidden = getattr(output, "embeddings", None)
        if hidden is None:
            raise RuntimeError("ESM-C output did not include hidden states.")
        hidden = hidden[0]
        special_ids = set(getattr(self.tokenizer, "all_special_ids", []))
        valid = self.torch.tensor(
            [token_id not in special_ids for token_id in token_ids],
            dtype=self.torch.bool,
            device=hidden.device,
        )
        embedding = hidden[valid].mean(dim=0).detach().float().cpu()
        if cache_path is not None:
            self.torch.save(embedding, cache_path)
        return embedding

    def _cache_path(self, sequence: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            f"{self.checkpoint_path}\n{sequence}".encode("utf-8")
        ).hexdigest()
        subdir = self.cache_dir / digest[:2]
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir / f"{digest}.pt"


class ESMFold2FastClient:
    def __init__(self, config: ProteinDeepMetricConfig):
        self.config = config
        self.model_name = config.esmfold2_model
        self.python = (
            config.esmfold2_python.expanduser()
            if config.esmfold2_python is not None
            else Path(sys.executable)
        )

    def fold_monomer(self, sequence: str, *, sample_id: str) -> dict[str, Any]:
        result = self._run(
            sample_id=sample_id,
            payload={"sequence": sequence},
        )
        return {
            "pLDDT": result.get("plddt"),
            "pTM": result.get("ptm"),
            "fold_backend": "esmfold2_fast",
            "esmfold2_model_name": self.model_name,
            "esmfold2_elapsed_sec": result.get("elapsed_sec"),
        }

    def fold_protein_complex(
        self,
        target_sequence: str,
        binder_sequence: str,
        *,
        sample_id: str,
    ) -> dict[str, Any]:
        result = self._run(
            sample_id=sample_id,
            payload={
                "target_sequence": target_sequence,
                "binder_sequence": binder_sequence,
            },
        )
        return {
            "Complex_pLDDT": result.get("plddt"),
            "ipTM": result.get("iptm"),
            "fold_backend": "esmfold2_fast",
            "esmfold2_model_name": self.model_name,
            "esmfold2_elapsed_sec": result.get("elapsed_sec"),
        }

    def fold_ligand_complex(
        self,
        binder_sequence: str,
        *,
        sample_id: str,
        smiles: str | None = None,
        ccd_code: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"binder_sequence": binder_sequence}
        if smiles:
            payload["ligand_smiles"] = smiles
        if ccd_code:
            payload["ligand_ccd"] = ccd_code
        result = self._run(sample_id=sample_id, payload=payload)
        return {
            "Complex_pLDDT": result.get("plddt"),
            "ipTM": result.get("iptm"),
            "fold_backend": "esmfold2_fast",
            "esmfold2_model_name": self.model_name,
            "esmfold2_elapsed_sec": result.get("elapsed_sec"),
        }

    def _run(self, *, sample_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        work_dir = self._sample_dir(sample_id)
        input_json = work_dir / "esmfold2_input.json"
        output_json = work_dir / "esmfold2_output.json"
        if output_json.is_file():
            cached = json.loads(output_json.read_text(encoding="utf-8"))
            if cached.get("ok"):
                return dict(cached["output"])

        request = {
            **payload,
            "model_name": self.model_name,
            "device": self.config.protein_device,
            "dtype": self.config.protein_dtype,
            "cache_dir": str(self.config.hf_cache_dir) if self.config.hf_cache_dir else None,
            "local_files_only": True,
            "num_loops": self.config.esmfold2_num_loops,
            "num_sampling_steps": self.config.esmfold2_num_sampling_steps,
            "num_diffusion_samples": self.config.esmfold2_num_diffusion_samples,
            "seed": self.config.esmfold2_seed,
        }
        work_dir.mkdir(parents=True, exist_ok=True)
        input_json.write_text(json.dumps(request, indent=2), encoding="utf-8")
        runner = Path(__file__).resolve().parents[3] / "scripts" / "run_esmfold2_fast.py"
        command = [
            str(self.python),
            str(runner),
            "--input-json",
            str(input_json),
            "--output-json",
            str(output_json),
            "--output-dir",
            str(work_dir / "esmfold2_artifacts"),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            part for part in [str(Path(__file__).resolve().parents[2]), env.get("PYTHONPATH", "")]
            if part
        )
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            capture_output=True,
            env=env,
            timeout=self.config.esmfold2_timeout_sec,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "ESMFold2-Fast failed "
                f"for {sample_id}: {completed.stderr or completed.stdout}"
            )
        result = json.loads(output_json.read_text(encoding="utf-8"))
        if not result.get("ok"):
            raise RuntimeError(f"ESMFold2-Fast failed for {sample_id}: {result.get('error')}")
        output = dict(result["output"])
        output.setdefault("elapsed_sec", time.perf_counter() - start)
        return output

    def _sample_dir(self, sample_id: str) -> Path:
        root = self.config.trace_dir or (Path.cwd() / "protein_deep_metrics")
        return root / "esmfold2" / sample_id


def trace_degeneracy(sequence: str) -> dict[str, Any]:
    stats = protein_degeneracy_stats(sequence)
    return {
        "is_degenerate": stats["is_degenerate"],
        "degeneracy_reasons": stats["degeneracy_reasons"],
        "normalized_aa_entropy": stats["normalized_aa_entropy"],
        "max_aa_fraction": stats["max_aa_fraction"],
        "max_kmer_fraction": stats["max_kmer_fraction"],
        "min_unique_kmer_fraction": stats["min_unique_kmer_fraction"],
        "max_homopolymer_fraction": stats["max_homopolymer_fraction"],
    }


def extract_valid_sequence(text: str) -> str:
    sequence = extract_protein_sequence(text)
    if not sequence or set(sequence) - VALID_AA:
        return ""
    return sequence


def clean_optional_string(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text.lower() in {"none", "null", "nan"}:
        return ""
    return text


def safe_sample_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    return f"{safe[:80]}_{digest}"


def safe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def resolve_local_esmc_checkpoint(
    model: str | Path,
    *,
    hf_cache_dir: Path | None = None,
) -> tuple[Path, str]:
    text = str(model).strip()
    if not text:
        raise RuntimeError("Missing ESM-C checkpoint path or local model alias.")
    path = Path(text).expanduser()
    if _is_local_path_string(text):
        if not path.exists():
            raise RuntimeError(f"Local ESM-C path not found: {path}")
        checkpoint = find_esmc_checkpoint_in_path(path)
        architecture = infer_esmc_architecture(checkpoint)
        return checkpoint, architecture

    alias = ESMC_MODEL_ALIASES.get(text.lower())
    if alias is None:
        raise RuntimeError(
            "ESM-C must be a local .pth file, a local snapshot directory, or one of "
            f"{sorted(ESMC_MODEL_ALIASES)} resolvable from the local Hugging Face cache."
        )
    cache_root = resolve_hf_hub_cache_dir(hf_cache_dir)
    repo_dir = cache_root / str(ESMC_ARCHITECTURES[alias]["repo"])
    snapshots_dir = repo_dir / "snapshots"
    if not snapshots_dir.is_dir():
        raise RuntimeError(
            f"Local ESM-C cache not found for {text}: expected {snapshots_dir}. "
            "Download the checkpoint before running evaluation or pass a local .pth path."
        )
    candidates = sorted(
        snapshots_dir.glob(f"*/data/weights/{ESMC_ARCHITECTURES[alias]['pattern']}"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise RuntimeError(
            f"Local ESM-C weight file not found under {snapshots_dir}; expected "
            f"data/weights/{ESMC_ARCHITECTURES[alias]['pattern']}."
        )
    return candidates[0].resolve(), alias


def find_esmc_checkpoint_in_path(path: Path) -> Path:
    if path.is_file():
        if path.suffix != ".pth":
            raise RuntimeError(f"ESM-C local checkpoint must be a .pth file: {path}")
        return path.absolute()
    candidates = sorted(
        path.glob("data/weights/esmc_*_v0.pth"),
        key=lambda candidate: candidate.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            path.glob("**/esmc_*_v0.pth"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
    if not candidates:
        raise RuntimeError(f"No ESM-C .pth checkpoint found under {path}")
    return candidates[0].absolute()


def infer_esmc_architecture(checkpoint: Path) -> str:
    name = checkpoint.name.lower()
    if "300m" in name:
        return "esmc_300m"
    if "600m" in name:
        return "esmc_600m"
    if "6b" in name:
        return "esmc_6b"
    raise RuntimeError(
        f"Could not infer ESM-C architecture from checkpoint name: {checkpoint.name}"
    )


def resolve_hf_hub_cache_dir(hf_cache_dir: Path | None) -> Path:
    if hf_cache_dir is not None:
        cache = hf_cache_dir.expanduser()
        if cache.name == "hub":
            return cache
        return cache / "hub"
    env_cache = os.environ.get("HF_HUB_CACHE")
    if env_cache:
        return Path(env_cache).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


def _is_local_path_string(value: str | Path | None) -> bool:
    if value is None:
        return False
    text = str(value)
    return text.startswith(("/", "./", "../", "~"))
