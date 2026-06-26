#!/usr/bin/env python3
"""Run ESMFold2-Fast for TheBioCollection protein deep metrics."""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
import types
from pathlib import Path
from typing import Any


DEFAULT_ESMFOLD2_MODEL = "biohub/ESMFold2-Fast"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one ESMFold2-Fast fold request.")
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_path = Path(args.output_json)
    try:
        payload = json.loads(Path(args.input_json).read_text(encoding="utf-8"))
        output = fold(payload, output_dir=Path(args.output_dir))
        result = {"ok": True, "output": output}
    except Exception as exc:
        result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    if not result["ok"]:
        raise SystemExit(result["error"])


def fold(payload: dict[str, Any], *, output_dir: Path) -> dict[str, Any]:
    start = time.perf_counter()
    deps = load_esmfold2_dependencies()
    model_name = normalize_esmfold2_model_name(payload.get("model_name"))
    device = str(payload.get("device") or "cuda")
    dtype = str(payload.get("dtype") or "auto")
    model_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "cache_dir": payload.get("cache_dir"),
        "local_files_only": bool(payload.get("local_files_only", True)),
    }
    torch_dtype = resolve_torch_dtype(deps["torch"], dtype)
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    try:
        try:
            model = load_auto_model_with_fallback(deps["AutoModel"], model_name, model_kwargs)
        except Exception:
            model = load_auto_model_with_fallback(
                deps["ESMFold2Model"],
                model_name,
                model_kwargs,
            )
    except Exception as exc:
        raise RuntimeError(
            "Could not load ESMFold2-Fast from local checkpoint files. "
            "Pass `model_name` as a local checkpoint directory or pre-populate the "
            "local Hugging Face cache; remote download/API access is intentionally disabled."
        ) from exc
    model = model.eval().to(resolve_device(deps["torch"], device))

    protein_inputs = []
    if payload.get("target_sequence") and payload.get("binder_sequence"):
        protein_inputs = [
            deps["ProteinInput"](id="A", sequence=normalize_sequence(payload["target_sequence"]), msa=None),
            deps["ProteinInput"](id="B", sequence=normalize_sequence(payload["binder_sequence"]), msa=None),
        ]
    elif payload.get("binder_sequence"):
        protein_inputs = [
            deps["ProteinInput"](id="A", sequence=normalize_sequence(payload["binder_sequence"]), msa=None)
        ]
    elif payload.get("sequence"):
        protein_inputs = [
            deps["ProteinInput"](id="A", sequence=normalize_sequence(payload["sequence"]), msa=None)
        ]
    else:
        raise ValueError("ESMFold2-Fast request requires sequence or binder/target sequences.")

    ligand_inputs = []
    if payload.get("ligand_ccd"):
        ligand_inputs.append(deps["LigandInput"](id="L", ccd=[str(payload["ligand_ccd"])]))
    elif payload.get("ligand_smiles"):
        ligand_inputs.append(deps["LigandInput"](id="L", smiles=str(payload["ligand_smiles"])))

    input_spec = deps["StructurePredictionInput"](sequences=[*protein_inputs, *ligand_inputs])
    builder = deps["ESMFold2InputBuilder"]()
    fold_kwargs = {
        "num_loops": int(payload.get("num_loops") or 3),
        "num_sampling_steps": int(payload.get("num_sampling_steps") or 50),
        "num_diffusion_samples": int(payload.get("num_diffusion_samples") or 1),
        "seed": int(payload.get("seed") or 0),
        "complex_id": "thebiocollection_eval",
        "msa_max_depth": 1,
        "msa_column_mask_rate": 0.0,
    }
    result = fold_with_optional_pae(builder, model, input_spec, fold_kwargs)
    if isinstance(result, list):
        result = result[0] if result else None
    if result is None:
        raise RuntimeError("ESMFold2-Fast returned no prediction.")

    output_dir.mkdir(parents=True, exist_ok=True)
    cif_path = output_dir / "prediction.cif"
    cif_path.write_text(result.complex.to_mmcif(), encoding="utf-8")

    return {
        "model_name": model_name,
        "fold_backend": "esmfold2_fast",
        "plddt": plddt_to_percent(getattr(result, "plddt", None)),
        "ptm": scalar_numeric(getattr(result, "ptm", None)),
        "iptm": scalar_numeric(getattr(result, "iptm", None)),
        "predicted_cif": str(cif_path),
        "elapsed_sec": time.perf_counter() - start,
    }


def load_esmfold2_dependencies() -> dict[str, Any]:
    try:
        import torch
        from esm.models.esmfold2 import (
            ESMFold2InputBuilder,
            LigandInput,
            ProteinInput,
            StructurePredictionInput,
        )
        from transformers import AutoModel
        try:
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
        except Exception:
            clear_transformers_esmfold2_modules()
            install_esmfold2_kernel_stub()
            from transformers.models.esmfold2.modeling_esmfold2 import ESMFold2Model
    except Exception as exc:
        raise RuntimeError(
            "Could not import ESMFold2-Fast dependencies. Install Biohub esm and a "
            "transformers build with transformers.models.esmfold2."
        ) from exc
    return {
        "torch": torch,
        "AutoModel": AutoModel,
        "ESMFold2Model": ESMFold2Model,
        "ESMFold2InputBuilder": ESMFold2InputBuilder,
        "ProteinInput": ProteinInput,
        "LigandInput": LigandInput,
        "StructurePredictionInput": StructurePredictionInput,
    }


def fold_with_optional_pae(builder: Any, model: Any, input_spec: Any, kwargs: dict[str, Any]) -> Any:
    call_kwargs = supported_call_kwargs(builder.fold, kwargs)
    if call_accepts_keyword(builder.fold, "include_pae"):
        call_kwargs["include_pae"] = True
        return builder.fold(model, input_spec, **call_kwargs)
    return builder.fold(IncludePaeModelWrapper(model), input_spec, **call_kwargs)


class IncludePaeModelWrapper:
    def __init__(self, model: Any):
        self._model = model

    def __getattr__(self, name: str) -> Any:
        return getattr(self._model, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        kwargs["include_pae"] = True
        return self._model(*args, **kwargs)


def call_accepts_keyword(callable_obj: Any, keyword: str) -> bool:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return False
    if keyword in signature.parameters:
        return True
    return any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )


def supported_call_kwargs(callable_obj: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return dict(kwargs)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    return {key: value for key, value in kwargs.items() if key in signature.parameters}


def load_auto_model_with_fallback(model_cls: Any, model_name: str, kwargs: dict[str, Any]) -> Any:
    clean_kwargs = {key: value for key, value in kwargs.items() if value is not None}
    try:
        return model_cls.from_pretrained(model_name, **clean_kwargs)
    except TypeError:
        clean_kwargs.pop("trust_remote_code", None)
        return model_cls.from_pretrained(model_name, **clean_kwargs)


def resolve_torch_dtype(torch: Any, dtype: str) -> Any | None:
    if dtype == "auto":
        return None
    if dtype == "float32":
        return torch.float32
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {dtype}")


def resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def plddt_to_percent(value: Any) -> float | None:
    score = mean_numeric(value)
    if score is None:
        return None
    if score <= 1.5:
        return float(score * 100.0)
    return float(score)


def mean_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().cpu()
    try:
        import numpy as np

        arr = np.asarray(value, dtype=float)
        if arr.size == 0:
            return None
        return float(arr.mean())
    except Exception:
        return scalar_numeric(value)


def scalar_numeric(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().float().mean().cpu()
    if hasattr(value, "item"):
        value = value.item()
    try:
        return float(value)
    except Exception:
        return None


def normalize_sequence(value: Any) -> str:
    return "".join(str(value or "").split()).upper()


def normalize_esmfold2_model_name(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return DEFAULT_ESMFOLD2_MODEL
    aliases = {
        "esmfold2": "biohub/ESMFold2-Fast",
        "esm-fold2": "biohub/ESMFold2-Fast",
        "biohub/esmfold2": "biohub/ESMFold2-Fast",
        "esmfold2-fast": "biohub/ESMFold2-Fast",
        "esm-fold2-fast": "biohub/ESMFold2-Fast",
        "biohub/esmfold2-fast": "biohub/ESMFold2-Fast",
    }
    return aliases.get(text.lower(), text)


def clear_transformers_esmfold2_modules() -> None:
    for name in list(sys.modules):
        if name.startswith("transformers.models.esmfold2.modeling_esmfold2"):
            del sys.modules[name]


def install_esmfold2_kernel_stub() -> None:
    kernels = types.ModuleType("transformers.models.esmfold2.kernels")
    sys.modules["transformers.models.esmfold2.kernels"] = kernels


if __name__ == "__main__":
    main()
