"""Metrics and normalization helpers for TheBioCollection-Eval."""

from __future__ import annotations

import ast
import json
import math
import re
from collections import Counter
from statistics import mean
from typing import Any

from rdkit import Chem, DataStructs
from rdkit.Chem import MACCSkeys, rdMolDescriptors
from rdkit import RDLogger

from thebiocollection_eval.eval.schema import EvalRecord

RDLogger.DisableLog("rdApp.*")

VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")
DEGENERACY_K_VALUES = (2, 3, 4)
DEGENERACY_MIN_LENGTH = 30
DEGENERACY_MAX_AA_FRACTION = 0.45
DEGENERACY_MIN_NORMALIZED_ENTROPY = 0.35
DEGENERACY_MAX_KMER_FRACTION = 0.25
DEGENERACY_MIN_UNIQUE_KMER_FRACTION = 0.05
DEGENERACY_MAX_HOMOPOLYMER_FRACTION = 0.30
INTERPRO_ID_RE = re.compile(r"(?i)\bIPR\d{6}\b")
INTERPRO_TYPE_SUFFIX_RE = re.compile(
    r"(?i)\s*\((?:domain|family|homologous_superfamily|repeat|conserved_site|"
    r"active_site|binding_site|site|unintegrated|unknown)\)\s*$"
)


def levenshtein(left: str, right: str) -> int:
    """Compute Levenshtein edit distance for short generated strings."""
    if left == right:
        return 0
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for i, left_char in enumerate(left, start=1):
        current = [i]
        for j, right_char in enumerate(right, start=1):
            insert = current[j - 1] + 1
            delete = previous[j] + 1
            replace = previous[j - 1] + int(left_char != right_char)
            current.append(min(insert, delete, replace))
        previous = current
    return previous[-1]


def extract_protein_sequence(text: str) -> str:
    """Extract a protein sequence from tagged or plain model output."""
    value = str(text or "").strip()
    tag_match = re.search(r"<protein>\s*([A-Za-z]+)\s*</protein>", value, flags=re.I | re.S)
    if tag_match:
        return re.sub(r"\s+", "", tag_match.group(1)).upper()
    fenced = re.search(r"```(?:\w+)?\s*([A-Za-z\s]+?)\s*```", value, flags=re.S)
    if fenced:
        candidate = re.sub(r"\s+", "", fenced.group(1)).upper()
        if candidate:
            return candidate
    candidates = re.findall(r"\b[A-Za-z]{5,}\b", value)
    candidates = [candidate.upper() for candidate in candidates]
    protein_like = [candidate for candidate in candidates if set(candidate) <= VALID_AA]
    if protein_like:
        return max(protein_like, key=len)
    return ""


def normalize_text(text: Any) -> str:
    value = str(text or "").strip()
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        value = value[1:-1].strip()
    value = re.sub(r"\s+", " ", value.lower())
    value = re.sub(r"^[\s:;,.]+|[\s:;,.]+$", "", value)
    return value


def parse_jsonish(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    for loader in (json.loads, ast.literal_eval):
        try:
            return loader(text)
        except Exception:
            pass
    return value


def extract_json_object(value: Any) -> Any | None:
    parsed = parse_jsonish(value)
    if isinstance(parsed, (dict, list)):
        return parsed
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    first_json = extract_first_json_text(text)
    if first_json is not None:
        parsed = parse_jsonish(first_json)
        if isinstance(parsed, (dict, list)):
            return parsed
    for start_char, end_char in (("{", "}"), ("[", "]")):
        start = text.find(start_char)
        end = text.rfind(end_char)
        if start == -1 or end == -1 or end <= start:
            continue
        candidate = text[start : end + 1]
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(candidate)
            except Exception:
                continue
            if isinstance(parsed, (dict, list)):
                return parsed
    return None


def extract_first_json_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S).strip()
    for start_index, start_char in enumerate(text):
        if start_char not in "{[":
            continue
        stack: list[str] = []
        in_string = False
        escaped = False
        for index in range(start_index, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in "{[":
                stack.append(char)
            elif char in "}]":
                if not stack:
                    break
                opener = stack.pop()
                if (opener == "{" and char != "}") or (opener == "[" and char != "]"):
                    break
                if not stack:
                    candidate = text[start_index : index + 1]
                    parsed = parse_jsonish(candidate)
                    if isinstance(parsed, (dict, list)):
                        return candidate
                    break
    return None


_COORDINATE_KEYS = {
    "start",
    "end",
    "position",
    "summit_position",
    "tss_position",
    "promoter_start",
    "promoter_end",
    "motif_start",
    "motif_end",
    "cds_start",
    "cds_end",
    "five_prime_utr_start",
    "five_prime_utr_end",
    "three_prime_utr_start",
    "three_prime_utr_end",
}
_IGNORED_SEQUENCE_KEYS = {
    "subsequence",
    "left_context",
    "right_context",
}
_IGNORED_CATEGORICAL_KEYS = _COORDINATE_KEYS | _IGNORED_SEQUENCE_KEYS


def span_localization_json_with_subsequence_metrics(
    pairs: list[tuple[EvalRecord, str]],
) -> dict[str, Any]:
    exact = 0
    coordinate_exact = 0
    categorical_exact = 0
    sequence_exact = 0
    for record, prediction in pairs:
        gold = extract_json_object(record.target)
        pred = extract_json_object(prediction)
        if gold is None or pred is None:
            continue
        exact += int(_canonical_json(gold) == _canonical_json(pred))
        coordinate_exact += int(
            _field_values(gold, _COORDINATE_KEYS) == _field_values(pred, _COORDINATE_KEYS)
        )
        categorical_exact += int(_categorical_values(gold) == _categorical_values(pred))
        sequence_exact += int(
            _field_values(gold, _IGNORED_SEQUENCE_KEYS)
            == _field_values(pred, _IGNORED_SEQUENCE_KEYS)
        )
    return {
        "exact_match": _safe_div(exact, len(pairs)),
        "coordinate_exact_match": _safe_div(coordinate_exact, len(pairs)),
        "categorical_field_exact_match": _safe_div(categorical_exact, len(pairs)),
        "sequence_field_exact_match": _safe_div(sequence_exact, len(pairs)),
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _drop_keys(value: Any, keys: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: _drop_keys(child, keys)
            for key, child in value.items()
            if key not in keys
        }
    if isinstance(value, list):
        return [_drop_keys(item, keys) for item in value]
    return value


def _field_values(value: Any, keys: set[str]) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    _collect_field_values(value, keys, values, prefix="$")
    return values


def _collect_field_values(
    value: Any, keys: set[str], out: list[tuple[str, Any]], *, prefix: str
) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}"
            if key in keys:
                out.append((child_prefix, value[key]))
            _collect_field_values(value[key], keys, out, prefix=child_prefix)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_field_values(item, keys, out, prefix=f"{prefix}[{index}]")


def _categorical_values(value: Any) -> list[tuple[str, Any]]:
    values: list[tuple[str, Any]] = []
    _collect_categorical_values(value, values, prefix="$")
    return values


def _collect_categorical_values(
    value: Any, out: list[tuple[str, Any]], *, prefix: str
) -> None:
    if isinstance(value, dict):
        for key in sorted(value):
            child_prefix = f"{prefix}.{key}"
            child = value[key]
            if key not in _IGNORED_CATEGORICAL_KEYS and not isinstance(child, (dict, list)):
                out.append((child_prefix, normalize_text(child)))
            _collect_categorical_values(child, out, prefix=child_prefix)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _collect_categorical_values(item, out, prefix=f"{prefix}[{index}]")


def label_exact_metrics(pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
    exact = 0
    for record, prediction in pairs:
        exact += int(str(record.target) == str(prediction))
    return {"exact_match": _safe_div(exact, len(pairs))}


def interpro_overlap_metrics(pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
    id_precisions = []
    id_recalls = []
    for record, prediction in pairs:
        gold_ids = gold_interpro_id_set(record)
        pred_ids = parse_interpro_id_set(prediction)
        if not gold_ids:
            continue
        id_overlap = len(gold_ids & pred_ids)
        id_precisions.append(_safe_div(id_overlap, len(pred_ids)))
        id_recalls.append(_safe_div(id_overlap, len(gold_ids)))
    return {
        "interpro_id_precision": mean(id_precisions) if id_precisions else None,
        "interpro_id_recall": mean(id_recalls) if id_recalls else None,
    }


def gold_interpro_id_set(record: EvalRecord) -> set[str]:
    ids, _ = gold_interpro_id_term_sets(record)
    return ids


def parse_interpro_id_set(value: Any) -> set[str]:
    ids, _ = parse_interpro_id_term_sets(value)
    return ids


def gold_interpro_id_term_sets(record: EvalRecord) -> tuple[set[str], set[str]]:
    entries = record.metadata.get("interpro_entries")
    ids: set[str] = set()
    terms: set[str] = set()
    if isinstance(entries, list):
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            interpro_id = normalize_interpro_id(entry.get("interpro_id"))
            term = normalize_interpro_term(entry.get("term"))
            if interpro_id:
                ids.add(interpro_id)
            if term:
                terms.add(term)
    if ids or terms:
        return ids, terms
    return parse_interpro_id_term_sets(record.target)


def parse_interpro_id_term_sets(value: Any) -> tuple[set[str], set[str]]:
    text = trim_interpro_prediction_noise(str(value or ""))
    ids = {match.group(0).upper() for match in INTERPRO_ID_RE.finditer(text)}
    terms: set[str] = set()
    for entry in split_interpro_entries(text):
        interpro_match = INTERPRO_ID_RE.search(entry)
        if not interpro_match:
            term = normalize_interpro_term(entry)
        else:
            term_text = entry[interpro_match.end() :]
            term_text = re.sub(r"^\s*[:\-]\s*", "", term_text)
            term = normalize_interpro_term(term_text)
        if term:
            terms.add(term)
    return ids, terms


def split_interpro_entries(text: str) -> list[str]:
    text = re.sub(r"^```(?:text)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    text = re.sub(
        r"(?im)^\s*(?:answer|prediction|predicted\s+interpro(?:\s+annotations?)?|"
        r"interpro(?:\s+domains?)?)\s*:\s*",
        "",
        text,
    )
    entries: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", line).strip()
        if not line:
            continue
        entries.extend(part.strip() for part in re.split(r";|\n", line) if part.strip())
    return entries


def trim_interpro_prediction_noise(text: str) -> str:
    cut = len(text)
    for pattern in (
        r"(?i)\bthe\s+given\s+molecule\b",
        r"(?i)\bthe\s+molecule\s+has\s+the\s+SMILES\s+representation\b",
        r"(?i)<smiles>",
        r"(?im)^\s*Question\s*:",
        r"(?im)^\s*Example\s+\d+\b",
        r"(?im)^\s*Now\s+answer\b",
        r"(?im)^\s*GO\s+terms?\s*:",
    ):
        match = re.search(pattern, text)
        if match:
            cut = min(cut, match.start())
    return text[:cut]


def normalize_interpro_id(value: Any) -> str:
    match = INTERPRO_ID_RE.search(str(value or ""))
    return match.group(0).upper() if match else ""


def normalize_interpro_term(value: Any) -> str:
    text = str(value or "").strip()
    text = INTERPRO_ID_RE.sub("", text)
    text = re.sub(r"\s*\[[^\]]+\]\s*$", "", text)
    text = INTERPRO_TYPE_SUFFIX_RE.sub("", text)
    text = re.sub(r"^[\s:;,.\"'`-]+|[\s:;,.\"'`-]+$", "", text)
    text = re.sub(r"\s+", " ", text.lower())
    if not text or text in {"none", "unknown", "n/a"}:
        return ""
    if len(text.split()) > 18:
        return ""
    return text


def canonicalize_smiles(smiles: Any, *, isomeric: bool = True) -> str | None:
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=isomeric)


_SMILES_ANSWER_CONTAMINATION_RE = re.compile(
    r"(?i)(?:"
    r"the\s+molecule\s+has\s+the\s+SMILES\s+representation|"
    r"<inchi>|"
    r"now\s+answer(?:\s+the\s+next\s+question)?|"
    r"example\s+\d+\b|"
    r"question\s*:|"
    r"task\s*:|"
    r"answer\s*:"
    r")"
)


def extract_first_smiles(text: str) -> str | None:
    stripped = text.strip()
    if canonicalize_smiles(stripped):
        return stripped
    tag_match = re.search(r"<smiles>\s*", stripped, flags=re.I)
    if tag_match:
        rest = stripped[tag_match.end() :]
        close_match = re.search(r"</smiles>", rest, flags=re.I)
        contamination_match = _SMILES_ANSWER_CONTAMINATION_RE.search(rest)
        end_candidates = [
            match.start()
            for match in (close_match, contamination_match)
            if match is not None
        ]
        end = min(end_candidates) if end_candidates else len(rest)
        candidate = rest[:end].strip().strip("`.,;:")
        return candidate or None
    code_blocks = re.findall(r"```(?:\w+)?\s*([^`\n][\s\S]*?)\s*```", stripped)
    if code_blocks:
        return code_blocks[0].strip().strip("`.,;:") or None
    contamination_match = _SMILES_ANSWER_CONTAMINATION_RE.search(stripped)
    if contamination_match:
        stripped = stripped[: contamination_match.start()].strip()
    candidates = re.findall(r"([A-Za-z0-9@+\-\[\]\(\)=#$\\/%.:]+)", stripped)
    for candidate in candidates:
        candidate = candidate.strip().strip("`.,;:")
        if len(candidate) < 1:
            continue
        if canonicalize_smiles(candidate):
            return candidate
    return None


def smiles_generation_metrics(pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
    valid = 0
    levenshteins = []
    maccs = []
    rdk = []
    morgan = []
    for record, prediction in pairs:
        target = canonicalize_smiles(record.target)
        pred_raw = extract_first_smiles(prediction)
        pred = canonicalize_smiles(pred_raw)
        if pred:
            valid += 1
        if target and pred:
            levenshteins.append(levenshtein(pred, target))
            target_mol = Chem.MolFromSmiles(target)
            pred_mol = Chem.MolFromSmiles(pred)
            if target_mol is not None and pred_mol is not None:
                maccs.append(
                    DataStructs.FingerprintSimilarity(
                        MACCSkeys.GenMACCSKeys(target_mol),
                        MACCSkeys.GenMACCSKeys(pred_mol),
                    )
                )
                rdk.append(
                    DataStructs.FingerprintSimilarity(
                        Chem.RDKFingerprint(target_mol), Chem.RDKFingerprint(pred_mol)
                    )
                )
                morgan.append(
                    DataStructs.TanimotoSimilarity(
                        rdMolDescriptors.GetMorganFingerprint(target_mol, 2),
                        rdMolDescriptors.GetMorganFingerprint(pred_mol, 2),
                    )
                )
    return {
        "valid_predictions": valid,
        "validity": _safe_div(valid, len(pairs)),
        "smiles_levenshtein": mean(levenshteins) if levenshteins else None,
        "maccs_tanimoto": mean(maccs) if maccs else None,
        "rdk_tanimoto": mean(rdk) if rdk else None,
        "morgan_tanimoto": mean(morgan) if morgan else None,
    }


def mol_instructions_forward_synthesis_metrics(pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
    valid = 0
    exact = 0
    levenshteins = []
    maccs = []
    morgan = []
    for record, prediction in pairs:
        target = canonicalize_smiles(record.target, isomeric=False)
        pred_raw = extract_first_smiles(prediction)
        pred = canonicalize_smiles(pred_raw, isomeric=False)
        if pred:
            valid += 1
        if target and pred:
            levenshteins.append(levenshtein(pred, target))
            target_mol = Chem.MolFromSmiles(target)
            pred_mol = Chem.MolFromSmiles(pred)
            if target_mol is not None and pred_mol is not None:
                try:
                    exact += int(Chem.MolToInchi(target_mol) == Chem.MolToInchi(pred_mol))
                except Exception:
                    exact += int(target == pred)
                maccs.append(
                    DataStructs.FingerprintSimilarity(
                        MACCSkeys.GenMACCSKeys(target_mol),
                        MACCSkeys.GenMACCSKeys(pred_mol),
                    )
                )
                morgan.append(
                    DataStructs.TanimotoSimilarity(
                        rdMolDescriptors.GetMorganFingerprint(target_mol, 2),
                        rdMolDescriptors.GetMorganFingerprint(pred_mol, 2),
                    )
                )
    return {
        "valid_predictions": valid,
        "validity": _safe_div(valid, len(pairs)),
        "exact_match": _safe_div(exact, len(pairs)),
        "smiles_levenshtein": mean(levenshteins) if levenshteins else None,
        "maccs_tanimoto": mean(maccs) if maccs else None,
        "morgan_tanimoto": mean(morgan) if morgan else None,
    }


def mollang_recognition_target(result_1: Any, result_2: Any) -> str:
    return f"result_1: {result_1}; result_2: {result_2}"


def parse_mollang_recognition_answer(text: str) -> dict[str, Any] | None:
    pattern = re.compile(
        r"result[_\s-]*1\s*:\s*(?P<r1>.*?)(?:;|\n|,)\s*"
        r"result[_\s-]*2\s*:\s*(?P<r2>[^\n]*?)(?:\n|$)",
        re.I | re.S,
    )
    match = pattern.search(text.strip())
    if match:
        return {
            "result_1": strip_mollang_contamination(match.group("r1")),
            "result_2": strip_mollang_contamination(match.group("r2")),
        }
    return None


_MOLLANG_CONTAMINATION_RE = re.compile(
    r"(?i)(?:"
    r"the\s+molecule\s+has\s+the\s+SMILES\s+representation|"
    r"<smiles>|"
    r"<inchi>|"
    r"now\s+answer\s+the\s+next\s+question|"
    r"example\s+\d+\b|"
    r"question\s*:|"
    r"task\s*:"
    r")"
)


def strip_mollang_contamination(value: Any) -> str:
    text = str(value or "").strip()
    match = _MOLLANG_CONTAMINATION_RE.search(text)
    if match:
        text = text[: match.start()]
    return text.strip()


def mollang_recognition_metrics(pairs: list[tuple[EvalRecord, str]]) -> dict[str, Any]:
    parsed = 0
    r1_exact = 0
    r2_exact = 0
    exact = 0
    for record, prediction in pairs:
        target = record.metadata.get("target_object")
        if not isinstance(target, dict):
            target = parse_mollang_recognition_answer(str(record.target))
        pred = parse_mollang_recognition_answer(prediction)
        if not target or not pred:
            continue
        parsed += 1
        r1_ok = normalize_text(target.get("result_1")) == normalize_text(pred.get("result_1"))
        r2_ok = _normalize_structure_value(target.get("result_2")) == _normalize_structure_value(
            pred.get("result_2")
        )
        r1_exact += int(r1_ok)
        r2_exact += int(r2_ok)
        exact += int(r1_ok and r2_ok)
    return {
        "result_1_exact_match": _safe_div(r1_exact, len(pairs)),
        "result_2_exact_match": _safe_div(r2_exact, len(pairs)),
        "exact_match": _safe_div(exact, len(pairs)),
    }


def protein_design_metrics(
    pairs: list[tuple[EvalRecord, str]],
    deep_metric_scorer: Any | None = None,
) -> dict[str, Any]:
    if deep_metric_scorer is None:
        raise RuntimeError("protein_design requires deep protein metrics.")
    valid = 0
    degeneracy = ProteinDegeneracyAccumulator()
    for record, prediction in pairs:
        pred_seq = extract_protein_sequence(prediction)
        if pred_seq and set(pred_seq) <= VALID_AA:
            valid += 1
            degeneracy.add(pred_seq)
    return {
        "validity": _safe_div(valid, len(pairs)),
        **degeneracy.metrics(total=len(pairs)),
        **deep_metric_scorer.score_protein_design(pairs),
    }


def binder_generation_metrics(
    pairs: list[tuple[EvalRecord, str]],
    deep_metric_scorer: Any | None = None,
) -> dict[str, Any]:
    if deep_metric_scorer is None:
        raise RuntimeError("binder_generation requires deep protein metrics.")
    valid = 0
    lengths = []
    degeneracy = ProteinDegeneracyAccumulator()
    for record, prediction in pairs:
        seq = extract_protein_sequence(prediction)
        if seq and set(seq) <= VALID_AA:
            valid += 1
            lengths.append(len(seq))
            degeneracy.add(seq)
    return {
        "valid_predictions": valid,
        "validity": _safe_div(valid, len(pairs)),
        "mean_generated_length": mean(lengths) if lengths else None,
        **degeneracy.metrics(total=len(pairs)),
        **deep_metric_scorer.score_binder_generation(pairs),
    }


def _normalize_structure_value(value: Any) -> Any:
    parsed = parse_jsonish(value)
    if isinstance(parsed, (list, tuple, dict)):
        return parsed
    return normalize_text(parsed)


class ProteinDegeneracyAccumulator:
    def __init__(self) -> None:
        self.degenerate = 0

    def add(self, sequence: str) -> None:
        stats = protein_degeneracy_stats(sequence)
        if not stats["valid_sequence"]:
            return
        self.degenerate += int(bool(stats["is_degenerate"]))

    def metrics(self, *, total: int) -> dict[str, Any]:
        degeneracy_rate = _safe_div(self.degenerate, total)
        return {
            "nondegeneracy": None
            if degeneracy_rate is None
            else 1.0 - degeneracy_rate
        }


def protein_degeneracy_stats(sequence: Any) -> dict[str, Any]:
    seq = re.sub(r"\s+", "", str(sequence or "")).upper()
    valid = bool(seq) and set(seq) <= VALID_AA
    if not valid:
        return {
            "valid_sequence": False,
            "is_degenerate": False,
            "degeneracy_reasons": [],
            "normalized_aa_entropy": None,
            "max_aa_fraction": None,
            "max_kmer_fraction": None,
            "min_unique_kmer_fraction": None,
            "max_homopolymer_fraction": None,
        }

    length = len(seq)
    aa_counts = Counter(seq)
    max_aa_fraction = max(aa_counts.values()) / length
    entropy = -sum((count / length) * math.log2(count / length) for count in aa_counts.values())
    normalized_entropy = entropy / math.log2(len(VALID_AA))
    max_homopolymer_fraction = longest_homopolymer_run(seq) / length

    max_kmer_fraction = 0.0
    unique_kmer_fractions = []
    for k in DEGENERACY_K_VALUES:
        if length < k:
            continue
        windows = length - k + 1
        counts = Counter(seq[i : i + k] for i in range(windows))
        max_kmer_fraction = max(max_kmer_fraction, max(counts.values()) / windows)
        unique_kmer_fractions.append(len(counts) / windows)
    min_unique_kmer_fraction = min(unique_kmer_fractions) if unique_kmer_fractions else 1.0

    reasons = []
    if length >= DEGENERACY_MIN_LENGTH and max_aa_fraction >= DEGENERACY_MAX_AA_FRACTION:
        reasons.append("single_amino_acid_dominance")
    if length >= DEGENERACY_MIN_LENGTH and normalized_entropy <= DEGENERACY_MIN_NORMALIZED_ENTROPY:
        reasons.append("low_amino_acid_entropy")
    if (
        length >= DEGENERACY_MIN_LENGTH
        and max_homopolymer_fraction >= DEGENERACY_MAX_HOMOPOLYMER_FRACTION
    ):
        reasons.append("long_homopolymer_run")
    if (
        length >= DEGENERACY_MIN_LENGTH
        and max_kmer_fraction >= DEGENERACY_MAX_KMER_FRACTION
        and min_unique_kmer_fraction <= DEGENERACY_MIN_UNIQUE_KMER_FRACTION
    ):
        reasons.append("repeated_kmer_pattern")

    return {
        "valid_sequence": True,
        "is_degenerate": bool(reasons),
        "degeneracy_reasons": reasons,
        "normalized_aa_entropy": normalized_entropy,
        "max_aa_fraction": max_aa_fraction,
        "max_kmer_fraction": max_kmer_fraction,
        "min_unique_kmer_fraction": min_unique_kmer_fraction,
        "max_homopolymer_fraction": max_homopolymer_fraction,
    }


def longest_homopolymer_run(sequence: str) -> int:
    if not sequence:
        return 0
    longest = 1
    current = 1
    previous = sequence[0]
    for char in sequence[1:]:
        if char == previous:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
            previous = char
    return max(longest, current)


def _safe_div(num: float, denom: float) -> float:
    return float(num / denom) if denom else 0.0

