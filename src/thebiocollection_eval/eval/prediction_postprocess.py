"""Task-aware cleanup for TheBioCollection-Eval predictions."""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from thebiocollection_eval.eval.schema import EvalRecord


_ZERO_WIDTH_RE = re.compile(r"[\u200b\u200c\u200d\ufeff\u00ad]")
_ANSWER_PREFIX_RE = re.compile(
    r"^\s*(?:\[\s*)?(?:answer|prediction|predicted\s+answer|output)(?:\s*\])?\s*:?\s*",
    re.I,
)
_CONTINUATION_RE = re.compile(
    r"(?im)(?:\n\s*)+(?:"
    r"now\s+answer(?:\s+the\s+next\s+question)?\.?|"
    r"example\s+\d+\b|"
    r"question\s*:|"
    r"input\s*:"
    r")"
)
_INLINE_CONTAMINATION_RE = re.compile(
    r"(?i)(?:"
    r"now\s+answer\s+the\s+next\s+question|"
    r"example\s+\d+\b|"
    r"question\s*:|"
    r"the\s+given\s+molecule\b|"
    r"the\s+molecule\s+(?:has\s+the\s+SMILES\s+representation|is\s+SMILES\b)|"
    r"<smiles>|"
    r"<inchi>|"
    r"the\s+compound\s+contains|"
    r"its\s+molecular\s+formula\s+is"
    r")"
)
_LABELS = (
    "yes_acceptor",
    "yes_donor",
    "yes_strong",
    "yes_weak",
    "yes",
    "no",
)


def postprocess_prediction(record: EvalRecord, prediction: Any) -> str:
    """Return the answer-like part of a generated prediction for scoring."""
    text = normalize_generated_text(prediction)
    segment = first_answer_segment(text)
    metric = record.metric

    if metric == "label_exact":
        if record.target_format == "mcqa_letter":
            return extract_mcqa_letter_answer(record, segment)
        if record.target_format == "choice_token":
            return extract_choice_token_answer(record, segment)
        return extract_label_answer(segment)
    if metric == "perturbation_deg_direction_json":
        return extract_first_json_answer(segment)
    if metric == "span_localization_json_with_subsequence":
        return extract_first_json_answer(segment)
    if metric in {"canonical_smiles_exact", "mol_instructions_forward_synthesis"}:
        return extract_smiles_answer(segment)
    if metric == "mollang_recognition_exact":
        return first_nonempty_line(cut_inline_contamination(segment))
    if metric in {
        "protein_design",
        "binder_generation",
        "interpro_overlap",
    }:
        return cut_inline_contamination(segment)
    return segment


def normalize_generated_text(prediction: Any) -> str:
    text = str(prediction or "").replace("\r\n", "\n").replace("\r", "\n")
    text = _ZERO_WIDTH_RE.sub("", text)
    return _ANSWER_PREFIX_RE.sub("", text.strip())


def first_answer_segment(text: str) -> str:
    match = _CONTINUATION_RE.search(text)
    if match:
        text = text[: match.start()]
    return _ANSWER_PREFIX_RE.sub("", text.strip())


def first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        line = _ANSWER_PREFIX_RE.sub("", line.strip())
        if line:
            return line
    return _ANSWER_PREFIX_RE.sub("", text.strip())


def cut_inline_contamination(text: str) -> str:
    match = _INLINE_CONTAMINATION_RE.search(text)
    if match:
        text = text[: match.start()]
    return _ANSWER_PREFIX_RE.sub("", text.strip())


def extract_first_json_answer(text: str) -> str:
    candidate = first_parseable_balanced_json(text)
    return candidate if candidate is not None else cut_inline_contamination(text)


def first_parseable_balanced_json(text: str) -> str | None:
    text = text.strip()
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
                    if is_jsonish(candidate):
                        return candidate
                    break
    return None


def is_jsonish(text: str) -> bool:
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, (dict, list)):
            return True
    return False


def extract_smiles_answer(text: str) -> str:
    text = text.strip()
    tag_match = re.search(r"<smiles>\s*", text, flags=re.I)
    if not tag_match:
        return first_nonempty_line(cut_inline_contamination(text))

    rest = text[tag_match.end() :]
    close_match = re.search(r"</smiles>", rest, flags=re.I)
    contamination_match = _INLINE_CONTAMINATION_RE.search(rest)
    end_candidates = [
        match.start()
        for match in (close_match, contamination_match)
        if match is not None
    ]
    end = min(end_candidates) if end_candidates else len(rest)
    candidate = rest[:end].strip()
    return f"<smiles>{candidate}</smiles>" if candidate else text



def extract_label_answer(text: str) -> str:
    line = first_nonempty_line(cut_inline_contamination(text))
    line = re.sub(r"(?i)^\s*(?:the\s+)?answer\s+(?:is|should\s+be)\s+", "", line)
    normalized = line.strip().lower()
    normalized = re.sub(r"^[\s:;,.\"'`]+|[\s:;,.\"'`]+$", "", normalized)
    normalized = re.sub(r"[\s-]+", "_", normalized)
    normalized = re.sub(r"[^a-z0-9_].*$", "", normalized)
    for label in _LABELS:
        if normalized == label or normalized.startswith(label + "_"):
            return label
    return normalized


def extract_mcqa_letter_answer(record: EvalRecord, text: str) -> str:
    line = first_nonempty_line(cut_inline_contamination(text))
    line = re.sub(r"(?i)^\s*(?:the\s+)?answer\s+(?:is|should\s+be)\s+", "", line)
    choices = (
        record.metadata.get("mcqa_choices")
        or record.metadata.get("choices")
    )
    allowed = set("ABCDEFGH")
    if isinstance(choices, list):
        letters = {
            str(choice.get("letter", "")).strip().upper()
            for choice in choices
            if isinstance(choice, dict)
        }
        allowed = {letter for letter in letters if len(letter) == 1} or allowed
    match = re.search(r"\b([A-H])\b", line.strip().upper())
    if match and match.group(1) in allowed:
        return match.group(1)
    stripped = re.sub(r"^[\s:;,.\"'`]+|[\s:;,.\"'`]+$", "", line.strip().upper())
    return stripped[:1] if stripped[:1] in allowed else stripped


def extract_choice_token_answer(record: EvalRecord, text: str) -> str:
    line = first_nonempty_line(cut_inline_contamination(text))
    line = re.sub(r"(?i)^\s*(?:the\s+)?answer\s+(?:is|should\s+be)\s+", "", line)
    normalized = re.sub(r"^[\s:;,.\"'`]+|[\s:;,.\"'`]+$", "", line.strip().lower())
    candidates = record.metadata.get("choice_candidates")
    if isinstance(candidates, list):
        cleaned = [
            (str(candidate).strip(), str(candidate).strip().lower())
            for candidate in candidates
            if str(candidate).strip()
        ]
        for original, candidate in cleaned:
            if normalized == candidate or normalized.startswith(candidate + " "):
                return original
    return normalized.split()[0] if normalized.split() else normalized

