"""Small, fail-closed verifiers for the Chapter 3 task contract."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from tokenize import TokenError

from sympy import simplify
from sympy.core.sympify import SympifyError
from sympy.parsing import sympy_parser
from sympy.polys.polyerrors import PolynomialError

VERIFIER_VERSION = "chapter3-v1"
_SPECIAL_RE = re.compile(r"<\|[^>]+?\|>")
_FINAL_RE = re.compile(r"(?:final\s+answer|answer)\s*[:=]\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_LATEX_FIXES = (
    (r"\\left\s*|\\right\s*", ""),
    (r"\\,|\\!|\\;|\\:", ""),
    (r"\\cdot|·|×", "*"),
    (r"\\dfrac|\\tfrac", r"\\frac"),
)


@dataclass(frozen=True)
class ExtractionResult:
    candidate: str | None
    method: str | None
    error: str | None = None

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class VerificationResult:
    status: str
    candidate: str | None
    expected: str
    extraction_method: str | None
    reason: str
    verifier_version: str = VERIFIER_VERSION

    def to_dict(self):
        return asdict(self)


def _boxed_contents(text: str) -> tuple[str | None, str | None]:
    start = text.rfind(r"\boxed")
    if start < 0:
        return None, None
    cursor = start + len(r"\boxed")
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor >= len(text) or text[cursor] != "{":
        return None, "malformed_boxed_expression"
    cursor += 1
    content_start = cursor
    depth = 1
    while cursor < len(text) and depth:
        if text[cursor] == "{":
            depth += 1
        elif text[cursor] == "}":
            depth -= 1
        cursor += 1
    if depth:
        return None, "unbalanced_boxed_expression"
    return text[content_start : cursor - 1].strip().strip("$ "), None


def extract_final_candidate(text: str) -> ExtractionResult:
    """Extract only explicit final-answer markers; fail closed otherwise."""
    if not text or not text.strip():
        return ExtractionResult(None, None, "empty_output")
    cleaned = _SPECIAL_RE.sub("", text).strip()
    boxed, boxed_error = _boxed_contents(cleaned)
    if boxed is not None:
        return ExtractionResult(boxed, "boxed", None)
    if boxed_error:
        return ExtractionResult(None, "boxed", boxed_error)
    matches = _FINAL_RE.findall(cleaned)
    if matches:
        candidate = matches[-1].strip().strip("$ ")
        return ExtractionResult(candidate or None, "final_answer_label", None)
    return ExtractionResult(None, None, "no_explicit_final_answer")


def _normalize_expression(text: str) -> str:
    normalized = text.strip().lower().replace("$", "").replace("%", "")
    normalized = normalized.replace("^", "**")
    for pattern, replacement in _LATEX_FIXES:
        normalized = re.sub(pattern, replacement, normalized)
    normalized = re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        r"(\1)/(\2)",
        normalized,
    )
    normalized = re.sub(r"\\sqrt\s*\{([^{}]+)\}", r"sqrt(\1)", normalized)
    normalized = normalized.replace("{", "").replace("}", "")
    return normalized.strip()


def _parse_expression(text: str):
    if not text or len(text) > 256:
        return None
    try:
        return sympy_parser.parse_expr(
            _normalize_expression(text),
            transformations=(
                *sympy_parser.standard_transformations,
                sympy_parser.implicit_multiplication_application,
            ),
            evaluate=True,
        )
    except (
        SympifyError,
        SyntaxError,
        TypeError,
        AttributeError,
        IndexError,
        TokenError,
        ValueError,
        PolynomialError,
    ):
        return None


def _numeric_equivalent(candidate: str, expected: str) -> tuple[bool, str]:
    predicted = _parse_expression(candidate)
    truth = _parse_expression(expected)
    if predicted is None:
        return False, "candidate_expression_parse_failed"
    if truth is None:
        return False, "task_answer_parse_failed"
    try:
        return bool(simplify(predicted - truth) == 0), "symbolic_equivalence"
    except (SympifyError, TypeError, ValueError):
        return False, "symbolic_comparison_failed"


def _logic_equivalent(candidate: str, expected: str) -> tuple[bool, str]:
    normalize = lambda value: re.sub(r"\s+", " ", value.strip().lower().rstrip(".!?")).strip()
    return normalize(candidate) == normalize(expected), "normalized_exact_match"


def verify_task(task: dict, output_text: str) -> VerificationResult:
    extracted = extract_final_candidate(output_text)
    expected = str(task["answer"])
    if extracted.candidate is None:
        return VerificationResult(
            "parse_error", None, expected, extracted.method, extracted.error or "candidate_missing"
        )
    answer_type = task.get("answer_type")
    if answer_type == "numeric":
        correct, reason = _numeric_equivalent(extracted.candidate, expected)
    elif answer_type == "logic":
        correct, reason = _logic_equivalent(extracted.candidate, expected)
    else:
        return VerificationResult(
            "verifier_error",
            extracted.candidate,
            expected,
            extracted.method,
            f"unsupported_answer_type:{answer_type}",
        )
    if reason == "task_answer_parse_failed":
        status = "verifier_error"
    else:
        status = "correct" if correct else "incorrect"
    return VerificationResult(status, extracted.candidate, expected, extracted.method, reason)
