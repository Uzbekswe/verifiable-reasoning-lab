"""Verifier interface and exact answer adapters."""

from .core import (
    ExtractionResult,
    VerificationResult,
    extract_final_candidate,
    refinement_feedback,
    verify_task,
)

__all__ = ["ExtractionResult", "VerificationResult", "extract_final_candidate", "refinement_feedback", "verify_task"]
