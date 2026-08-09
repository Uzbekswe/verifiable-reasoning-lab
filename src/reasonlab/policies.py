"""Inference-time reasoning policies for Chapters 4 and 5."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models.backend import GenerationResult, Qwen3Backend
from .verification import extract_final_candidate


@dataclass(frozen=True)
class PolicyResult:
    """A selected answer plus all attempts used to select it."""

    method: str
    selected: GenerationResult
    candidates: tuple[GenerationResult, ...]
    selection: dict[str, Any]

    @property
    def text(self) -> str:
        return self.selected.text

    @property
    def generated_token_count(self) -> int:
        return sum(candidate.generated_token_count for candidate in self.candidates)

    @property
    def elapsed_seconds(self) -> float:
        return sum(candidate.elapsed_seconds for candidate in self.candidates)

    @property
    def tokens_per_second(self) -> float:
        elapsed = self.elapsed_seconds
        return self.generated_token_count / elapsed if elapsed else 0.0

    @property
    def device(self) -> str:
        return self.selected.device

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "selected": self.selected.to_dict(),
            "attempts": len(self.candidates),
            "selection": self.selection,
        }


def _sample_candidates(
    backend: Qwen3Backend,
    prompt: str,
    attempts: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
) -> tuple[GenerationResult, ...]:
    if attempts < 1:
        raise ValueError("attempts must be positive")
    return tuple(
        backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + index,
        )
        for index in range(attempts)
    )


def best_of_n(
    backend: Qwen3Backend,
    prompt: str,
    n: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int | None = 0,
) -> PolicyResult:
    """Generate N samples and select the highest mean raw model log-probability.

    This is deliberately model-scored. It never consults the task answer or
    verifier, which would turn best-of-N into an oracle-assisted comparison.
    """
    candidates = _sample_candidates(backend, prompt, n, max_new_tokens, temperature, top_p, seed)
    scores = [candidate.mean_logprob if candidate.mean_logprob is not None else float("-inf") for candidate in candidates]
    selected_index = max(range(len(candidates)), key=lambda index: (scores[index], -index))
    return PolicyResult(
        method="best_of_n_logprob",
        selected=candidates[selected_index],
        candidates=candidates,
        selection={"selected_index": selected_index, "mean_logprobs": scores, "score": "mean_raw_logprob"},
    )


def self_consistency(
    backend: Qwen3Backend,
    prompt: str,
    n: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int | None = 0,
) -> PolicyResult:
    """Generate N samples and select the unique majority extracted answer."""
    candidates = _sample_candidates(backend, prompt, n, max_new_tokens, temperature, top_p, seed)
    extracted = [extract_final_candidate(candidate.text).candidate for candidate in candidates]
    vote_keys = [candidate if candidate is not None else "<parse_error>" for candidate in extracted]
    counts = Counter(vote_keys)
    top_count = max(counts.values()) if counts else 0
    winners = [answer for answer, count in counts.items() if count == top_count]
    # Ties are resolved by model score only for a deterministic returned text;
    # the tie and vote table remain visible in the artifact.
    winner = max(winners, key=lambda answer: max(
        (candidate.mean_logprob if candidate.mean_logprob is not None else float("-inf"))
        for candidate, extracted_answer in zip(candidates, vote_keys)
        if extracted_answer == answer
    )) if winners else "<parse_error>"
    selected_index = next(index for index, answer in enumerate(vote_keys) if answer == winner)
    return PolicyResult(
        method="self_consistency",
        selected=candidates[selected_index],
        candidates=candidates,
        selection={
            "selected_index": selected_index,
            "vote_counts": dict(counts),
            "majority_winners": winners,
            "tie": len(winners) > 1,
            "score": "majority_extracted_answer",
        },
    )
