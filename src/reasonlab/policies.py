"""Inference-time reasoning policies for Chapters 4 and 5."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

from .models.backend import GenerationResult, Qwen3Backend
from .verification import extract_final_candidate, refinement_feedback, verify_task


@dataclass(frozen=True)
class PolicyResult:
    """A selected answer plus all attempts used to select it."""

    method: str
    selected: GenerationResult
    candidates: tuple[GenerationResult, ...]
    selection: dict[str, Any]
    overhead: tuple[GenerationResult, ...] = ()

    @property
    def text(self) -> str:
        return self.selected.text

    @property
    def generated_token_count(self) -> int:
        return sum(candidate.generated_token_count for candidate in (*self.candidates, *self.overhead))

    @property
    def elapsed_seconds(self) -> float:
        return sum(candidate.elapsed_seconds for candidate in (*self.candidates, *self.overhead))

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
            "overhead_calls": len(self.overhead),
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


def _refine_prompt(task_prompt: str, draft: str, feedback: str, critique: str) -> str:
    return (
        "Revise a math or logic answer carefully. Do not invent facts or mention the hidden answer.\n\n"
        f"Question:\n{task_prompt}\n\n"
        f"Previous answer:\n{draft}\n\n"
        f"Verifier feedback:\n{feedback}\n\n"
        f"Reviewer notes:\n{critique}\n\n"
        "Re-solve the problem, show only brief work, and end with exactly \\boxed{ANSWER}.\n"
        "Revised answer:"
    )


def _critique_prompt(task_prompt: str, draft: str, feedback: str) -> str:
    return (
        "Review this math or logic response without supplying the correct answer. "
        "Identify a likely arithmetic, logic, or formatting issue and give a short repair plan.\n\n"
        f"Question:\n{task_prompt}\n\n"
        f"Draft:\n{draft}\n\n"
        f"Verifier feedback:\n{feedback}\n\n"
        "Review notes:"
    )


def self_refine(
    backend: Qwen3Backend,
    task: dict,
    prompt: str,
    max_refinements: int = 1,
    max_new_tokens: int = 64,
    critique_max_tokens: int = 48,
    temperature: float = 0.7,
    top_p: float = 0.9,
    seed: int | None = 0,
) -> PolicyResult:
    """Iteratively revise a response using verifier status without answer leakage."""
    if max_refinements < 0:
        raise ValueError("max_refinements must be non-negative")
    current = backend.generate(
        prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_cache=True,
    )
    candidates = [current]
    overhead = []
    history = []
    current_check = verify_task(task, current.text)
    history.append({"iteration": 0, **current_check.to_dict()})
    if current_check.status == "correct":
        return PolicyResult(
            method="self_refinement",
            selected=current,
            candidates=tuple(candidates),
            overhead=tuple(overhead),
            selection={"revision_count": 0, "verification_history": history, "stopped_early": True},
        )

    for iteration in range(max_refinements):
        feedback = refinement_feedback(current_check)
        critique = backend.generate(
            _critique_prompt(prompt, current.text, feedback),
            max_new_tokens=critique_max_tokens,
            temperature=min(temperature, 0.5),
            top_p=top_p,
            seed=None if seed is None else seed + iteration * 3 + 1,
            use_cache=True,
        )
        overhead.append(critique)
        revised = backend.generate(
            _refine_prompt(prompt, current.text, feedback, critique.text),
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + iteration * 3 + 2,
            use_cache=True,
        )
        candidates.append(revised)
        revised_check = verify_task(task, revised.text)
        history.append({"iteration": iteration + 1, **revised_check.to_dict()})
        current, current_check = revised, revised_check
        if current_check.status == "correct":
            break
    return PolicyResult(
        method="self_refinement",
        selected=current,
        candidates=tuple(candidates),
        overhead=tuple(overhead),
        selection={
            "revision_count": len(candidates) - 1,
            "verification_history": history,
            "stopped_early": current_check.status == "correct",
        },
    )
