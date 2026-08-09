"""Deterministic tiny policy used to validate the GRPO training plumbing locally."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .models.backend import GenerationResult
from .rlvr import train_grpo


class TinyTokenizer:
    def encode(self, text: str) -> list[int]:
        return [0]

    def decode(self, token_ids) -> str:
        token_id = int(token_ids[0]) if token_ids else 0
        return r"\boxed{4}" if token_id == 1 else r"\boxed{5}"


class TinyPolicy(nn.Module):
    """A two-answer causal policy with learnable logits."""

    def __init__(self):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, sequence = input_ids.shape
        return self.logits.view(1, 1, -1).expand(batch, sequence, -1)


@dataclass
class PatternBackend:
    model: TinyPolicy
    tokenizer: TinyTokenizer
    calls: int = 0

    def generate(self, prompt: str, **kwargs) -> GenerationResult:
        token_id = 1 if self.calls % 2 == 0 else 2
        self.calls += 1
        logprob = float(torch.log_softmax(self.model.logits.detach().float(), dim=-1)[token_id])
        return GenerationResult(
            text=self.tokenizer.decode([token_id]),
            token_ids=[token_id],
            prompt_token_count=1,
            generated_token_count=1,
            elapsed_seconds=0.001,
            tokens_per_second=1000.0,
            device="cpu",
            use_cache=False,
            stopped_on_eos=False,
            mean_logprob=logprob,
            temperature=kwargs.get("temperature", 0.8),
            top_p=kwargs.get("top_p", 0.9),
            seed=kwargs.get("seed"),
        )


def run_tiny_grpo_smoke(steps: int = 6, num_rollouts: int = 4) -> dict:
    """Train a two-token policy and report the correct-token probability shift."""
    model = TinyPolicy()
    backend = PatternBackend(model=model, tokenizer=TinyTokenizer())
    optimizer = torch.optim.SGD(model.parameters(), lr=0.4)
    initial_probability = float(torch.softmax(model.logits.detach(), dim=-1)[1])
    tasks = [{"prompt": "What is 2 + 2?", "answer": "4", "answer_type": "numeric"}]
    metrics = train_grpo(
        backend,
        tasks,
        optimizer,
        steps=steps,
        num_rollouts=num_rollouts,
        max_new_tokens=1,
        temperature=0.8,
        top_p=0.9,
        seed=0,
    )
    final_probability = float(torch.softmax(model.logits.detach(), dim=-1)[1])
    return {
        "model": "tiny-two-answer-policy",
        "steps": steps,
        "num_rollouts": num_rollouts,
        "initial_correct_probability": initial_probability,
        "final_correct_probability": final_probability,
        "probability_delta": final_probability - initial_probability,
        "metrics": metrics,
        "finite": math.isfinite(final_probability),
    }
