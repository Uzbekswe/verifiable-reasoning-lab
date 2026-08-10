"""Hard-distillation data preparation and a bounded supervised update."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

DISTILL_DATASET_LICENSE = "Apache-2.0"


@dataclass(frozen=True)
class DistillationExample:
    """One licensed teacher trace and its source problem."""

    row_index: int
    problem: str
    answer: str
    thinking: str
    content: str

    @property
    def target_text(self) -> str:
        return f"{self.thinking.strip()}\n\n{self.content.strip()}".strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "problem": self.problem,
            "answer": self.answer,
            "thinking_characters": len(self.thinking),
            "content_characters": len(self.content),
        }


@dataclass(frozen=True)
class EncodedDistillationExample:
    """Tokenized causal-LM sequence with the prompt boundary preserved."""

    row_index: int
    input_ids: tuple[int, ...]
    prompt_length: int

    @property
    def token_count(self) -> int:
        return len(self.input_ids)


def load_distillation_examples(path: str | Path, limit: int | None = None) -> list[DistillationExample]:
    """Load and validate the local copy of an Apache-2.0 teacher dataset."""
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list):
        raise TypeError("distillation data must be a JSON list")
    examples = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"distillation row {row_index} must be an object")
        required = ("problem", "message_thinking", "message_content")
        if any(not str(row.get(field, "")).strip() for field in required):
            raise ValueError(f"distillation row {row_index} is missing a required field")
        examples.append(
            DistillationExample(
                row_index=row_index,
                problem=str(row["problem"]),
                answer=str(row.get("gtruth_answer", "")),
                thinking=str(row["message_thinking"]),
                content=str(row["message_content"]),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples


def render_distillation_prompt(problem: str) -> str:
    return (
        "You are a careful math solver. Show your reasoning and end with a concise final answer.\n\n"
        f"Problem:\n{problem.strip()}\n\nSolution:\n"
    )


def encode_distillation_example(tokenizer, example: DistillationExample) -> EncodedDistillationExample:
    prompt_ids = tokenizer.encode(render_distillation_prompt(example.problem))
    target_ids = tokenizer.encode(example.target_text)
    if not prompt_ids or not target_ids:
        raise ValueError(f"distillation row {example.row_index} encoded to an empty sequence")
    return EncodedDistillationExample(
        row_index=example.row_index,
        input_ids=tuple(prompt_ids + target_ids),
        prompt_length=len(prompt_ids),
    )


def select_fitting_examples(
    tokenizer,
    examples: list[DistillationExample],
    limit: int,
    max_length: int,
) -> list[EncodedDistillationExample]:
    """Select the first valid rows under the declared context cap."""
    if limit < 1 or max_length < 2:
        raise ValueError("limit must be positive and max_length must be at least 2")
    selected = []
    for example in examples:
        encoded = encode_distillation_example(tokenizer, example)
        if encoded.token_count <= max_length:
            selected.append(encoded)
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        raise ValueError(f"only found {len(selected)} rows fitting max_length={max_length}")
    return selected


def hard_distillation_loss(model: torch.nn.Module, input_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Cross-entropy on teacher tokens, masking the prompt prefix."""
    if input_ids.ndim != 1:
        raise ValueError("input_ids must have shape [sequence]")
    if not 0 < prompt_length < input_ids.numel():
        raise ValueError("prompt_length must leave at least one target token")
    logits = model(input_ids.unsqueeze(0)).squeeze(0).float()
    labels = input_ids.clone()
    labels[:prompt_length] = -100
    return F.cross_entropy(logits[:-1], labels[1:], ignore_index=-100)


def hard_distillation_step(
    model: torch.nn.Module,
    encoded: EncodedDistillationExample,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float = 1.0,
) -> dict[str, Any]:
    """Apply one supervised hard-distillation update with finite-gradient safety."""
    input_ids = torch.tensor(encoded.input_ids, device=device, dtype=torch.long)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss = hard_distillation_loss(model, input_ids, encoded.prompt_length)
    finite_loss = bool(torch.isfinite(loss).item())
    if not finite_loss:
        return {
            "row_index": encoded.row_index,
            "loss": float("nan"),
            "gradient_norm": None,
            "update_applied": False,
            "nonfinite_loss": True,
            "nonfinite_gradient": False,
        }
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    nonfinite_gradient = any(not bool(torch.isfinite(gradient).all().item()) for gradient in gradients)
    if nonfinite_gradient:
        optimizer.zero_grad(set_to_none=True)
        return {
            "row_index": encoded.row_index,
            "loss": float(loss.detach().item()),
            "gradient_norm": None,
            "update_applied": False,
            "nonfinite_loss": False,
            "nonfinite_gradient": True,
        }
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()
    return {
        "row_index": encoded.row_index,
        "loss": float(loss.detach().item()),
        "gradient_norm": float(gradient_norm.detach().item()),
        "update_applied": True,
        "nonfinite_loss": False,
        "nonfinite_gradient": False,
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
