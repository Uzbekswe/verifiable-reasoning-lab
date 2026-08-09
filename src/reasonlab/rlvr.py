"""Verifier-backed rewards and a small, inspectable GRPO implementation.

Chapter 6 deliberately stops at the unclipped policy-gradient objective. KL
regularization and clipped objectives belong to the Chapter 7 comparison.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from .datasets import render_prompt
from .models.backend import GenerationResult, Qwen3Backend
from .verification import extract_final_candidate, verify_task


@dataclass(frozen=True)
class RewardResult:
    """A scalar RLVR reward plus the verifier evidence that produced it."""

    reward: float
    status: str
    reason: str
    candidate: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def configure_trainable_scope(model: torch.nn.Module, scope: str = "all") -> dict[str, int | str]:
    """Select which model parameters may receive GRPO updates.

    ``output_head`` is the local numerical diagnostic: the Transformer
    backbone stays frozen while the vocabulary projection remains trainable.
    This is intentionally not presented as equivalent to full-model GRPO.
    """
    if scope not in {"all", "output_head"}:
        raise ValueError("scope must be 'all' or 'output_head'")
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")
    if scope == "output_head":
        output_head = getattr(model, "out_head", None)
        if output_head is None:
            raise ValueError("output_head scope requires a model.out_head module")
        for parameter in output_head.parameters():
            parameter.requires_grad_(True)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {"scope": scope, "total_parameters": total, "trainable_parameters": trainable}


def verifier_reward(task: dict, output_text: str) -> RewardResult:
    """Map the shared verifier contract to a binary correctness reward.

    Formatting failures and incorrect answers both receive zero. Keeping this
    mapping binary prevents the training objective from learning a hand-made
    notion of partial mathematical credit.
    """
    checked = verify_task(task, output_text)
    return RewardResult(
        reward=1.0 if checked.status == "correct" else 0.0,
        status=checked.status,
        reason=checked.reason,
        candidate=checked.candidate,
    )


@dataclass(frozen=True)
class Rollout:
    """One sampled completion and its detached verifier reward."""

    generation: GenerationResult
    reward: RewardResult
    seed: int | None

    @property
    def text(self) -> str:
        return self.generation.text

    @property
    def token_ids(self) -> tuple[int, ...]:
        return tuple(self.generation.token_ids)

    @property
    def old_logprob(self) -> float:
        """Sequence log-probability recorded under the sampling policy."""
        if self.generation.mean_logprob is None:
            raise ValueError("rollout is missing mean_logprob required by Chapter 7 ratios")
        return self.generation.mean_logprob * self.generation.generated_token_count

    def to_dict(self) -> dict[str, Any]:
        return {
            "generation": self.generation.to_dict(),
            "reward": self.reward.to_dict(),
            "seed": self.seed,
        }


def collect_rollouts(
    backend: Qwen3Backend,
    task: dict,
    prompt: str,
    num_rollouts: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int | None = 0,
) -> tuple[Rollout, ...]:
    """Sample a group and score every completion with the exact verifier."""
    if num_rollouts < 2:
        raise ValueError("GRPO requires at least two rollouts per prompt")
    rollouts = []
    for index in range(num_rollouts):
        rollout_seed = None if seed is None else seed + index
        generation = backend.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            seed=rollout_seed,
        )
        rollouts.append(
            Rollout(
                generation=generation,
                reward=verifier_reward(task, generation.text),
                seed=rollout_seed,
            )
        )
    return tuple(rollouts)


def group_relative_advantages(rewards: torch.Tensor | list[float], eps: float = 1e-4) -> torch.Tensor:
    """Normalize rewards within one prompt's rollout group.

    The unbiased sample standard deviation matches the book's Chapter 6
    walkthrough. If all rewards are equal, every advantage is zero and the
    policy receives no update from that uninformative group.
    """
    values = torch.as_tensor(rewards, dtype=torch.float32)
    if values.ndim != 1 or values.numel() < 2:
        raise ValueError("rewards must be a one-dimensional group of at least two values")
    if eps <= 0:
        raise ValueError("eps must be positive")
    return (values - values.mean()) / (values.std(unbiased=True) + eps)


def sequence_logprob(model: torch.nn.Module, token_ids: torch.Tensor, prompt_length: int) -> torch.Tensor:
    """Return the summed log-probability of completion tokens with gradients."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [sequence]")
    if prompt_length < 1 or prompt_length >= token_ids.numel():
        raise ValueError("prompt_length must leave at least one completion token")
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
    return selected[prompt_length - 1 :].sum()


def sequence_logprob_and_entropy(
    model: torch.nn.Module, token_ids: torch.Tensor, prompt_length: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return completion log-probability and mean token entropy."""
    if token_ids.ndim != 1:
        raise ValueError("token_ids must have shape [sequence]")
    if prompt_length < 1 or prompt_length >= token_ids.numel():
        raise ValueError("prompt_length must leave at least one completion token")
    logits = model(token_ids.unsqueeze(0)).squeeze(0).float()
    logprobs = torch.log_softmax(logits, dim=-1)
    selected = logprobs[:-1].gather(1, token_ids[1:].unsqueeze(-1)).squeeze(-1)
    start = prompt_length - 1
    completion_logprobs = selected[start:]
    completion_distributions = logprobs[start:-1]
    entropy = -(completion_distributions.exp() * completion_distributions).sum(dim=-1).mean()
    return completion_logprobs.sum(), entropy


def policy_gradient_loss(log_probs: torch.Tensor, advantages: torch.Tensor) -> torch.Tensor:
    """Compute Chapter 6's advantage-weighted sequence policy loss."""
    if log_probs.ndim != 1 or advantages.ndim != 1 or log_probs.shape != advantages.shape:
        raise ValueError("log_probs and advantages must be matching one-dimensional tensors")
    return -(advantages.detach() * log_probs).mean()


def clipped_policy_gradient_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """PPO-style sequence-ratio objective used by Chapter 7."""
    if clip_epsilon <= 0:
        raise ValueError("clip_epsilon must be positive")
    if not (log_probs.shape == old_log_probs.shape == advantages.shape):
        raise ValueError("log_probs, old_log_probs, and advantages must have matching shapes")
    ratios = torch.exp(log_probs - old_log_probs.detach())
    clipped = torch.clamp(ratios, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    advantage_signal = advantages.detach()
    objective = torch.minimum(ratios * advantage_signal, clipped * advantage_signal)
    loss = -objective.mean()
    clip_fraction = ((ratios != clipped).float()).mean()
    return loss, ratios, clip_fraction


def format_reward(output_text: str) -> float:
    """Reward the project's explicit ``\\boxed{...}`` output contract."""
    extraction = extract_final_candidate(output_text)
    return float(extraction.method == "boxed")


def kl_penalty(
    log_probs: torch.Tensor,
    reference_log_probs: torch.Tensor,
    coefficient: float,
    mode: str = "simple",
    ratios: torch.Tensor | None = None,
) -> torch.Tensor:
    """Penalize drift from a frozen reference policy."""
    if coefficient < 0:
        raise ValueError("coefficient must be non-negative")
    if mode not in {"simple", "reweighted"}:
        raise ValueError("mode must be 'simple' or 'reweighted'")
    if log_probs.shape != reference_log_probs.shape:
        raise ValueError("log_probs and reference_log_probs must have matching shapes")
    drift = log_probs - reference_log_probs.detach()
    if mode == "reweighted":
        if ratios is None:
            raise ValueError("reweighted KL requires policy ratios")
        drift = ratios.detach() * drift
    return coefficient * drift.mean()


@dataclass(frozen=True)
class GRPOStats:
    loss: torch.Tensor
    pg_loss: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    advantages: torch.Tensor
    rollouts: tuple[Rollout, ...]
    old_log_probs: torch.Tensor | None = None
    ratios: torch.Tensor | None = None
    clip_fraction: torch.Tensor | None = None
    kl_loss: torch.Tensor | None = None
    entropy: torch.Tensor | None = None
    format_rewards: torch.Tensor | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": float(self.loss.detach().item()),
            "pg_loss": float(self.pg_loss.detach().item()),
            "log_probs": self.log_probs.detach().cpu().tolist(),
            "rewards": self.rewards.detach().cpu().tolist(),
            "advantages": self.advantages.detach().cpu().tolist(),
            "rollouts": [rollout.to_dict() for rollout in self.rollouts],
            "old_log_probs": None
            if self.old_log_probs is None
            else self.old_log_probs.detach().cpu().tolist(),
            "ratios": None if self.ratios is None else self.ratios.detach().cpu().tolist(),
            "clip_fraction": None
            if self.clip_fraction is None
            else float(self.clip_fraction.detach().item()),
            "kl_loss": None if self.kl_loss is None else float(self.kl_loss.detach().item()),
            "entropy": None if self.entropy is None else float(self.entropy.detach().item()),
            "format_rewards": None
            if self.format_rewards is None
            else self.format_rewards.detach().cpu().tolist(),
        }


def compute_grpo_loss(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    rollouts: tuple[Rollout, ...],
    clip_epsilon: float | None = None,
    reference_model: torch.nn.Module | None = None,
    kl_coeff: float = 0.0,
    kl_mode: str = "simple",
    format_reward_weight: float = 0.0,
) -> GRPOStats:
    """Turn sampled rollouts into Chapter 6 or Chapter 7 loss statistics."""
    if len(rollouts) < 2:
        raise ValueError("GRPO requires at least two rollouts")
    if kl_coeff and reference_model is None:
        raise ValueError("reference_model is required when kl_coeff is non-zero")
    if format_reward_weight < 0:
        raise ValueError("format_reward_weight must be non-negative")
    device = next(model.parameters()).device
    prompt_ids = tokenizer.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt must tokenize to at least one token")
    prompt_length = len(prompt_ids)
    log_probs = []
    entropies = []
    old_log_probs = []
    reference_log_probs = []
    for rollout in rollouts:
        if not rollout.token_ids:
            raise ValueError("rollouts must contain at least one generated token")
        full_ids = torch.tensor(prompt_ids + list(rollout.token_ids), device=device, dtype=torch.long)
        logprob, entropy = sequence_logprob_and_entropy(model, full_ids, prompt_length)
        log_probs.append(logprob)
        entropies.append(entropy)
        if clip_epsilon is not None:
            old_log_probs.append(rollout.old_logprob)
        if kl_coeff:
            with torch.no_grad():
                reference_log_probs.append(sequence_logprob(reference_model, full_ids, prompt_length))
    log_probs_tensor = torch.stack(log_probs)
    old_log_probs_tensor = (
        torch.tensor(old_log_probs, device=device, dtype=log_probs_tensor.dtype)
        if clip_epsilon is not None
        else None
    )
    entropy_tensor = torch.stack(entropies).mean()
    verifier_rewards = torch.tensor(
        [rollout.reward.reward for rollout in rollouts], device=device, dtype=torch.float32
    )
    format_rewards = torch.tensor(
        [format_reward(rollout.text) for rollout in rollouts], device=device, dtype=torch.float32
    )
    rewards = verifier_rewards + format_reward_weight * format_rewards
    advantages = group_relative_advantages(rewards).to(device)
    ratios = None
    clip_fraction = None
    if clip_epsilon is None:
        pg_loss = policy_gradient_loss(log_probs_tensor, advantages)
    else:
        pg_loss, ratios, clip_fraction = clipped_policy_gradient_loss(
            log_probs_tensor, old_log_probs_tensor, advantages, clip_epsilon
        )
    kl_loss = None
    if kl_coeff:
        reference_tensor = torch.stack(reference_log_probs).to(device)
        kl_loss = kl_penalty(log_probs_tensor, reference_tensor, kl_coeff, kl_mode, ratios)
    else:
        reference_tensor = None
    loss = pg_loss if kl_loss is None else pg_loss + kl_loss
    return GRPOStats(
        loss,
        pg_loss,
        log_probs_tensor,
        rewards,
        advantages,
        rollouts,
        old_log_probs_tensor,
        ratios,
        clip_fraction,
        kl_loss,
        entropy_tensor,
        format_rewards,
    )


def save_grpo_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    metrics: list[dict[str, Any]],
    config: dict[str, Any],
) -> None:
    """Save enough state to resume a local GRPO experiment."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
        },
        destination,
    )


def load_grpo_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Restore model and optional optimizer state, returning the run ledger."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in payload:
        optimizer.load_state_dict(payload["optimizer_state_dict"])
    return payload


def train_grpo(
    backend: Qwen3Backend,
    tasks: list[dict],
    optimizer: torch.optim.Optimizer,
    steps: int,
    num_rollouts: int = 4,
    max_new_tokens: int = 64,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int | None = 0,
    grad_clip: float = 1.0,
    clip_epsilon: float | None = None,
    reference_model: torch.nn.Module | None = None,
    kl_coeff: float = 0.0,
    kl_mode: str = "simple",
    format_reward_weight: float = 0.0,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 0,
) -> list[dict[str, Any]]:
    """Run a small sequential GRPO loop and return its rollout ledger."""
    if not tasks:
        raise ValueError("tasks must not be empty")
    if steps < 1:
        raise ValueError("steps must be positive")
    if grad_clip <= 0:
        raise ValueError("grad_clip must be positive")
    metrics: list[dict[str, Any]] = []
    model = backend.model
    model.train()
    for step in range(1, steps + 1):
        task = tasks[(step - 1) % len(tasks)]
        prompt = render_prompt(task)
        optimizer.zero_grad(set_to_none=True)
        rollouts = collect_rollouts(
            backend,
            task,
            prompt,
            num_rollouts=num_rollouts,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=None if seed is None else seed + (step - 1) * num_rollouts,
        )
        model.train()
        stats = compute_grpo_loss(
            model,
            backend.tokenizer,
            prompt,
            rollouts,
            clip_epsilon=clip_epsilon,
            reference_model=reference_model,
            kl_coeff=kl_coeff,
            kl_mode=kl_mode,
            format_reward_weight=format_reward_weight,
        )
        has_policy_signal = torch.count_nonzero(stats.advantages).item() != 0
        has_kl_signal = kl_coeff != 0.0
        if not has_policy_signal and not has_kl_signal:
            # A homogeneous reward group has no relative learning signal.
            # Skipping backward/step also avoids meaningless zero-gradient
            # clipping diagnostics on low-precision backends.
            grad_norm = 0.0
            update_applied = False
            nonfinite_gradient = False
        else:
            stats.loss.backward()
            grad_norm_tensor = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(grad_norm_tensor):
                # Never write a checkpoint after a corrupting optimizer step.
                optimizer.zero_grad(set_to_none=True)
                grad_norm = None
                update_applied = False
                nonfinite_gradient = True
            else:
                grad_norm = float(grad_norm_tensor.item())
                optimizer.step()
                update_applied = True
                nonfinite_gradient = False
        row = {
            "step": step,
            "loss": float(stats.loss.detach().item()),
            "mean_reward": float(stats.rewards.mean().item()),
            "mean_advantage": float(stats.advantages.mean().item()),
            "mean_generated_tokens": sum(r.generation.generated_token_count for r in rollouts)
            / len(rollouts),
            "gradient_norm": grad_norm,
            "correct_count": sum(r.reward.status == "correct" for r in rollouts),
            "update_applied": update_applied,
            "nonfinite_gradient": nonfinite_gradient,
            "mean_ratio": None
            if stats.ratios is None
            else float(stats.ratios.detach().mean().item()),
            "clip_fraction": None
            if stats.clip_fraction is None
            else float(stats.clip_fraction.detach().item()),
            "kl_loss": None if stats.kl_loss is None else float(stats.kl_loss.detach().item()),
            "mean_entropy": None
            if stats.entropy is None
            else float(stats.entropy.detach().item()),
            "mean_format_reward": float(stats.format_rewards.mean().item()),
        }
        metrics.append(row)
        if checkpoint_path is not None and checkpoint_every and step % checkpoint_every == 0:
            save_grpo_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                step,
                metrics,
                {
                    "steps": steps,
                    "num_rollouts": num_rollouts,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                    "top_p": top_p,
                    "seed": seed,
                    "grad_clip": grad_clip,
                    "clip_epsilon": clip_epsilon,
                    "kl_coeff": kl_coeff,
                    "kl_mode": kl_mode,
                    "format_reward_weight": format_reward_weight,
                },
            )
    return metrics


def write_grpo_ledger(path: str | Path, metrics: list[dict[str, Any]]) -> None:
    """Write a compact JSON ledger independently of binary checkpoints."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
