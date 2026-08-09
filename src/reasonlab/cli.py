"""Small command-line surface for generation and evaluation experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tomllib
import torch

from .datasets import load_split, write_manifests
from .evaluation import evaluate_tasks
from .models.backend import Qwen3Backend
from .policies import best_of_n, self_consistency, self_refine
from .rl_smoke import run_tiny_grpo_smoke
from .rlvr import configure_trainable_scope, train_grpo


def _smoke(args: argparse.Namespace) -> int:
    config = {}
    if args.config:
        with Path(args.config).open("rb") as handle:
            config = tomllib.load(handle)
    backend = Qwen3Backend.from_pretrained(
        config.get("model_dir", ".cache/models/qwen3"),
        device=config.get("device", "auto"),
        download=args.download,
    )
    result = backend.generate(
        config.get("prompt", "Explain why 2 + 2 = 4 in one sentence."),
        max_new_tokens=int(config.get("max_new_tokens", 32)),
        use_cache=bool(config.get("use_cache", True)),
    )
    record = {"provenance": backend.provenance(), "generation": result.to_dict()}
    output_path = config.get("output")
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    return 0


def _build_data(args: argparse.Namespace) -> int:
    metadata = write_manifests(args.out_dir, seed=args.seed)
    print(json.dumps(metadata, indent=2))
    return 0


def _grpo_smoke(args: argparse.Namespace) -> int:
    with Path(args.config).open("rb") as handle:
        config = tomllib.load(handle)
    result = run_tiny_grpo_smoke(
        steps=int(config.get("steps", 6)),
        num_rollouts=int(config.get("num_rollouts", 4)),
    )
    output_path = config.get("output")
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


def _grpo_train(args: argparse.Namespace) -> int:
    with Path(args.config).open("rb") as handle:
        config = tomllib.load(handle)
    tasks = load_split(config.get("manifest_dir", "data/manifests"), config.get("split", "train"))
    if args.limit:
        tasks = tasks[: args.limit]
    backend = Qwen3Backend.from_pretrained(
        config.get("model_dir", ".cache/models/qwen3"),
        device=config.get("device", "auto"),
        download=args.download,
    )
    trainable_scope = str(config.get("trainable_scope", "all"))
    parameter_summary = configure_trainable_scope(backend.model, trainable_scope)
    trainable_parameters = [parameter for parameter in backend.model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise ValueError("training scope selected no trainable parameters")
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(config.get("learning_rate", 1e-6)),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    metrics = train_grpo(
        backend,
        tasks,
        optimizer,
        steps=int(config.get("steps", 1)),
        num_rollouts=int(config.get("num_rollouts", 2)),
        max_new_tokens=int(config.get("max_new_tokens", 64)),
        temperature=float(config.get("temperature", 0.8)),
        top_p=float(config.get("top_p", 0.9)),
        seed=config.get("seed", 0),
        grad_clip=float(config.get("grad_clip", 1.0)),
        checkpoint_path=config.get("checkpoint"),
        checkpoint_every=int(config.get("checkpoint_every", 0)),
    )
    result = {
        "provenance": backend.provenance(),
        "config": config,
        "parameter_summary": parameter_summary,
        "metrics": metrics,
    }
    output_path = config.get("output")
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"provenance": result["provenance"], "metrics": metrics}, indent=2))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    with Path(args.config).open("rb") as handle:
        config = tomllib.load(handle)
    tasks = load_split(config.get("manifest_dir", "data/manifests"), config.get("split", "test"))
    if args.limit:
        tasks = tasks[: args.limit]
    backend = Qwen3Backend.from_pretrained(
        config.get("model_dir", ".cache/models/qwen3"),
        device=config.get("device", "auto"),
        download=args.download,
    )
    max_new_tokens = int(config.get("max_new_tokens", 64))
    use_cache = bool(config.get("use_cache", True))
    policy = config.get("policy", "greedy")
    temperature = float(config.get("temperature", 0.8))
    top_p = float(config.get("top_p", 0.9))
    attempts = int(config.get("attempts", 4))
    seed_base = config.get("seed", 0)
    task_index = 0
    seed_stride = 3 if policy == "self_refinement" else max(attempts, 1)

    def generate(prompt):
        nonlocal task_index
        task_seed = None if seed_base is None else int(seed_base) + task_index * seed_stride
        task_index += 1
        if policy == "greedy":
            return backend.generate(prompt, max_new_tokens=max_new_tokens, use_cache=use_cache)
        if policy == "sample":
            return backend.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                use_cache=use_cache,
                temperature=temperature,
                top_p=top_p,
                seed=task_seed,
            )
        if policy == "best_of_n":
            return best_of_n(
                backend,
                prompt,
                n=attempts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=task_seed,
            )
        if policy == "self_consistency":
            return self_consistency(
                backend,
                prompt,
                n=attempts,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=task_seed,
            )
        raise ValueError(f"unsupported policy: {policy}")

    generate_task = None
    if policy == "self_refinement":
        def generate_task(task, prompt):
            nonlocal task_index
            task_seed = None if seed_base is None else int(seed_base) + task_index * seed_stride
            task_index += 1
            return self_refine(
                backend,
                task,
                prompt,
                max_refinements=int(config.get("max_refinements", 1)),
                max_new_tokens=max_new_tokens,
                critique_max_tokens=int(config.get("critique_max_tokens", 48)),
                temperature=temperature,
                top_p=top_p,
                seed=task_seed,
            )

    result = evaluate_tasks(
        tasks,
        generate,
        generate_task=generate_task,
    )
    result["provenance"] = backend.provenance()
    result["policy_config"] = {
        "policy": policy,
        "attempts": attempts,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed_base,
        "max_new_tokens": max_new_tokens,
        "max_refinements": int(config.get("max_refinements", 1)),
        "critique_max_tokens": int(config.get("critique_max_tokens", 48)),
    }
    output_path = config.get("output")
    if output_path:
        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"provenance": result["provenance"], "summary": result["summary"]}, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="reasonlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run a local Qwen3 generation smoke test")
    smoke.add_argument("--config", default="configs/ch02_smoke.toml")
    smoke.add_argument("--download", action="store_true", help="download missing model files")
    smoke.set_defaults(handler=_smoke)
    data = subparsers.add_parser("data", help="build or inspect frozen task manifests")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    build = data_subparsers.add_parser("build", help="generate original Chapter 3 manifests")
    build.add_argument("--out-dir", default="data/manifests")
    build.add_argument("--seed", type=int, default=20260809)
    build.set_defaults(handler=_build_data)
    grpo_smoke = subparsers.add_parser("grpo-smoke", help="run the deterministic tiny GRPO training smoke test")
    grpo_smoke.add_argument("--config", default="configs/ch06_grpo_smoke.toml")
    grpo_smoke.set_defaults(handler=_grpo_smoke)
    grpo_train = subparsers.add_parser("grpo-train", help="run an explicit Qwen3 GRPO training experiment")
    grpo_train.add_argument("--config", default="configs/ch06_grpo_train.toml")
    grpo_train.add_argument("--limit", type=int, default=0, help="optional bounded prefix for diagnostics")
    grpo_train.add_argument("--download", action="store_true", help="download missing model files")
    grpo_train.set_defaults(handler=_grpo_train)
    evaluate = subparsers.add_parser("evaluate", help="evaluate Qwen3 on a frozen manifest split")
    evaluate.add_argument("--config", default="configs/ch03_eval.toml")
    evaluate.add_argument("--limit", type=int, default=0, help="optional bounded prefix for diagnostics")
    evaluate.add_argument("--download", action="store_true", help="download missing model files")
    evaluate.set_defaults(handler=_evaluate)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
