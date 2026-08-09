"""Small Chapter 2 command-line surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tomllib

from .datasets import load_split, write_manifests
from .evaluation import evaluate_tasks
from .models.backend import Qwen3Backend


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
    result = evaluate_tasks(
        tasks,
        lambda prompt: backend.generate(prompt, max_new_tokens=max_new_tokens, use_cache=use_cache),
    )
    result["provenance"] = backend.provenance()
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
    smoke = subparsers.add_parser("smoke", help="run a local Qwen3 Chapter 2 generation")
    smoke.add_argument("--config", default="configs/ch02_smoke.toml")
    smoke.add_argument("--download", action="store_true", help="download missing model files")
    smoke.set_defaults(handler=_smoke)
    data = subparsers.add_parser("data", help="build or inspect frozen task manifests")
    data_subparsers = data.add_subparsers(dest="data_command", required=True)
    build = data_subparsers.add_parser("build", help="generate original Chapter 3 manifests")
    build.add_argument("--out-dir", default="data/manifests")
    build.add_argument("--seed", type=int, default=20260809)
    build.set_defaults(handler=_build_data)
    evaluate = subparsers.add_parser("evaluate", help="evaluate Qwen3 on a frozen manifest split")
    evaluate.add_argument("--config", default="configs/ch03_eval.toml")
    evaluate.add_argument("--limit", type=int, default=0, help="optional bounded prefix for diagnostics")
    evaluate.add_argument("--download", action="store_true", help="download missing model files")
    evaluate.set_defaults(handler=_evaluate)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
