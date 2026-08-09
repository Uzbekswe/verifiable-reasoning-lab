"""Small Chapter 2 command-line surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import tomllib

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


def main() -> int:
    parser = argparse.ArgumentParser(prog="reasonlab")
    subparsers = parser.add_subparsers(dest="command", required=True)
    smoke = subparsers.add_parser("smoke", help="run a local Qwen3 Chapter 2 generation")
    smoke.add_argument("--config", default="configs/ch02_smoke.toml")
    smoke.add_argument("--download", action="store_true", help="download missing model files")
    smoke.set_defaults(handler=_smoke)
    args = parser.parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
