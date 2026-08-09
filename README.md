# Verifiable Reasoning Lab

An independent, reproducible study of small reasoning-model methods on
verifier-backed tasks. The project follows Sebastian Raschka's official
[reasoning-from-scratch repository](https://github.com/rasbt/reasoning-from-scratch)
as a technical guide, while keeping its experiment code and evidence separate.

## Chapter 2 quickstart

```bash
uv sync --extra dev
uv run pytest
uv run reasonlab smoke --config configs/ch02_smoke.toml --download
```

The last command downloads the licensed Qwen3-0.6B base checkpoint to
`.cache/models/qwen3/`, runs deterministic greedy generation, and writes the
small provenance record to `artifacts/runs/ch02_smoke.json`. Model weights and
caches are intentionally excluded from Git.

Chapter 2 establishes the pure-PyTorch generation substrate: tokenization,
autoregressive greedy decoding, EOS stopping, streaming, device selection,
latency/token accounting, and KV-cache parity. Evaluation and training begin
only after this path is reliable.

## Chapter 3 quickstart

```bash
uv run reasonlab data build --out-dir data/manifests
uv run reasonlab evaluate --config configs/ch03_eval.toml
```

The primary suite is original and generated deterministically from a recorded
seed. MATH-500 remains a secondary compatibility path and must be supplied from
a verified, licensed local file; it is not silently downloaded or bundled.

The project is private during development. See [`plan.md`](plan.md) for the
complete scope, milestone gates, budget policy, and backup plans.
