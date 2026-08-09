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

Chapter 4 policies use the same evaluator and frozen split:

```bash
uv run reasonlab evaluate --config configs/ch04_sample.toml --limit 8
uv run reasonlab evaluate --config configs/ch04_best_of_n.toml --limit 8
uv run reasonlab evaluate --config configs/ch04_eval.toml --limit 8
```

Best-of-N is scored by mean raw model log-probability. Self-consistency votes on
extracted final answers. Neither policy uses the ground-truth verifier to select
an answer.

The full 120-task comparison is recorded in
[`ch04_comparison.json`](artifacts/runs/ch04_comparison.json). On this fixed
suite, one sampled attempt reached 14.17% accuracy, best-of-4 reached 10.00%,
and self-consistency-4 reached 10.83%. The four-attempt methods used about
3.7× the generated tokens per task, so the result is an accuracy/compute tradeoff,
not an unconditional improvement.

The project is private during development. See [`plan.md`](plan.md) for the
complete scope, milestone gates, budget policy, and backup plans.

## Chapter 5: verifier-guided self-refinement

Self-refinement keeps the Qwen3 weights fixed. It generates a draft, runs the
same exact verifier used for evaluation, asks the model for a short critique,
and gives the critique plus verifier feedback to a revision prompt. The
feedback is deliberately fail-closed: it can identify a parse/format problem or
say that exact verification failed, but it never contains the canonical answer.
The loop stops immediately when the draft verifies, so easy tasks do not pay
the critique cost.

```bash
uv run reasonlab evaluate --config configs/ch05_eval.toml --limit 8
uv run reasonlab evaluate --config configs/ch05_eval.toml
```

The full 120-task run is recorded in
[`ch05_self_refinement.json`](artifacts/runs/ch05_self_refinement.json). With
one permitted revision, accuracy was 24.17% (29/120), compared with 14.17%
for the one-sample Chapter 4 policy. The improvement came with a cost: mean
generated tokens rose to 103.81 and mean latency to 7.30 seconds; 104 of 120
tasks used the critique/revision path. Of the 75 final parse failures, 74 were
already parse failures after the draft and remained failures after revision.
These results motivate the next adaptive-budget milestone: spend refinement
only when a cheap format-aware signal predicts that it is likely to help.
