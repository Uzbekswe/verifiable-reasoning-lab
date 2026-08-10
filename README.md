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

## Chapter 6: RLVR and GRPO foundation

RLVR changes model weights using a reward computed by the existing verifier.
For this project the reward is intentionally binary: a correctly parsed and
verified answer receives `1.0`; an incorrect answer, parse failure, or verifier
failure receives `0.0`. GRPO samples a group of answers for one prompt, computes
`(reward - group_mean) / (group_std + epsilon)`, and maximizes the
advantage-weighted summed log-probability of the sampled completions. This is
the unclipped Chapter 6 objective; KL penalties and clipping are deferred to
Chapter 7.

```bash
uv run reasonlab grpo-smoke --config configs/ch06_grpo_smoke.toml
uv run reasonlab grpo-train --config configs/ch06_grpo_train.toml --limit 1
```

The tiny deterministic smoke test is a real optimizer loop: the approved-token
probability rose from 0.333 to 0.676 after six steps. An all-parameter Qwen3
diagnostic previously produced a non-finite MPS/bfloat16 gradient, so the safety
gate refused the update. The safer local path freezes the Transformer backbone
and trains only `out_head`: the same one-prompt/four-rollout diagnostic produced
three correct and one failed rollout (`mean_reward=0.75`), a finite gradient
norm of `113.5`, and an applied update. This is a numerical and plumbing
success, not evidence that the full Qwen3 policy improved: 155.6M of 751.6M
parameters were trainable. The run is saved in
[`ch06_grpo_train.json`](artifacts/runs/ch06_grpo_train.json). The checkpoint is
kept under `.cache/` and excluded from Git because it is multi-gigabyte.

## Chapter 7: GRPO stability and diagnostics

Chapter 7 adds safeguards around the Chapter 6 objective without changing the
verifier. The implementation tracks completion entropy and old/new policy
ratios, supports PPO-style sequence-ratio clipping, supports an optional frozen
reference-policy KL penalty, and exposes an optional explicit-format reward.
All terms are configurable; disabling them reproduces the Chapter 6 loss.

```bash
uv run reasonlab grpo-train --config configs/ch07_grpo_stable.toml --limit 1
```

The reduced-parameter Qwen3 diagnostic enabled clipping (`epsilon=0.2`) and a
small format-reward weight (`0.1`). It produced a finite gradient norm of
`125.5`, applied the update, and recorded mean ratio `1.058`, clip fraction
`0.0`, entropy `0.521`, and mean reward `0.825` (the latter includes format
shaping). This is a one-step stability diagnostic—not a held-out performance
claim—and it deliberately leaves KL disabled on the real Qwen run because a
full frozen reference copy is memory-expensive. KL behavior is covered by
deterministic unit tests.

### M4 evidence gate: bounded train/validation result

The local evidence gate uses the same frozen manifests, seed, sampling policy,
and verifier before and after training. The bounded run trains only `out_head`
for four GRPO steps, with four rollouts per prompt and a `0.1` format-reward
weight; the 2.0 GB checkpoint stays in `.cache/` and is intentionally not
committed.

```bash
uv run reasonlab evaluate --config configs/m4_validation_base.toml
uv run reasonlab grpo-train --config configs/m4_gate_train.toml --limit 8
uv run reasonlab evaluate --config configs/m4_validation_checkpoint.toml
```

| policy | validation accuracy | parse errors | mean tokens | mean latency |
| --- | ---: | ---: | ---: | ---: |
| base, sampled | 16/80 (20.0%) | 64 | 41.525 | 2.568 s |
| output-head checkpoint | 16/80 (20.0%) | 64 | 41.525 | 2.552 s |

The checkpoint matches the base on every family and difficulty slice. The
training loop itself was numerically safe: three of four steps applied finite
updates and one step correctly skipped because its grouped rewards had zero
variance. This closes the reduced-parameter local evidence gate, but it is not
evidence that all 751.6M Qwen3 parameters improved. The earlier all-parameter
MPS/bfloat16 run remains negative numerical evidence; no cloud GPU experiment
was purchased or started.

## M5: adaptive reasoning budget

The adaptive policy treats the verifier as a compute-allocation signal, not as
an answer generator. It starts with a 32-token attempt, stops immediately when
that attempt verifies, spends one 64-token escalation after a failure, and
spends a third attempt only when the remaining failure is a parse failure or
below the predeclared mean-logprob threshold (`-0.5`). A matched fixed policy
always spends three 64-token attempts and uses the same verifier-based
selection rule. This makes the comparison an allocation experiment rather than
a comparison of different answer selectors.

```bash
uv run reasonlab evaluate --config configs/m5_fixed_budget_test.toml
uv run reasonlab evaluate --config configs/m5_adaptive_test.toml
```

On the untouched 120-task test split:

| policy | accuracy | mean tokens/task | mean latency/task | verifier-selected answers |
| --- | ---: | ---: | ---: | ---: |
| fixed 3-attempt budget | 44/120 (36.67%) | 124.98 | 10.19 s | 44 |
| adaptive budget | 43/120 (35.83%) | 91.76 | 8.24 s | 43 |

Adaptive inference saved 3,987 generated tokens overall (26.6%) and 1.95
seconds per task, at a 0.83 percentage-point accuracy cost. It stopped after
one attempt on 9 tasks, two attempts on 21, and all three on 90. This is the
intended portfolio result: a measurable accuracy/compute tradeoff with saved
per-task decisions, not a claim that adaptive inference universally improves
accuracy. The validation artifacts are in
[`m5_fixed_budget_validation.json`](artifacts/runs/m5_fixed_budget_validation.json)
and [`m5_adaptive_validation.json`](artifacts/runs/m5_adaptive_validation.json);
the final test artifacts are in
[`m5_fixed_budget_test.json`](artifacts/runs/m5_fixed_budget_test.json) and
[`m5_adaptive_test.json`](artifacts/runs/m5_adaptive_test.json).
