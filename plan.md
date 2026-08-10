# Verifiable Reasoning Lab

Status: active build; decisions through Chapter 2 are recorded below

Project: Building a Small Reasoning Model — evaluating inference scaling, GRPO/RLVR, adaptive reasoning budgets, and optional distillation on verifiable tasks.

Repository: [Uzbekswe/verifiable-reasoning-lab](https://github.com/Uzbekswe/verifiable-reasoning-lab) (private during development)

Learning reference: [Sebastian Raschka's official reasoning-from-scratch repository](https://github.com/rasbt/reasoning-from-scratch), used as an attributed technical guide. This project is an independent implementation and experiment, not a copy of the companion notebooks.

## 1. Product definition

The primary product is a reproducible ML/AI engineering research system. It should let a reviewer reconstruct what was tested, how it was generated, how answers were verified, what resources were used, and whether a method helped.

The final portfolio package will contain:

- an independently organized Python package and CLI;
- a pinned Qwen3-0.6B base-model inference path;
- a frozen verifier-backed math/logic evaluation suite;
- comparable baseline, inference-scaling, self-refinement, RLVR/GRPO, and adaptive-budget runs;
- an evidence-gated decision about whether distillation is worth doing;
- a concise technical report in Markdown, reproducibility commands, failure analysis, and a small local Gradio demo if the experiment surface is stable enough;
- saved prompts, seeds, model revisions, generation settings, dataset hashes, logs, metrics, and resource measurements.

The target audience is ML/AI engineering hiring managers, with research-engineering rigor.

## 2. Success criteria

### Book-aligned method success

Each method is judged using the book's controlled-comparison approach:

1. Run the method against a frozen verifier-backed set.
2. Report final-answer accuracy and parsing/verifier outcomes.
3. Report compute required: attempts, generated tokens, latency, throughput, memory, and cloud cost where relevant.
4. Compare against the same base model and held-out tasks.
5. Call a method successful only when it improves the predeclared held-out result within the declared resource budget.

A regression or no-gain result is valid evidence. The project must not manufacture a positive result by changing the test set, prompt, model revision, or stopping rule after seeing outcomes.

### Project success

The project succeeds when the full comparison is reproducible and honestly interpretable, even if one or more methods fail to beat the base model. A portfolio claim is not “this small model became generally intelligent”; it is “these controlled reasoning interventions produced these measured changes on these verifiable tasks at this resource cost.”

## 3. Collaboration and teaching contract

For each major concept:

1. Teach the 20% of the concept that explains most of the implementation.
2. Connect it to the final experiment and explain the tradeoffs.
3. Implement the approved scope in the repository.
4. Walk through the important code, tests, and outputs.
5. Record what was learned, what was measured, and what remains uncertain.

The default is teach → implement → walkthrough. Small code-along exercises are added selectively for central ideas such as decoding, answer extraction, reward calculation, and GRPO advantages; project plumbing, packaging, and repeatability remain handled centrally.

## 4. Design principles

- A reasoning model is normally the same decoder-only Transformer plus different inference and post-training procedures; no magical reasoning layer is assumed.
- Keep one deep experiment-running interface. Hide orchestration, validation, retries, logging, provenance, and metric calculation behind it.
- Put a seam only where behavior genuinely varies and at least two adapters are justified.
- Treat the interface as the test surface. Tests assert observable outcomes, not internal implementation details.
- Freeze evaluation inputs before training. Never use the held-out test split for model selection.
- Prefer deterministic, exact, verifier-backed rewards over subjective judge scores for the core experiment.
- Keep raw model weights and caches out of Git; retain download provenance and cryptographic fingerprints.
- Make cloud execution opt-in, bounded, logged, and auto-terminating.

## 5. Repository design

The repository will grow toward this shape:

```text
verifiable-reasoning-lab/
├── plan.md
├── README.md                         # portfolio entry point and quickstart
├── pyproject.toml                    # pinned project metadata and tooling
├── uv.lock                           # authoritative environment lock
├── src/reasonlab/
│   ├── cli.py                        # thin command entry points
│   ├── config.py                     # validated experiment specifications
│   ├── models/                       # model-loading and generation adapters
│   ├── policies/                     # baseline, scaling, refinement, adaptive policies
│   ├── verification/                 # task parsing and exact verifiers
│   ├── evaluation/                   # run orchestration and metric aggregation
│   ├── training/                     # RLVR/GRPO and optional distillation
│   └── artifacts/                    # immutable run records and manifests
├── chapter_notes/                    # concise learning notes and code pointers
├── configs/                          # versioned, human-readable run settings
├── data/                             # manifests and small metadata; not model caches
├── experiments/                      # bounded launch scripts and analysis entry points
├── tests/                             # interface-level and verifier tests
└── .github/                          # CI and repository hygiene once code begins
```

The public CLI should eventually expose a small surface such as:

```text
reasonlab smoke --config configs/ch02_smoke.yaml
reasonlab evaluate --config configs/eval_baseline.yaml
reasonlab compare --run-dir artifacts/runs/<run-id>
reasonlab demo
```

The CLI is a caller of deep modules, not the place where model, verifier, or training logic lives.

## 6. Deep-module design

### Experiment runner

Interface: accept a resolved experiment specification and return a finalized run record or a typed incomplete-run result.

Implementation: validate settings, pin provenance, allocate budgets, dispatch tasks to a reasoning policy, write atomic attempt records, resume safely, and finalize metrics only when coverage and integrity checks pass.

### Model backend seam

Adapters: a local Qwen3-0.6B backend and a deterministic fake backend for tests. A bounded cloud/job adapter is introduced only when a real second production execution path exists.

Implementation responsibilities: tokenizer/model loading, device selection, generation, EOS handling, KV-cache behavior, timing, token counts, memory measurement, and failure classification.

### Reasoning-policy seam

Adapters: greedy baseline, temperature/top-p sampling, best-of-N/self-consistency, verifier-guided self-refinement, GRPO-trained policy, and adaptive budget policy.

The policy owns attempt allocation and stopping decisions. It does not own dataset truth or metric aggregation.

### Verifier seam

Adapters: exact numeric/algebraic answer verification and small logic-task verification. Optional Python tasks are deferred until a safe sandbox and unit-test contract are approved.

The verifier returns structured outcomes such as correct, incorrect, parse_error, verifier_error, or budget_exhausted. A verifier error is never silently counted as a wrong answer.

### Artifact ledger

The ledger stores immutable run manifests, raw outputs, parsed answers, verifier results, attempt order, settings, hashes, resource usage, and cost. A failed write prevents finalization; replay resumes from completed task records.

## 7. Study sequence and milestone gates

Every completed milestone receives one professional commit and is pushed immediately to the private GitHub repository. Commits describe the delivered capability, not internal milestone numbering. No milestone is declared complete until its tests and observable outputs have been checked.

### M0 — Project contract and repository foundation

Deliverables: this plan, repository hygiene, environment policy, contribution/attribution rules, and the first README skeleton.

Exit criteria: scope and decisions approved; repository is clean, private, and reproducible from a documented environment command.

### M1 — Chapter 2 generation foundation

Book concepts: tokenizer encode/decode, loading pretrained Qwen3-0.6B base weights, logits and greedy argmax, autoregressive generation, EOS stopping, streaming output, token/latency measurement, KV cache, and optional `torch.compile`.

Deliverables: pinned model manifest, local MPS/CPU device fallback, model backend interface, no-cache/cache parity test, smoke CLI, and generation logs.

Exit criteria: a clean local smoke run on the M1 MacBook Air; tokenizer round-trip tests pass; cache and no-cache outputs agree for a fixed seed/prompt; no cloud spend.

Status: complete. The pure-PyTorch Qwen3-0.6B adapter, greedy generation, EOS stopping,
streaming token path, device selection, atomic model download, provenance record, and
cache/no-cache parity tests are implemented. A real MPS smoke run is saved at
`artifacts/runs/ch02_smoke.json`; the downloaded model remains outside Git.

### M2 — Chapter 3 evaluator and frozen data

Book concepts: answer extraction, exact verification, accuracy evaluation, parser failures, and the limits of small benchmark samples.

Deliverables: primary original math/logic suite, secondary MATH-500 compatibility evaluation, fixed train/validation/test manifests, verifier tests, answer schema, baseline evaluator, and dataset hashes.

Exit criteria: verifier tests pass; every task has a stable ID and canonical answer; held-out test is frozen before training; baseline results are saved with complete provenance.

Status: core complete. The original primary suite contains 400/80/120 train/validation/test
tasks across arithmetic, algebra, sequence, and logic families. Manifest hashes are recorded
in `data/manifests/manifest.json`; the full 120-task base-model baseline is saved at
`artifacts/runs/ch03_baseline.json`. The MATH-500 compatibility loader accepts a user-supplied
licensed JSON file without bundling or silently downloading the dataset.

### M3 — Chapters 4–5 inference-time scaling

Book concepts: spend more inference compute without changing weights through sampling, best-of-N/self-consistency, and verifier-guided self-refinement.

Deliverables: paired comparisons for greedy, temperature/top-p sampling, best-of-N/self-consistency, and self-refinement; token/latency/accuracy curves; failure slices.

Exit criteria: all methods use the same frozen test set and prompt contract; compute budgets are visible; comparisons can be reproduced from saved run records.

Status: complete. Seeded temperature/top-p sampling, model-scored best-of-N,
self-consistency voting, and verifier-guided self-refinement are implemented
and tested. Full 120-task real-Qwen artifacts and paired comparison evidence are
saved for the Chapter 4 policies and the one-revision Chapter 5 policy. The
self-refinement loop records draft/revision histories, counts critique overhead,
stops early on verified drafts, and never gives the canonical answer to the
model. On this fixed suite, self-refinement reached 24.17% accuracy versus
14.17% for one sampled attempt, at 103.81 generated tokens and 7.30 seconds per
task. The result is evidence about this fixed suite and resource budget, not a
general benchmark claim; the high residual parse-error rate is an explicit
reason to make adaptive compute selective in M5.

### M4 — Chapters 6–7 RLVR/GRPO

Book concepts: verifiable rewards, group-relative advantages, policy-gradient loss, clipping/KL choices, rollout collection, and training diagnostics.

Deliverables: small train/validation run, reward unit tests, rollout ledger, training checkpoints, validation tracking, and one bounded cloud experiment only if M2–M3 are trustworthy.

Exit criteria: reward computation is independently tested; training can resume or stop safely; held-out evaluation is performed once per selected checkpoint; cost remains inside the approved cap.

Status: Chapters 6–7 implementation complete; the bounded reduced-parameter evidence gate is
closed, while full-model training remains unclaimed. The binary
verifier reward, grouped rollout collector, group-relative advantages,
gradient-bearing sequence log-probabilities, policy-gradient loss, checkpoint
round-trip, CLI, and deterministic tiny-policy smoke test are implemented and
tested. The tiny policy moves its approved-token probability from 0.333 to
0.676. The all-parameter Qwen3/MPS diagnostic produced a mixed reward (3/4
correct) but a non-finite bfloat16 gradient; the safety gate skipped the update.
The approved local reduced-parameter path now freezes the Transformer and trains
only `out_head`: the same diagnostic produced a finite gradient norm of 113.5
and applied the update, with 155.6M of 751.6M parameters trainable. This is a
numerical/plumbing gate, not a full-policy improvement claim. No cloud money was
spent. Chapter 7 clipping, entropy, optional KL, and format-reward diagnostics
are now implemented and tested. A reduced-scope one-step run applied a finite
update with clipping enabled. The local gate then used four GRPO steps, four
rollouts, and an `out_head` scope; its checkpoint was evaluated on all 80
validation tasks with the same sampled policy as the base. Base and checkpoint
both scored 16/80 (20.0%), with 64 parse errors, 41.525 generated tokens per
task, and no verifier errors. The checkpoint therefore provides reproducible
negative evidence rather than a held-out gain. The training run, base
validation, checkpoint validation, and config files are saved under
`artifacts/runs/` and `configs/`. No cloud money was spent. A bounded GPU
full-model diagnostic remains optional and requires explicit approval.

### M5 — Original adaptive reasoning budget

Deliverables: a policy that starts with cheap generation and spends additional attempts/refinement only when verifier or confidence signals indicate difficulty; fixed-budget and adaptive-budget comparisons.

Exit criteria: paired held-out results include accuracy, tokens, latency, and cost; the policy is evaluated against equal-cost or equal-token baselines; any “improvement” is described as an accuracy-compute tradeoff, not just accuracy.

### M6 — Chapter 8 distillation decision and optional experiment

Decision gate: distillation proceeds only if the prior experiments show a useful teacher/student tradeoff and the remaining budget supports a bounded run.

If approved: use only open/licensed teachers and data; start with hard/sequence distillation. Explain why soft/KL distillation requires a shared tokenizer and teacher logits. If not approved: write a measured “not pursued” decision with the evidence and budget reason.

### M7 — Portfolio packaging

Deliverables: README, reproducibility commands, concise technical report, final tables/plots, failure analysis, model/data provenance, test suite, and local CLI plus Gradio demo if stable.

Exit criteria: a clean checkout can run the smoke test; every reported number maps to a saved artifact; no benchmark or portfolio claim exceeds the evidence; final project remains private until owner approval.

## 8. Evaluation contract

Primary evidence will be an original, verifier-backed suite of short math and logic tasks. MATH-500 is secondary and clearly labeled for comparison with the book.

Approved initial scale:

- 400 training tasks;
- 80 validation tasks;
- 120 frozen test tasks.

Each task should include a stable ID, family, difficulty tag, prompt version, canonical answer, verifier version, license/source metadata, and split. The test manifest must be immutable once M2 begins.

Per-run records should include:

- model and tokenizer identifier plus revision/hash;
- code revision, dependency lock, hardware/device, and seed;
- prompt template and generation settings;
- raw output, parsed answer, verifier result, and failure reason;
- attempts, generated tokens, latency, throughput, peak memory, and cost;
- aggregate accuracy, parse rate, verifier-error rate, per-family/difficulty results, and paired failure examples.

## 9. Budget and execution policy

- M0–M3: local only unless a separately approved diagnostic is necessary.
- Hard project cloud cap: $24.
- Reserve: $6 for one correction or final bounded run.
- Every cloud job requires a written forecast, exact stop condition, maximum wall time, automatic termination, and owner approval.
- No interactive GPU workspace is left running while reviewing results.
- If pricing or resource availability changes, re-estimate before launch; do not assume the prior price.

## 10. Backup plans

| Risk | Primary response | Fallback | What we report |
| --- | --- | --- | --- |
| Qwen3 load is too slow on the M1 | Use MPS with CPU fallback and short smoke prompts | Run only the smallest parity tests locally; reserve cloud for approved evaluation | Device, time, and omitted scope |
| MPS is unstable or unsupported | Keep CPU as the authoritative correctness path | Use cloud only for a bounded performance run | Backend differences and reproducibility limits |
| Model download or cache is corrupted | Atomic download plus SHA-256 verification | Redownload from the documented licensed source | Revision, hash, and failure log |
| Parser/verifier ambiguity | Start with exact numeric and simple logic tasks | Narrow task families; fail closed on verifier errors | Coverage and excluded cases |
| Benchmark contamination is plausible | Make original suite primary and MATH-500 secondary | Report only the original suite for project claims | Dataset provenance and limitation |
| GRPO exceeds memory or cost | Reduce rollout count/context and validate rewards offline | Skip training and report a justified non-run | Budget, hardware, and decision evidence |
| Adaptive policy adds tokens without value | Compare equal-token/equal-cost baselines | Keep it as a negative-result extension | Accuracy-compute frontier |
| Distillation is not justified | Stop at the evidence gate | Publish the decision not to distill | Teacher, budget, and expected-value analysis |
| Demo becomes a distraction | Keep CLI and artifacts authoritative | Ship a CLI-only release | Scope decision and reason |

## 11. Decisions resolved before Chapter 2

The owner approved the following operating decisions:

1. Use the original verifier-backed suite as primary evidence and MATH-500 as clearly labeled secondary compatibility evidence.
2. Start with a 400/80/120 train/validation/frozen-test split, subject to a justified revision during Chapter 3 data inspection.
3. Implement the Chapter 2 model adapter with the book's readable pure-PyTorch Qwen path first; add a Transformers adapter only if a later training or deployment need justifies it.
4. Keep the hard $24 cloud cap, $6 reserve, written forecast, exact stop condition, automatic termination, and owner approval for every cloud job.
5. Ship a CLI-first experiment surface; add a small local Gradio demo only after the evaluator and experiment plumbing are stable.

### Chapter 1 checkpoint

- A reasoning system normally keeps the same decoder-only Transformer architecture as its base model; reasoning behavior comes from inference procedures and post-training, not a magical reasoning layer.
- The verifier is a shared contract: it evaluates answers, supplies verifiable rewards for RLVR/GRPO, and can provide structured feedback for refinement.
- The next key distinction is inference-time compute (spending more tokens/attempts with unchanged weights) versus weight updates (training changes parameters and therefore future behavior).

M0, M1, M2, M3, and the reduced-parameter M4 evidence gate are complete.
Chapters 6–7 reward, GRPO, and stability implementations are complete, and
the reduced-parameter local training path is numerically safe. The measured
checkpoint did not improve held-out accuracy, so full-model Qwen training
remains unclaimed after the all-parameter MPS failure. M5 adaptive-budget
experiments are next; no cloud job runs without explicit approval.

## 12. Source and attribution notes

- Official course source: https://github.com/rasbt/reasoning-from-scratch
- Official Chapter 1: https://github.com/rasbt/reasoning-from-scratch/tree/main/ch01
- Official Chapter 2: https://github.com/rasbt/reasoning-from-scratch/tree/main/ch02
- Official Chapter 6 walkthrough: https://github.com/rasbt/reasoning-from-scratch/blob/main/ch06/01_main-chapter-code/ch06_main.ipynb
- Official Chapter 7 walkthrough: https://github.com/rasbt/reasoning-from-scratch/blob/main/ch07/01_main-chapter-code/ch07_main.ipynb
- Official Qwen3-from-scratch weight source: https://huggingface.co/rasbt/qwen3-from-scratch
- The project will include attribution and preserve upstream model/data license notices.
- No unauthorized book copy or book excerpt is required for this plan.
