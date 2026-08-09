import torch

from reasonlab.models.backend import GenerationResult
from reasonlab.rl_smoke import TinyPolicy, TinyTokenizer
from reasonlab.rlvr import (
    Rollout,
    compute_grpo_loss,
    group_relative_advantages,
    load_grpo_checkpoint,
    policy_gradient_loss,
    save_grpo_checkpoint,
    sequence_logprob,
    verifier_reward,
)


def test_verifier_reward_is_binary_and_format_sensitive():
    task = {"answer": "4", "answer_type": "numeric"}
    assert verifier_reward(task, r"\boxed{4}").reward == 1.0
    assert verifier_reward(task, "4").reward == 0.0
    assert verifier_reward(task, r"\boxed{5}").reward == 0.0


def test_group_relative_advantages_match_chapter_formula():
    advantages = group_relative_advantages(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(advantages, torch.tensor([0.8659, 0.8659, -0.8659, -0.8659]), atol=1e-3)
    assert torch.equal(group_relative_advantages([1.0, 1.0]), torch.zeros(2))


def test_policy_gradient_detaches_advantages_but_keeps_logprob_gradient():
    log_probs = torch.tensor([-1.0, -2.0], requires_grad=True)
    advantages = torch.tensor([1.0, -1.0], requires_grad=True)
    loss = policy_gradient_loss(log_probs, advantages)
    loss.backward()
    assert torch.allclose(log_probs.grad, torch.tensor([-0.5, 0.5]))
    assert advantages.grad is None


def test_sequence_logprob_backpropagates_through_tiny_policy():
    model = TinyPolicy()
    token_ids = torch.tensor([0, 1])
    logprob = sequence_logprob(model, token_ids, prompt_length=1)
    logprob.backward()
    assert logprob.ndim == 0
    assert model.logits.grad is not None


def test_compute_grpo_loss_produces_differentiable_stats():
    model = TinyPolicy()
    tokenizer = TinyTokenizer()
    generations = []
    for token_id, text in ((1, r"\boxed{4}"), (2, r"\boxed{5}")):
        generation = GenerationResult(
            text=text,
            token_ids=[token_id],
            prompt_token_count=1,
            generated_token_count=1,
            elapsed_seconds=0.001,
            tokens_per_second=1000.0,
            device="cpu",
            use_cache=False,
            stopped_on_eos=False,
        )
        generations.append(
            Rollout(
                generation=generation,
                reward=verifier_reward({"answer": "4", "answer_type": "numeric"}, text),
                seed=token_id,
            )
        )
    stats = compute_grpo_loss(
        model,
        tokenizer,
        "What is 2 + 2?",
        tuple(generations),
    )
    stats.loss.backward()
    assert stats.loss.requires_grad
    assert stats.rewards.tolist() == [1.0, 0.0]
    assert model.logits.grad is not None


def test_grpo_checkpoint_round_trip(tmp_path):
    model = TinyPolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model.logits.data[1] = 2.0
    path = tmp_path / "grpo.pt"
    save_grpo_checkpoint(path, model, optimizer, 3, [{"step": 3}], {"lr": 0.1})

    restored = TinyPolicy()
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.1)
    payload = load_grpo_checkpoint(path, restored, restored_optimizer)
    assert payload["step"] == 3
    assert torch.equal(restored.logits, model.logits)
