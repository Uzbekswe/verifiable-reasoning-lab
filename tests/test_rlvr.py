import torch

from reasonlab.models.backend import GenerationResult
from reasonlab.rl_smoke import TinyPolicy, TinyTokenizer
from reasonlab.rlvr import (
    Rollout,
    clipped_policy_gradient_loss,
    compute_grpo_loss,
    configure_trainable_scope,
    format_reward,
    group_relative_advantages,
    kl_penalty,
    load_grpo_checkpoint,
    policy_gradient_loss,
    save_grpo_checkpoint,
    sequence_logprob,
    verifier_reward,
)


class HeadModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.out_head = torch.nn.Linear(2, 3)

    def forward(self, input_ids):
        hidden = self.backbone(torch.ones(*input_ids.shape, 2))
        return self.out_head(hidden)


def test_verifier_reward_is_binary_and_format_sensitive():
    task = {"answer": "4", "answer_type": "numeric"}
    assert verifier_reward(task, r"\boxed{4}").reward == 1.0
    assert verifier_reward(task, "4").reward == 0.0
    assert verifier_reward(task, r"\boxed{5}").reward == 0.0


def test_output_head_scope_freezes_backbone():
    model = HeadModel()
    summary = configure_trainable_scope(model, "output_head")
    assert summary["scope"] == "output_head"
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.out_head.parameters())
    assert summary["trainable_parameters"] < summary["total_parameters"]


def test_group_relative_advantages_match_chapter_formula():
    advantages = group_relative_advantages(torch.tensor([1.0, 1.0, 0.0, 0.0]))
    assert torch.allclose(advantages, torch.tensor([0.8659, 0.8659, -0.8659, -0.8659]), atol=1e-3)
    assert torch.equal(group_relative_advantages([1.0, 1.0]), torch.zeros(2))


def test_clipped_policy_gradient_limits_large_ratio():
    new = torch.tensor([0.0, 0.0], requires_grad=True)
    old = torch.tensor([-3.0, 0.0])
    advantages = torch.tensor([1.0, -1.0])
    loss, ratios, clip_fraction = clipped_policy_gradient_loss(new, old, advantages, 0.2)
    assert torch.allclose(ratios, torch.tensor([20.0855, 1.0]), atol=1e-3)
    assert clip_fraction.item() == 0.5
    loss.backward()
    assert new.grad is not None


def test_kl_penalty_supports_simple_and_reweighted_modes():
    new = torch.tensor([-1.0, -2.0], requires_grad=True)
    reference = torch.tensor([-1.5, -1.0])
    ratios = torch.tensor([2.0, 0.5])
    simple = kl_penalty(new, reference, 0.1)
    reweighted = kl_penalty(new, reference, 0.1, mode="reweighted", ratios=ratios)
    assert simple.item() < 0
    assert reweighted.item() > 0


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


def test_format_reward_only_accepts_boxed_contract():
    assert format_reward(r"\boxed{4}") == 1.0
    assert format_reward("Final answer: 4") == 0.0


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


def test_compute_grpo_loss_can_enable_chapter7_terms():
    model = TinyPolicy()
    generation = GenerationResult(
        text=r"\boxed{4}",
        token_ids=[1],
        prompt_token_count=1,
        generated_token_count=1,
        elapsed_seconds=0.001,
        tokens_per_second=1000.0,
        device="cpu",
        use_cache=False,
        stopped_on_eos=False,
        mean_logprob=-1.0,
    )
    rollout = Rollout(
        generation=generation,
        reward=verifier_reward({"answer": "4", "answer_type": "numeric"}, generation.text),
        seed=0,
    )
    second = Rollout(
        generation=GenerationResult(
            **{**generation.to_dict(), "text": r"\boxed{5}", "token_ids": [2], "mean_logprob": -1.0}
        ),
        reward=verifier_reward({"answer": "4", "answer_type": "numeric"}, r"\boxed{5}"),
        seed=1,
    )
    stats = compute_grpo_loss(
        model,
        TinyTokenizer(),
        "What is 2 + 2?",
        (rollout, second),
        clip_epsilon=0.2,
        format_reward_weight=0.1,
    )
    assert stats.ratios is not None
    assert stats.clip_fraction is not None
    assert stats.entropy is not None
    stats.loss.backward()


def test_compute_grpo_loss_can_add_reference_kl_penalty():
    model = TinyPolicy()
    reference = TinyPolicy()
    reference.logits.data[1] = 0.5
    generation = GenerationResult(
        text=r"\boxed{4}",
        token_ids=[1],
        prompt_token_count=1,
        generated_token_count=1,
        elapsed_seconds=0.001,
        tokens_per_second=1000.0,
        device="cpu",
        use_cache=False,
        stopped_on_eos=False,
        mean_logprob=-1.0,
    )
    other = GenerationResult(**{**generation.to_dict(), "text": r"\boxed{5}", "token_ids": [2]})
    task = {"answer": "4", "answer_type": "numeric"}
    rollouts = tuple(
        Rollout(generation=g, reward=verifier_reward(task, g.text), seed=index)
        for index, g in enumerate((generation, other))
    )
    stats = compute_grpo_loss(
        model,
        TinyTokenizer(),
        "What is 2 + 2?",
        rollouts,
        reference_model=reference,
        kl_coeff=0.02,
    )
    assert stats.kl_loss is not None
    stats.loss.backward()


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
