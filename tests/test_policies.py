from reasonlab.models.backend import GenerationResult
from reasonlab.policies import adaptive_budget, best_of_n, fixed_verifier_budget, self_consistency


class StubBackend:
    def __init__(self, outputs):
        self.outputs = outputs

    def generate(self, prompt, **kwargs):
        index = kwargs["seed"] - 10
        answer, score = self.outputs[index]
        return GenerationResult(
            text=f"\\boxed{{{answer}}}",
            token_ids=[index],
            prompt_token_count=1,
            generated_token_count=1,
            elapsed_seconds=0.1,
            tokens_per_second=10.0,
            device="cpu",
            use_cache=True,
            stopped_on_eos=True,
            mean_logprob=score,
            temperature=0.8,
            top_p=0.9,
            seed=kwargs["seed"],
        )


def test_best_of_n_uses_model_score_not_answer_truth():
    result = best_of_n(StubBackend([("wrong", -0.2), ("right", -0.8), ("also", -0.1)]), "prompt", n=3, seed=10)
    assert result.text == r"\boxed{also}"
    assert result.selection["selected_index"] == 2
    assert result.generated_token_count == 3


def test_self_consistency_votes_extracted_answers():
    result = self_consistency(StubBackend([("A", -0.2), ("B", -0.3), ("A", -0.4)]), "prompt", n=3, seed=10)
    assert result.text == r"\boxed{A}"
    assert result.selection["vote_counts"] == {"A": 2, "B": 1}
    assert result.selection["tie"] is False


class AdaptiveStubBackend:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def generate(self, prompt, **kwargs):
        index = len(self.calls)
        self.calls.append(kwargs)
        answer, score = self.outputs[index]
        return GenerationResult(
            text=answer,
            token_ids=[index],
            prompt_token_count=1,
            generated_token_count=2,
            elapsed_seconds=0.1,
            tokens_per_second=20.0,
            device="cpu",
            use_cache=True,
            stopped_on_eos=True,
            mean_logprob=score,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            seed=kwargs.get("seed"),
        )


def test_adaptive_budget_stops_after_verified_first_attempt():
    backend = AdaptiveStubBackend([(r"\boxed{4}", -0.1), (r"\boxed{5}", -0.2)])
    result = adaptive_budget(
        backend,
        {"answer": "4", "answer_type": "numeric"},
        "What is 2 + 2?",
        max_extra_attempts=2,
        seed=10,
    )
    assert len(backend.calls) == 1
    assert result.selection["stop_reason"] == "initial_verifier_success"
    assert result.generated_token_count == 2


def test_fixed_verifier_budget_always_spends_all_attempts():
    backend = AdaptiveStubBackend([(r"\boxed{5}", -0.1), (r"\boxed{4}", -0.2), (r"\boxed{6}", -0.3)])
    result = fixed_verifier_budget(
        backend,
        {"answer": "4", "answer_type": "numeric"},
        "What is 2 + 2?",
        attempts=3,
        seed=10,
    )
    assert len(backend.calls) == 3
    assert result.selection["selected_by"] == "verifier_success"
    assert result.selection["selected_index"] == 1


def test_adaptive_budget_escalates_only_after_verifier_failure():
    backend = AdaptiveStubBackend([(r"\boxed{5}", -0.1), (r"\boxed{4}", -0.2)])
    result = adaptive_budget(
        backend,
        {"answer": "4", "answer_type": "numeric"},
        "What is 2 + 2?",
        max_extra_attempts=2,
        seed=10,
    )
    assert len(backend.calls) == 2
    assert result.text.endswith(r"\boxed{4}")
    assert result.selection["selected_by"] == "verifier_success"


def test_adaptive_budget_uses_low_confidence_gate_for_third_attempt():
    backend = AdaptiveStubBackend(
        [(r"\boxed{5}", -0.1), (r"\boxed{6}", -0.8), (r"\boxed{4}", -0.2)]
    )
    result = adaptive_budget(
        backend,
        {"answer": "4", "answer_type": "numeric"},
        "What is 2 + 2?",
        max_extra_attempts=2,
        confidence_threshold=-0.5,
        seed=10,
    )
    assert len(backend.calls) == 3
    assert result.text.endswith(r"\boxed{4}")
    assert result.selection["stop_reason"] == "second_escalation_verifier_success"
