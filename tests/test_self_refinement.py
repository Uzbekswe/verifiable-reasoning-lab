from reasonlab.models.backend import GenerationResult
from reasonlab.policies import self_refine


class RefineStubBackend:
    def __init__(self):
        self.calls = []

    def generate(self, prompt, **kwargs):
        self.calls.append(prompt)
        if "Review this math" in prompt:
            text = "Recheck the arithmetic before revising."
        elif "Revise a math" in prompt:
            text = r"Recomputed result: \boxed{4}"
        else:
            text = r"Initial result: \boxed{5}"
        return GenerationResult(
            text=text,
            token_ids=[1],
            prompt_token_count=2,
            generated_token_count=1,
            elapsed_seconds=0.01,
            tokens_per_second=100.0,
            stopped_on_eos=False,
            device="cpu",
            use_cache=True,
            mean_logprob=-0.1,
            temperature=kwargs.get("temperature"),
            top_p=kwargs.get("top_p"),
            seed=kwargs.get("seed"),
        )


def test_self_refine_revises_until_verifier_passes():
    backend = RefineStubBackend()
    result = self_refine(
        backend,
        {"answer": "4", "answer_type": "numeric"},
        "What is 2 + 2?",
        max_refinements=1,
    )
    assert result.selected.text.endswith(r"\boxed{4}")
    assert result.selection["revision_count"] == 1
    assert result.selection["stopped_early"] is True
    assert result.to_dict()["overhead_calls"] == 1
    assert result.generated_token_count == 3


def test_self_refine_stops_without_overhead_when_draft_is_correct():
    backend = RefineStubBackend()
    backend.generate = lambda prompt, **kwargs: GenerationResult(
        text=r"\boxed{4}",
        token_ids=[1],
        prompt_token_count=2,
        generated_token_count=1,
        elapsed_seconds=0.01,
        tokens_per_second=100.0,
        stopped_on_eos=False,
        device="cpu",
        use_cache=True,
        mean_logprob=-0.1,
    )
    result = self_refine(
        backend,
        {"answer": "4", "answer_type": "numeric"},
        "What is 2 + 2?",
        max_refinements=2,
    )
    assert result.selection["revision_count"] == 0
    assert result.to_dict()["overhead_calls"] == 0
