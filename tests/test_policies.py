from reasonlab.models.backend import GenerationResult
from reasonlab.policies import best_of_n, self_consistency


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
