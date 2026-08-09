import torch

from reasonlab.models.backend import greedy_generate, greedy_stream


class DeterministicToyModel:
    """Tiny next-token oracle used to test generation invariants without weights."""

    def __init__(self, sequence):
        self.cfg = type("Config", (), {"n_layers": 1})()
        self.sequence = list(sequence)
        self.position = 0

    def eval(self):
        return self

    def reset_kv_cache(self):
        self.position = 0

    def __call__(self, input_ids, cache=None):
        if cache is None:
            self.position = input_ids.shape[1] - 2
        token = self.sequence[self.position]
        self.position += 1
        logits = torch.full((1, input_ids.shape[1], 16), -100.0)
        logits[:, -1, token] = 100.0
        return logits


def test_greedy_generation_stops_on_eos():
    model = DeterministicToyModel([3, 4, 2, 9])
    output, _, stopped, prompt_length = greedy_generate(
        model, torch.tensor([[7, 8]]), max_new_tokens=8, eos_token_id=2, use_cache=False
    )
    assert output.tolist() == [[3, 4]]
    assert stopped is True
    assert prompt_length == 2


def test_cached_and_uncached_generation_match():
    prompt = torch.tensor([[7, 8]])
    uncached, _, _, _ = greedy_generate(
        DeterministicToyModel([3, 4, 5, 6]), prompt, 3, eos_token_id=None, use_cache=False
    )
    cached, _, _, _ = greedy_generate(
        DeterministicToyModel([3, 4, 5, 6]), prompt, 3, eos_token_id=None, use_cache=True
    )
    assert torch.equal(uncached, cached)


def test_streaming_selects_the_same_tokens():
    tokens = list(
        greedy_stream(
            DeterministicToyModel([3, 4, 5, 6]),
            torch.tensor([[7, 8]]),
            max_new_tokens=3,
            eos_token_id=None,
            use_cache=True,
        )
    )
    assert tokens == [3, 4, 5]
