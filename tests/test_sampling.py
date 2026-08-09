import pytest
import torch

from reasonlab.models.backend import generate_sequence


class FlatLogitModel:
    def __init__(self, vocab_size=8):
        self.cfg = type("Config", (), {"n_layers": 1})()
        self.vocab_size = vocab_size

    def eval(self):
        return self

    def reset_kv_cache(self):
        return None

    def __call__(self, input_ids, cache=None):
        logits = torch.zeros(1, input_ids.shape[1], self.vocab_size)
        return logits


def test_seeded_sampling_is_reproducible_and_cache_invariant():
    prompt = torch.tensor([[1, 2]])
    first = generate_sequence(FlatLogitModel(), prompt, 5, None, True, 0.8, 0.9, seed=42)[0]
    second = generate_sequence(FlatLogitModel(), prompt, 5, None, True, 0.8, 0.9, seed=42)[0]
    uncached = generate_sequence(FlatLogitModel(), prompt, 5, None, False, 0.8, 0.9, seed=42)[0]
    assert torch.equal(first, second)
    assert torch.equal(first, uncached)


def test_sampling_parameters_are_validated():
    with pytest.raises(ValueError, match="temperature"):
        generate_sequence(FlatLogitModel(), torch.tensor([[1]]), 2, None, True, -1.0, 0.9)
    with pytest.raises(ValueError, match="top_p"):
        generate_sequence(FlatLogitModel(), torch.tensor([[1]]), 2, None, True, 0.8, 0.0)
