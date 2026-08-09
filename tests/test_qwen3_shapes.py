import torch

from reasonlab.models.backend import greedy_generate
from reasonlab.models.qwen3 import KVCache, Qwen3Config, Qwen3Model


def test_tiny_qwen3_forward_supports_cached_decode():
    config = Qwen3Config(
        vocab_size=32,
        context_length=32,
        emb_dim=16,
        n_heads=4,
        n_layers=2,
        hidden_dim=32,
        head_dim=4,
        n_kv_groups=2,
        dtype=torch.float32,
    )
    model = Qwen3Model(config).eval()
    prompt = torch.tensor([[1, 2, 3]])
    uncached = model(prompt)[:, -1]
    cache = KVCache(config.n_layers)
    model.reset_kv_cache()
    cached = model(prompt, cache=cache)[:, -1]
    next_cached = model(torch.tensor([[4]]), cache=cache)[:, -1]
    assert uncached.shape == (1, config.vocab_size)
    assert cached.shape == uncached.shape
    assert next_cached.shape == uncached.shape


def test_tiny_qwen3_cached_and_uncached_tokens_match():
    config = Qwen3Config(
        vocab_size=32,
        context_length=32,
        emb_dim=16,
        n_heads=4,
        n_layers=2,
        hidden_dim=32,
        head_dim=4,
        n_kv_groups=2,
        dtype=torch.float32,
    )
    torch.manual_seed(7)
    cached_model = Qwen3Model(config).eval()
    uncached_model = Qwen3Model(config).eval()
    uncached_model.load_state_dict(cached_model.state_dict())
    prompt = torch.tensor([[1, 2, 3]])
    uncached, _, _, _ = greedy_generate(uncached_model, prompt, 5, None, use_cache=False)
    cached, _, _, _ = greedy_generate(cached_model, prompt, 5, None, use_cache=True)
    assert torch.equal(uncached, cached)
