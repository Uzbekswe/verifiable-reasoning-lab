"""Experiment-facing Qwen3 adapter and generation provenance."""

from __future__ import annotations

import hashlib
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import torch

from .qwen3 import QWEN3_06B_CONFIG, KVCache, Qwen3Model, Qwen3Tokenizer

MODEL_REPO = "https://huggingface.co/rasbt/qwen3-from-scratch/resolve/main"


@dataclass(frozen=True)
class GenerationResult:
    text: str
    token_ids: list[int]
    prompt_token_count: int
    generated_token_count: int
    elapsed_seconds: float
    tokens_per_second: float
    device: str
    use_cache: bool
    stopped_on_eos: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def select_device(preferred: str = "auto") -> torch.device:
    if preferred != "auto":
        device = torch.device(preferred)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available")
        return device
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        return torch.device("xpu")
    return torch.device("cpu")


def _sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(url, headers={"User-Agent": "verifiable-reasoning-lab/0.1"})
    with urlopen(request, timeout=60) as response, temporary.open("wb") as output:
        while chunk := response.read(8 * 1024 * 1024):
            output.write(chunk)
    temporary.replace(destination)
    return destination


def ensure_qwen3_files(model_dir: str | Path, download: bool = False) -> tuple[Path, Path]:
    model_dir = Path(model_dir)
    model_path = model_dir / "qwen3-0.6B-base.pth"
    tokenizer_path = model_dir / "tokenizer-base.json"
    missing = [path for path in (model_path, tokenizer_path) if not path.is_file()]
    if missing and not download:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(
            f"Missing {names}. Re-run with --download to fetch licensed files from "
            f"{MODEL_REPO}, or place them in {model_dir}."
        )
    if not model_path.is_file():
        _download(f"{MODEL_REPO}/qwen3-0.6B-base.pth", model_path)
    if not tokenizer_path.is_file():
        _download(f"{MODEL_REPO}/tokenizer-base.json", tokenizer_path)
    return model_path, tokenizer_path


class Qwen3Backend:
    """Deep adapter: one generation operation, complete local provenance."""

    def __init__(self, model: Qwen3Model, tokenizer: Qwen3Tokenizer, device: torch.device, model_path: Path):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.model_path = model_path

    @classmethod
    def from_pretrained(
        cls, model_dir: str | Path, device: str = "auto", download: bool = False
    ) -> Qwen3Backend:
        model_path, tokenizer_path = ensure_qwen3_files(model_dir, download=download)
        target = select_device(device)
        model = Qwen3Model(QWEN3_06B_CONFIG)
        state = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        model.to(target)
        tokenizer = Qwen3Tokenizer(tokenizer_path)
        return cls(model, tokenizer, target, model_path)

    def generate(self, prompt: str, max_new_tokens: int = 64, use_cache: bool = True) -> GenerationResult:
        if not prompt.strip():
            raise ValueError("prompt must contain non-whitespace text")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device, dtype=torch.long)
        output_ids, elapsed, stopped_on_eos, prompt_length = greedy_generate(
            self.model, prompt_ids, max_new_tokens, self.tokenizer.eos_token_id, use_cache
        )
        token_count = int(output_ids.shape[1])
        return GenerationResult(
            text=self.tokenizer.decode(output_ids[0]),
            token_ids=output_ids[0].tolist(),
            prompt_token_count=prompt_length,
            generated_token_count=token_count,
            elapsed_seconds=elapsed,
            tokens_per_second=token_count / elapsed if token_count else 0.0,
            device=str(self.device),
            use_cache=use_cache,
            stopped_on_eos=stopped_on_eos,
        )

    def provenance(self) -> dict[str, Any]:
        return {
            "model_id": "rasbt/qwen3-from-scratch:qwen3-0.6B-base",
            "model_path": str(self.model_path),
            "model_sha256": _sha256(self.model_path),
            "device": str(self.device),
            "torch_version": torch.__version__,
        }

    def stream(self, prompt: str, max_new_tokens: int = 64, use_cache: bool = True):
        """Yield decoded token pieces while preserving the same greedy policy."""
        prompt_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device, dtype=torch.long)
        for token_id in greedy_stream(
            self.model, prompt_ids, max_new_tokens, self.tokenizer.eos_token_id, use_cache
        ):
            yield self.tokenizer.decode([token_id])


@torch.inference_mode()
def greedy_generate(model, prompt_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None, use_cache: bool):
    """Return a GenerationResult-like token sequence for a batch of one."""
    if prompt_ids.shape[0] != 1:
        raise ValueError("Chapter 2 generation currently supports batch size 1")
    model.eval()
    prompt_length = prompt_ids.shape[1]
    generated: list[torch.Tensor] = []
    stopped_on_eos = False
    start = time.perf_counter()
    if use_cache:
        cache = KVCache(n_layers=model.cfg.n_layers)
        model.reset_kv_cache()
        logits = model(prompt_ids, cache=cache)[:, -1]
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                stopped_on_eos = True
                break
            generated.append(next_token)
            logits = model(next_token, cache=cache)[:, -1]
    else:
        tokens = prompt_ids
        for _ in range(max_new_tokens):
            logits = model(tokens)[:, -1]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            if eos_token_id is not None and int(next_token.item()) == eos_token_id:
                stopped_on_eos = True
                break
            generated.append(next_token)
            tokens = torch.cat((tokens, next_token), dim=1)
    elapsed = max(time.perf_counter() - start, 1e-9)
    output_ids = torch.cat(generated, dim=1) if generated else prompt_ids[:, :0]
    return output_ids, elapsed, stopped_on_eos, prompt_length


@torch.inference_mode()
def greedy_stream(model, prompt_ids: torch.Tensor, max_new_tokens: int, eos_token_id: int | None, use_cache: bool):
    """Yield greedy token IDs as soon as each token is selected."""
    if prompt_ids.shape[0] != 1:
        raise ValueError("Chapter 2 generation currently supports batch size 1")
    model.eval()
    if use_cache:
        cache = KVCache(n_layers=model.cfg.n_layers)
        model.reset_kv_cache()
        logits = model(prompt_ids, cache=cache)[:, -1]
        for _ in range(max_new_tokens):
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                return
            yield token_id
            logits = model(next_token, cache=cache)[:, -1]
    else:
        tokens = prompt_ids
        for _ in range(max_new_tokens):
            logits = model(tokens)[:, -1]
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                return
            yield token_id
            tokens = torch.cat((tokens, next_token), dim=1)
