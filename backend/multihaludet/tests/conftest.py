"""
Fixtures for the multihaludet test suite.

None of these tests download a real model or hit the network - the
whole point of `score_existing_response` being deterministic,
teacher-forced local inference is that it's testable without either.
`FakeCausalLM` / `FakeTokenizer` stand in for `AutoModelForCausalLM` /
`AutoTokenizer` with the same call signatures
`HFGenerationBackend.score_existing_response` actually uses, wired
directly onto a real `HFGenerationBackend` instance (bypassing
`_ensure_loaded`, which is the only part of that class that would need
network/GPU/`transformers`).
"""

from __future__ import annotations

import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Make the `backend/` directory (containing the `multihaludet`, `config`,
# etc. packages) importable regardless of pytest's invocation cwd.
_BACKEND_DIR = Path(__file__).resolve().parents[2]
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from multihaludet.generation_backend import (  # noqa: E402
    _CHAT_SYSTEM_FALLBACK,
    GenerationBundle,
    HFGenerationBackend,
)

VOCAB_SIZE = 64
HIDDEN_SIZE = 8
NUM_LAYERS = 3


class _FakeConfig:
    def __init__(self, num_hidden_layers: int, hidden_size: int) -> None:
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size


class FakeCausalLM(torch.nn.Module):
    """Minimal stand-in for a HF `AutoModelForCausalLM`. Deterministic
    (fixed seed) so hidden states/logits can be independently
    recomputed and checked in tests. Only implements the subset of the
    HF interface `score_existing_response` calls: a single
    `forward(input_ids, output_hidden_states=True, use_cache=False)`."""

    def __init__(
        self, vocab_size: int = VOCAB_SIZE, hidden_size: int = HIDDEN_SIZE, num_layers: int = NUM_LAYERS
    ) -> None:
        super().__init__()
        self.config = _FakeConfig(num_layers, hidden_size)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        g = torch.Generator().manual_seed(0)
        # Registered as buffers (not nn.Parameters) - this backbone is
        # frozen / never trained, matching the real Qwen backbone.
        self.register_buffer("embedding_table", torch.randn(vocab_size, hidden_size, generator=g))
        self.register_buffer("lm_head_weight", torch.randn(hidden_size, vocab_size, generator=g))

    def eval(self):  # noqa: D401 - mirrors nn.Module.eval() no-op-ish behavior
        return super().eval()

    def forward(self, input_ids: torch.Tensor, output_hidden_states: bool = True, use_cache: bool = False):
        base = self.embedding_table[input_ids[0]]  # [seq_len, H]
        # Layer l's hidden state is a simple deterministic function of
        # the token at that position (base * (l+1)) - no real attention
        # needed, just something reproducible to assert against.
        hidden_states = tuple((base * (layer + 1)).unsqueeze(0) for layer in range(self.num_layers + 1))
        final_hidden = hidden_states[-1].squeeze(0)  # [seq_len, H]
        logits = (final_hidden @ self.lm_head_weight).unsqueeze(0)  # [1, seq_len, V]
        return SimpleNamespace(hidden_states=hidden_states, logits=logits)


class FakeTokenizer:
    """Whitespace/word-level fake tokenizer - deterministic id assignment
    via crc32 (not Python's randomized `hash()`), no vocab download."""

    def __init__(self, vocab_size: int = VOCAB_SIZE) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def _encode_word(self, word: str) -> int:
        return 2 + (zlib.crc32(word.encode("utf-8")) % (self.vocab_size - 2))

    def __call__(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True):
        ids = [self._encode_word(w) for w in text.split()]
        return SimpleNamespace(input_ids=torch.tensor([ids], dtype=torch.long))

    def apply_chat_template(self, messages, add_generation_prompt: bool = True, return_tensors: str = "pt"):
        # No chat template - forces score_existing_response /
        # generate_with_states down their plain-concat fallback path,
        # which is what we want to exercise deterministically here.
        raise NotImplementedError("FakeTokenizer has no chat template")

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(int(i)) for i in ids.tolist())


def fallback_prompt_text(query: str, system: str | None = None) -> str:
    """Reproduces the exact fallback prompt string
    HFGenerationBackend.score_existing_response builds when
    apply_chat_template is unavailable, so tests can independently
    recompute expected token positions."""
    system_prompt = system or _CHAT_SYSTEM_FALLBACK
    return f"{system_prompt}\n\nUser: {query}\nAssistant:"


@pytest.fixture
def fake_backend() -> HFGenerationBackend:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    backend = HFGenerationBackend(device=device)
    backend._tokenizer = FakeTokenizer()
    backend._model = FakeCausalLM().to(backend.device)
    backend._num_layers = backend._model.num_layers
    backend._hidden_size = backend._model.hidden_size
    backend._ensure_loaded = lambda: None  # no real load; already "loaded" above
    return backend


def make_synthetic_bundle(
    num_layers: int = NUM_LAYERS,
    num_steps: int = 5,
    hidden_size: int = HIDDEN_SIZE,
    vocab_size: int = VOCAB_SIZE,
    seed: int = 0,
) -> GenerationBundle:
    """A GenerationBundle with the right shapes but no dependency on any
    tokenizer/model - for testing pipeline.py's MultiHaluDetModel in
    isolation from the generation backend."""
    rng = np.random.default_rng(seed)
    return GenerationBundle(
        text="synthetic",
        layer_step_hidden=rng.normal(size=(num_layers + 1, num_steps, hidden_size)).astype(np.float32),
        step_logits=rng.normal(size=(num_steps, vocab_size)).astype(np.float32),
        chosen_token_ids=rng.integers(0, vocab_size, size=(num_steps,)).astype(np.int64),
        num_layers=num_layers,
        hidden_size=hidden_size,
        prompt_token_count=3,
    )


@pytest.fixture
def tiny_model_kwargs() -> dict:
    """Small MultiHaluDetModel hyperparameters so tests run fast on CPU."""
    return dict(
        hidden_size=HIDDEN_SIZE,
        num_sampled_layers=3,
        attention_scales=[1, 2],
        encoder_dim=12,
        encoder_heads=2,
        encoder_layers=1,
        global_top_k=3,
        global_hidden_dim=8,
        ensemble_members=3,
    )
