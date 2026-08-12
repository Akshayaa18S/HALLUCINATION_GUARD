"""
Local in-process LLM backend for the MultiHaluDet branch.

Replaces Ollama for this pipeline. Ollama only exposes a REST endpoint
that returns final text; MultiHaluDet needs the model's internal
hidden-state trajectory and token-level logits during generation, which
is only obtainable by holding the model's weights in-process (HF
`transformers`) and reading `model.generate(..., output_hidden_states=True,
output_scores=True, return_dict_in_generate=True)`.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger("hallucination_guard.multihaludet.generation_backend")

_CHAT_SYSTEM_FALLBACK = (
    "You are a factual AI assistant. Answer queries concisely using only well-known facts. "
    "Do not invent names, dates, statistics, or achievements."
)


@dataclass
class GenerationBundle:
    """Everything the MultiHaluDet feature pipeline needs from one generation call.

    Shapes (T = number of generated tokens, L = total model layers,
    H = hidden size, V = vocab size):
      - layer_step_hidden: float32 array [L+1, T, H] - hidden state of the
        last position at each generation step, for every layer including
        the embedding output (layer 0). Layer 0 is the embedding output,
        layers 1..L are transformer block outputs, matching HF's
        `hidden_states` tuple convention.
      - step_logits: float32 array [T, V] - the logits HF returned for
        each generated token (i.e. the distribution *at* that step, not a
        one-hot of the chosen token) - required for top-k prob / entropy /
        logit-statistics global features.
      - chosen_token_ids: int array [T]
    """

    text: str
    layer_step_hidden: np.ndarray
    step_logits: np.ndarray
    chosen_token_ids: np.ndarray
    num_layers: int  # L (transformer blocks, excludes embedding layer)
    hidden_size: int
    prompt_token_count: int
    step_entropy: np.ndarray | None = None
    query: str = ""

    def is_empty(self) -> bool:
        return self.step_logits.shape[0] == 0


class GenerationBackendError(Exception):
    pass


class HFGenerationBackend:
    """Thin, lazily-initialized wrapper around a local HF causal LM.

    Loaded once per process (model weights are not tiny) and reused for
    every request. Thread-safe lazy init via a lock; generation itself is
    left to the caller to serialize if needed (a single small model on
    CPU should not be called concurrently from multiple requests without
    care - that's the same constraint Ollama has today, just made local).
    """

    def __init__(
        self,
        model_name: str | None = None,
        device: str | None = None,
        dtype: str | None = None,
        max_new_tokens: int | None = None,
    ) -> None:
        from config import settings

        self.model_name = model_name or settings.multihaludet_model_name
        self.device = device or settings.multihaludet_device
        self.dtype_name = dtype or settings.multihaludet_dtype
        self.max_new_tokens = max_new_tokens or settings.multihaludet_max_new_tokens

        self._model = None
        self._tokenizer = None
        self._num_layers: int | None = None
        self._hidden_size: int | None = None
        self._lock = threading.Lock()

    # -- lazy load -----------------------------------------------------

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if self.model_name.upper() == "MOCK":
                self._num_layers = 28
                self._hidden_size = 2048
                self._model = "MOCK"
                self._tokenizer = None
                return
            try:
                import torch
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:
                logger.warning(
                    "Torch/Transformers not installed. Operating HFGenerationBackend in mock evaluation mode."
                )
                self._num_layers = 28
                self._hidden_size = 2048
                self._model = "MOCK"
                self._tokenizer = None
                return

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            torch_dtype = dtype_map.get(self.dtype_name, torch.float32)

            if self.model_name == "fake":
                self._num_layers = 1
                self._hidden_size = 256
                self._model = "MOCK"
                self._tokenizer = None
                return

            logger.info(
                "Loading MultiHaluDet generation model %s (device=%s, dtype=%s)",
                self.model_name,
                self.device,
                self.dtype_name,
            )
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self._model = AutoModelForCausalLM.from_pretrained(
                    self.model_name, torch_dtype=torch_dtype, low_cpu_mem_usage=True
                )
            except Exception as exc:
                if self.model_name == "fake":
                    self._num_layers = 1
                    self._hidden_size = 256
                    self._model = "MOCK"
                    self._tokenizer = None
                    return
                raise GenerationBackendError(f"Failed to load HF generation model {self.model_name}: {exc}") from exc

            self._model.to(self.device)
            self._model.eval()

            config = self._model.config
            self._num_layers = getattr(config, "num_hidden_layers", None) or getattr(
                config, "n_layer", None
            )
            self._hidden_size = getattr(config, "hidden_size", None) or getattr(
                config, "n_embd", None
            )
            if self._num_layers is None or self._hidden_size is None:
                raise GenerationBackendError(
                    f"Could not determine num_layers/hidden_size from "
                    f"{self.model_name}'s config; MultiHaluDet needs both."
                )

    @property
    def num_layers(self) -> int:
        self._ensure_loaded()
        assert self._num_layers is not None
        return self._num_layers

    @property
    def hidden_size(self) -> int:
        self._ensure_loaded()
        assert self._hidden_size is not None
        return self._hidden_size

    def is_available(self) -> bool:
        try:
            self._ensure_loaded()
            return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("MultiHaluDet generation backend unavailable: %s", exc)
            return False

    # -- generation ------------------------------------------------------

    def generate_with_states(
        self,
        prompt: str,
        system: str | None = None,
        do_sample: bool = False,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
    ) -> GenerationBundle:
        """Generate a response and capture per-layer hidden states + logits
        for every generated token, in the same forward pass that produced
        the answer (so the internal signal reflects the actual response,
        not a second, separately-sampled pass)."""
        if self._model == "MOCK":
            rng = np.random.RandomState(hash(prompt) % (2**32 - 1))
            num_tokens = 15
            num_layers = self.num_layers
            hidden_dim = self.hidden_size
            vocab_size = 32000

            layer_step_hidden = rng.normal(0, 1, size=(num_layers + 1, num_tokens, hidden_dim)).astype(np.float32)
            step_logits = rng.normal(0, 1, size=(num_tokens, vocab_size)).astype(np.float32)
            chosen_token_ids = rng.randint(0, vocab_size, size=num_tokens)

            return GenerationBundle(
                text=f"Mock response for: {prompt}",
                layer_step_hidden=layer_step_hidden,
                step_logits=step_logits,
                chosen_token_ids=chosen_token_ids,
                num_layers=num_layers,
                hidden_size=hidden_dim,
                prompt_token_count=10,
            )

        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        system_prompt = system or _CHAT_SYSTEM_FALLBACK
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            # Base (non-chat-templated) models: fall back to plain concat.
            text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
            input_ids = tokenizer(text, return_tensors="pt").input_ids

        input_ids = input_ids.to(self.device)
        prompt_token_count = int(input_ids.shape[1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": do_sample,
            "repetition_penalty": 1.1,
            "output_hidden_states": True,
            "output_scores": True,
            "return_dict_in_generate": True,
            "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }

        if do_sample:
            gen_kwargs["temperature"] = temperature if temperature is not None else 0.7
            gen_kwargs["top_p"] = top_p if top_p is not None else 0.8
            gen_kwargs["top_k"] = top_k if top_k is not None else 20
        else:
            gen_kwargs["temperature"] = None
            gen_kwargs["top_p"] = None
            gen_kwargs["top_k"] = None
            if hasattr(model, "generation_config") and model.generation_config is not None:
                model.generation_config.temperature = None
                model.generation_config.top_p = None
                model.generation_config.top_k = None

        with torch.no_grad():
            out = model.generate(input_ids, **gen_kwargs)

        sequences = out.sequences  # [1, prompt_len + T]
        generated_ids = sequences[0, prompt_token_count:]
        text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        # out.hidden_states: tuple length T, each a tuple length (L+1) of
        # [1, seq_len_at_that_step, H]. Step 0 covers the whole prompt (we
        # only want the last position, i.e. the first generated token's
        # conditioning state); steps 1..T-1 cover exactly one new token.
        step_hidden_states = out.hidden_states or ()
        num_steps = len(step_hidden_states)
        num_layers_plus_embed = len(step_hidden_states[0]) if num_steps else 0

        if num_steps == 0:
            return GenerationBundle(
                text=text,
                layer_step_hidden=np.zeros((0, 0, self.hidden_size), dtype=np.float32),
                step_logits=np.zeros((0, 0), dtype=np.float32),
                chosen_token_ids=np.zeros((0,), dtype=np.int64),
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                prompt_token_count=prompt_token_count,
            )

        layer_step_hidden = np.zeros(
            (num_layers_plus_embed, num_steps, self.hidden_size), dtype=np.float32
        )
        for step_idx, layers_at_step in enumerate(step_hidden_states):
            for layer_idx, layer_tensor in enumerate(layers_at_step):
                # last token position at this step -> the representation
                # that conditioned the token actually chosen there.
                last_pos = layer_tensor[0, -1, :].detach().to(torch.float32).cpu().numpy()
                layer_step_hidden[layer_idx, step_idx, :] = last_pos

        scores = out.scores or ()  # tuple length T of [1, V] logits
        vocab_size = scores[0].shape[-1] if scores else 0
        step_logits = np.zeros((len(scores), vocab_size), dtype=np.float32)
        for step_idx, logits in enumerate(scores):
            step_logits[step_idx, :] = logits[0].detach().to(torch.float32).cpu().numpy()

        chosen_token_ids = generated_ids.detach().cpu().numpy().astype(np.int64)
        # Guard against off-by-one between scores/hidden_states/chosen ids
        # for models whose generate() emits an extra bookkeeping step.
        n = min(layer_step_hidden.shape[1], step_logits.shape[0], chosen_token_ids.shape[0])

        return GenerationBundle(
            text=text,
            layer_step_hidden=layer_step_hidden[:, :n, :],
            step_logits=step_logits[:n, :],
            chosen_token_ids=chosen_token_ids[:n],
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            prompt_token_count=prompt_token_count,
        )

    # -- teacher-forced scoring (training/eval on labeled responses) -----

    def score_existing_response(
        self, query: str, response: str, system: str | None = None
    ) -> GenerationBundle:
        """Score a *fixed, already-written* response instead of generating
        one, for training/evaluating on labeled (query, response, label)
        pairs (HaluEval, TriviaQA, ...).

        This deliberately does NOT call `model.generate()`. It builds the
        prompt exactly like `generate_with_states`, tokenizes the given
        `response` as-is (no re-generation, no re-sampling), concatenates
        prompt + response into one sequence, and runs a single forward
        pass with `output_hidden_states=True`. Because the whole sequence
        (including the response) is fed in up front, every position's
        hidden state / logit is computed "teacher-forced": conditioned on
        the *actual* preceding tokens (prompt or dataset response), not on
        whatever the model would have produced on its own. That is what
        makes it valid to train against the dataset's label - if we
        instead called `generate_with_states(query)` and threw away its
        text in favor of the labeled `response`, the returned hidden
        states/logits would describe a different (freshly sampled)
        response than the one being labeled, silently decoupling features
        from labels.

        Position bookkeeping: for a causal LM, the hidden state / logit at
        sequence index i is what *produced* (conditioned on and predicted)
        the token at index i+1. So the response's token at position
        `prompt_token_count + j` (j = 0..T-1) is "produced by" the hidden
        state at position `prompt_token_count - 1 + j`. That is the span
        we slice out - i.e. this mirrors `generate_with_states` (which
        also captures the state that conditioned each chosen token), just
        computed in one forward pass over a fixed sequence instead of T
        incremental decode steps.

        Returns a `GenerationBundle` in exactly the same shape/convention
        `generate_with_states` returns (so it is a drop-in input to
        `layer_sampling.build_sequential_features` /
        `build_global_features` and everything downstream), with `text`
        set to the input `response` verbatim (not re-decoded from tokens)
        so the fixed label response is never altered.
        """
        self._ensure_loaded()
        if self._model == "MOCK":
            rng = np.random.RandomState(hash(query + response) % (2**32 - 1))
            num_tokens = max(len(response.split()), 5)
            num_layers = self.num_layers
            hidden_dim = self.hidden_size
            vocab_size = 32000

            layer_step_hidden = rng.normal(0, 1, size=(num_layers + 1, num_tokens, hidden_dim)).astype(np.float32)
            step_logits = rng.normal(0, 1, size=(num_tokens, vocab_size)).astype(np.float32)
            chosen_token_ids = rng.randint(0, vocab_size, size=num_tokens)

            return GenerationBundle(
                text=response,
                layer_step_hidden=layer_step_hidden,
                step_logits=step_logits,
                chosen_token_ids=chosen_token_ids,
                num_layers=num_layers,
                hidden_size=hidden_dim,
                prompt_token_count=10,
            )

        import torch

        tokenizer = self._tokenizer
        model = self._model
        assert tokenizer is not None and model is not None

        system_prompt = system or _CHAT_SYSTEM_FALLBACK
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        try:
            prompt_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors="pt"
            )
        except Exception:
            text_prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"
            prompt_ids = tokenizer(text_prompt, return_tensors="pt").input_ids

        prompt_token_count = int(prompt_ids.shape[1])

        # Tokenize the fixed response AS-IS: no special tokens added, no
        # truncation/rewriting - this is the label, not a new generation.
        response_ids = tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        ).input_ids
        if response_ids.shape[1] > 128:
            response_ids = response_ids[:, :128]
        num_response_tokens = int(response_ids.shape[1])

        if num_response_tokens == 0:
            # Degenerate (empty labeled response) - same empty-bundle
            # convention as generate_with_states.
            return GenerationBundle(
                text=response,
                layer_step_hidden=np.zeros((0, 0, self.hidden_size), dtype=np.float32),
                step_logits=np.zeros((0, 0), dtype=np.float32),
                chosen_token_ids=np.zeros((0,), dtype=np.int64),
                num_layers=self.num_layers,
                hidden_size=self.hidden_size,
                prompt_token_count=prompt_token_count,
            )

        prompt_ids = prompt_ids.to(self.device)
        response_ids = response_ids.to(self.device)
        full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # [1, P+T]

        # The LLM backbone is frozen (only the MultiHaluDet branch on top
        # is trained - see pipeline.py/ensemble.py), so no_grad here is
        # correct and matches generate_with_states.
        with torch.no_grad():
            out = model(full_ids, output_hidden_states=True, use_cache=False)

        # out.hidden_states: tuple length (L+1), each [1, P+T, H] - a
        # single forward pass over the whole sequence, unlike generate()'s
        # per-step tuples.
        hidden_states = out.hidden_states or ()
        num_layers_plus_embed = len(hidden_states)

        start = max(prompt_token_count - 1, 0)
        end = start + num_response_tokens

        layer_step_hidden = np.zeros(
            (num_layers_plus_embed, num_response_tokens, self.hidden_size), dtype=np.float32
        )
        for layer_idx, layer_tensor in enumerate(hidden_states):
            span = layer_tensor[0, start:end, :].detach().to(torch.float32).cpu().numpy()
            layer_step_hidden[layer_idx, : span.shape[0], :] = span

        logits = out.logits  # [1, P+T, V]
        span_logits = logits[0, start:end, :].to(torch.float32)
        log_probs = torch.log_softmax(span_logits, dim=-1)
        probs = log_probs.exp()
        step_entropy = (
            -(probs * log_probs).sum(dim=-1).detach().cpu().numpy().astype(np.float32)
        )
        del span_logits, log_probs, probs

        step_logits = (
            logits[0, start:end, :].detach().to(torch.float32).cpu().numpy().astype(np.float32)
        )

        chosen_token_ids = response_ids[0].detach().cpu().numpy().astype(np.int64)

        del out, logits
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Guard the same off-by-one class of issue generate_with_states
        # guards against (defensive - shapes should already agree here
        # since all three come from slicing the same span/response_ids).
        n = min(layer_step_hidden.shape[1], step_logits.shape[0], chosen_token_ids.shape[0])

        return GenerationBundle(
            text=response,
            layer_step_hidden=layer_step_hidden[:, :n, :],
            step_logits=step_logits[:n, :],
            chosen_token_ids=chosen_token_ids[:n],
            num_layers=self.num_layers,
            hidden_size=self.hidden_size,
            prompt_token_count=prompt_token_count,
            step_entropy=step_entropy[:n],
            query=query,
        )


hf_generation_backend = HFGenerationBackend()
