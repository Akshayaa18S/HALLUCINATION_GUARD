"""
Tests for HFGenerationBackend.score_existing_response - teacher-forced
scoring of a fixed, already-written response.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from multihaludet.tests.conftest import fallback_prompt_text


def test_fixed_response_text_is_preserved_verbatim(fake_backend):
    response = "  Paris is   the capital of France."
    bundle = fake_backend.score_existing_response("What is the capital of France?", response)
    # The label response must come back exactly as given - not
    # re-decoded from tokens (which could silently normalize
    # whitespace/casing/punctuation and desync it from the label).
    assert bundle.text == response


def test_does_not_call_generate(fake_backend):
    """score_existing_response must never call model.generate() - doing
    so would score the model's own freshly-sampled text instead of the
    dataset's fixed labeled response."""

    def _boom(*args, **kwargs):
        raise AssertionError("model.generate() must not be called by score_existing_response")

    fake_backend._model.generate = _boom
    bundle = fake_backend.score_existing_response("query", "a fixed response")
    assert bundle.text == "a fixed response"


def test_hidden_state_and_logit_shapes(fake_backend):
    response = "one two three four"
    bundle = fake_backend.score_existing_response("some query words", response)

    num_response_tokens = len(response.split())
    model = fake_backend._model

    assert bundle.layer_step_hidden.shape == (
        model.num_layers + 1,
        num_response_tokens,
        model.hidden_size,
    )
    assert bundle.step_logits.shape == (num_response_tokens, model.vocab_size)
    assert bundle.chosen_token_ids.shape == (num_response_tokens,)
    assert bundle.num_layers == model.num_layers
    assert bundle.hidden_size == model.hidden_size
    assert bundle.layer_step_hidden.dtype == np.float32
    assert bundle.step_logits.dtype == np.float32
    assert not bundle.is_empty()


def test_teacher_forced_positions_are_correct(fake_backend):
    """Verifies the actual position-slicing logic: response token j
    (0-indexed) must be paired with the hidden state at
    prompt_token_count - 1 + j (the causal-LM convention: the state at
    position i conditions the prediction of the token at i+1)."""
    query = "q1 q2"
    response = "r1 r2 r3"
    bundle = fake_backend.score_existing_response(query, response)

    tokenizer = fake_backend._tokenizer
    model = fake_backend._model

    prompt_ids = tokenizer(fallback_prompt_text(query), return_tensors="pt").input_ids.to(fake_backend.device)
    response_ids = tokenizer(response, return_tensors="pt", add_special_tokens=False).input_ids.to(fake_backend.device)
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)

    prompt_token_count = prompt_ids.shape[1]
    num_response_tokens = response_ids.shape[1]
    start = prompt_token_count - 1
    end = start + num_response_tokens

    expected_base = model.embedding_table[full_ids[0, start:end]]  # [T, H]
    for layer in range(model.num_layers + 1):
        expected = (expected_base * (layer + 1)).detach().cpu().numpy()
        np.testing.assert_allclose(bundle.layer_step_hidden[layer], expected, atol=1e-5)

    assert bundle.chosen_token_ids.tolist() == response_ids[0].tolist()
    assert bundle.prompt_token_count == prompt_token_count


def test_different_responses_yield_different_token_ids(fake_backend):
    b1 = fake_backend.score_existing_response("q", "the sky is blue")
    b2 = fake_backend.score_existing_response("q", "the sky is green")
    assert b1.chosen_token_ids.tolist() != b2.chosen_token_ids.tolist()
    assert b1.text != b2.text


def test_same_response_is_deterministic(fake_backend):
    b1 = fake_backend.score_existing_response("query text", "a stable response")
    b2 = fake_backend.score_existing_response("query text", "a stable response")
    np.testing.assert_array_equal(b1.chosen_token_ids, b2.chosen_token_ids)
    np.testing.assert_allclose(b1.layer_step_hidden, b2.layer_step_hidden)
    np.testing.assert_allclose(b1.step_logits, b2.step_logits)


def test_empty_response_returns_empty_bundle(fake_backend):
    bundle = fake_backend.score_existing_response("query", "")
    assert bundle.is_empty()
    assert bundle.text == ""
    assert bundle.layer_step_hidden.shape[0] == 0 or bundle.layer_step_hidden.size == 0
    assert bundle.step_logits.shape == (0, 0)
    assert bundle.chosen_token_ids.shape == (0,)


@pytest.mark.parametrize("response", ["single", "two words", "three whole words here", ""])
def test_response_length_matches_token_count(fake_backend, response):
    bundle = fake_backend.score_existing_response("q", response)
    expected_len = len(response.split())
    assert bundle.step_logits.shape[0] == expected_len
    assert bundle.chosen_token_ids.shape[0] == expected_len
