"""
Integration tests: multihaludet/training/train.py and evaluate.py wired
to HFGenerationBackend.score_existing_response and
MultiHaluDetModel.compute_deep_features/predict_from_features, using the
fake backend (no network/GPU/real model download).
"""

from __future__ import annotations

import torch

from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.training import evaluate as evaluate_mod
from multihaludet.training import train as train_mod
from multihaludet.training.datasets import HallucinationExample


def _toy_examples() -> list[HallucinationExample]:
    return [
        HallucinationExample("What is the capital of France?", "Paris is the capital of France.", False),
        HallucinationExample("What is the capital of France?", "The moon is the capital of France.", True),
        HallucinationExample("Who wrote Hamlet?", "William Shakespeare wrote Hamlet.", False),
        HallucinationExample("Who wrote Hamlet?", "Hamlet was written by a robot in 2999.", True),
    ]


def test_score_example_uses_teacher_forced_scoring(fake_backend):
    ex = _toy_examples()[0]
    bundle = train_mod._score_example(fake_backend, ex)
    assert bundle.text == ex.response
    assert not bundle.is_empty()


def test_run_epoch_updates_parameters_and_returns_loss(fake_backend, tiny_model_kwargs):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    examples = _toy_examples()
    cached_bundles = train_mod.precompute_generation_bundles(fake_backend, examples)

    before = {n: p.clone() for n, p in model.named_parameters()}
    avg_loss = train_mod._run_epoch(model, examples, list(range(len(examples))), cached_bundles, optimizer)

    assert isinstance(avg_loss, float)
    assert avg_loss == avg_loss  # not NaN
    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed, "a training epoch over labeled examples produced no parameter updates"


def test_evaluate_examples_returns_expected_metric_schema(fake_backend, tiny_model_kwargs):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    examples = _toy_examples()
    cached_bundles = train_mod.precompute_generation_bundles(fake_backend, examples)
    metrics = train_mod._evaluate_examples(model, examples, list(range(len(examples))), cached_bundles)
    assert {"accuracy", "f1", "auc", "n"}.issubset(metrics.keys())
    assert metrics["n"] == len(_toy_examples())


def test_evaluate_split_uses_forward_and_reports_is_trained(fake_backend, tiny_model_kwargs, tmp_path):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    ckpt_path = tmp_path / "ckpt.pt"
    model.save_checkpoint(str(ckpt_path))

    loaded_model = MultiHaluDetModel(**tiny_model_kwargs)
    assert loaded_model.load_checkpoint(str(ckpt_path)) is True
    assert loaded_model.is_trained is True

    metrics = evaluate_mod.evaluate_split(loaded_model, fake_backend, _toy_examples())
    assert {"accuracy", "f1", "auc", "n"}.issubset(metrics.keys())
    assert metrics["n"] == len(_toy_examples())


def test_evaluate_split_skips_empty_responses(fake_backend, tiny_model_kwargs):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    examples = _toy_examples() + [HallucinationExample("q", "", True)]
    metrics = evaluate_mod.evaluate_split(model, fake_backend, examples)
    # The empty-response example must be skipped, not crash or be
    # silently counted as a scored example.
    assert metrics["n"] == len(_toy_examples())


def test_full_kfold_train_produces_a_loadable_checkpoint(fake_backend, tiny_model_kwargs, tmp_path, monkeypatch):
    """Exercises train.train()'s k-fold loop end-to-end against the fake
    backend, standing in for HFGenerationBackend(model_name=..., ...)."""
    monkeypatch.setattr(train_mod, "HFGenerationBackend", lambda *a, **kw: fake_backend)

    import argparse

    args = argparse.Namespace(
        halueval_qa=None,
        halueval_dialogue=None,
        halueval_summarization=None,
        triviaqa=None,
        french=None,
        bangla=None,
        amharic=None,
        model_name="fake",
        device="cpu",
        folds=2,
        epochs=1,
        lr=0.1,
        seed=0,
        resume_from=None,
        checkpoint_out=str(tmp_path / "multihaludet.pt"),
        allow_reduced_ensemble=True,
    )

    # train.train() calls _collect_examples(args), which needs at least
    # one dataset path; monkeypatch it directly to our in-memory toy set
    # instead of routing through datasets.py's file loaders.
    monkeypatch.setattr(train_mod, "_collect_examples", lambda _args: _toy_examples())

    train_mod.train(args)

    out_path = tmp_path / "multihaludet.pt"
    assert out_path.exists()

    # train.train() constructs MultiHaluDetModel(hidden_size=backend.hidden_size)
    # with default architecture hyperparameters (not tiny_model_kwargs) -
    # match that here so the checkpoint's shapes line up.
    model = MultiHaluDetModel(hidden_size=fake_backend.hidden_size)
    assert model.load_checkpoint(str(out_path)) is True
    assert model.is_trained is True

