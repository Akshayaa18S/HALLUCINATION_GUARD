"""
Tests for the pipeline.py training path added alongside
score_existing_response: MultiHaluDetModel.compute_deep_features /
predict_from_features (gradient flow) and checkpoint save/load
(is_trained semantics).
"""

from __future__ import annotations

import torch

from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.tests.conftest import make_synthetic_bundle


def _new_model(tiny_model_kwargs) -> MultiHaluDetModel:
    return MultiHaluDetModel(**tiny_model_kwargs)


def test_model_starts_untrained(tiny_model_kwargs):
    model = _new_model(tiny_model_kwargs)
    assert model.is_trained is False
    assert model.checkpoint_path is None


def test_forward_reports_is_trained_flag(tiny_model_kwargs):
    model = _new_model(tiny_model_kwargs)
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])
    result = model(bundle)
    assert result["is_trained"] is False
    assert result["checkpoint_path"] is None
    assert 0.0 <= result["internal_hallucination_probability"] <= 1.0


def test_forward_does_not_require_grad_context(tiny_model_kwargs):
    """forward() must be safe to call with no ambient no_grad (it wraps
    itself), so it works from ordinary inference call sites."""
    model = _new_model(tiny_model_kwargs)
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])
    result = model(bundle)
    assert isinstance(result["deep_features"], list)


def test_compute_deep_features_matches_forward_deep_features(tiny_model_kwargs):
    """The training path (compute_deep_features) and the inference path
    (forward) must compute the exact same fused vector for the same
    input - they share the same underlying submodules."""
    model = _new_model(tiny_model_kwargs)
    model.eval()  # disable dropout in the ensemble base-learner heads
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])

    with torch.no_grad():
        fused = model.compute_deep_features(bundle)
    result = model(bundle)

    torch.testing.assert_close(fused, torch.tensor(result["deep_features"]), atol=1e-4, rtol=1e-4)


def test_gradients_reach_all_trainable_components(tiny_model_kwargs):
    """The whole point of compute_deep_features/predict_from_features:
    a loss computed from them must backprop into every trainable branch
    (multi-scale attention, layer-weighted encoder, self-attention
    pooling, global branch, gated fusion, ensemble), not just the last
    layer."""
    model = _new_model(tiny_model_kwargs)
    model.train()
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])

    fused = model.compute_deep_features(bundle)
    assert fused.requires_grad

    out = model.predict_from_features(fused)
    target = torch.tensor(1.0)
    loss = torch.nn.functional.binary_cross_entropy(out["final_probability"], target)
    loss.backward()

    trainable_top_level_modules = [
        "multi_scale_attention",
        "layer_weighted_encoder",
        "self_attention_pooling",
        "global_branch",
        "gated_fusion",
        "ensemble",
    ]
    for module_name in trainable_top_level_modules:
        submodule = getattr(model, module_name)
        grads = [p.grad for p in submodule.parameters() if p.requires_grad]
        assert grads, f"{module_name} has no trainable parameters"
        assert any(g is not None and torch.any(g != 0) for g in grads), (
            f"no nonzero gradient reached {module_name}"
        )


def test_compute_deep_features_rejects_empty_bundle(tiny_model_kwargs):
    model = _new_model(tiny_model_kwargs)
    empty_bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"], num_steps=0)
    try:
        model.compute_deep_features(empty_bundle)
        raised = False
    except ValueError:
        raised = True
    assert raised, "compute_deep_features should reject an empty GenerationBundle"


def test_optimizer_step_updates_trainable_parameters(tiny_model_kwargs):
    """End-to-end mini training step, mirroring train.py's inner loop."""
    model = _new_model(tiny_model_kwargs)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])

    before = {n: p.clone() for n, p in model.named_parameters()}

    fused = model.compute_deep_features(bundle)
    out = model.predict_from_features(fused)
    loss = torch.nn.functional.binary_cross_entropy(out["final_probability"], torch.tensor(1.0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    changed = [n for n, p in model.named_parameters() if not torch.equal(before[n], p)]
    assert changed, "optimizer.step() did not change any parameters"


def test_checkpoint_save_and_load_roundtrip(tiny_model_kwargs, tmp_path):
    model = _new_model(tiny_model_kwargs)
    # Nudge weights away from init so the roundtrip is a meaningful check.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(0.01)

    ckpt_path = tmp_path / "multihaludet.pt"
    model.save_checkpoint(str(ckpt_path))
    assert ckpt_path.exists()

    fresh = _new_model(tiny_model_kwargs)
    loaded = fresh.load_checkpoint(str(ckpt_path))

    assert loaded is True
    for (n1, p1), (n2, p2) in zip(model.state_dict().items(), fresh.state_dict().items()):
        assert n1 == n2
        torch.testing.assert_close(p1, p2)


def test_is_trained_false_when_no_checkpoint_path_given(tiny_model_kwargs):
    model = _new_model(tiny_model_kwargs)
    loaded = model.load_checkpoint(None)
    assert loaded is False
    assert model.is_trained is False
    assert model.checkpoint_path is None


def test_is_trained_false_when_checkpoint_path_missing(tiny_model_kwargs, tmp_path):
    model = _new_model(tiny_model_kwargs)
    missing_path = tmp_path / "does_not_exist.pt"
    loaded = model.load_checkpoint(str(missing_path))
    assert loaded is False
    assert model.is_trained is False
    assert model.checkpoint_path is None


def test_is_trained_true_only_after_loading_valid_checkpoint(tiny_model_kwargs, tmp_path):
    model = _new_model(tiny_model_kwargs)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])

    # Taking gradient steps by itself must NOT flip is_trained - only
    # load_checkpoint() succeeding should (see pipeline.py's comment on
    # the field).
    fused = model.compute_deep_features(bundle)
    out = model.predict_from_features(fused)
    loss = torch.nn.functional.binary_cross_entropy(out["final_probability"], torch.tensor(1.0))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    assert model.is_trained is False

    ckpt_path = tmp_path / "trained.pt"
    model.save_checkpoint(str(ckpt_path))

    fresh = _new_model(tiny_model_kwargs)
    assert fresh.is_trained is False
    fresh.load_checkpoint(str(ckpt_path))
    assert fresh.is_trained is True
    assert fresh.checkpoint_path == str(ckpt_path)


def test_is_trained_false_after_failed_load(tiny_model_kwargs, tmp_path):
    corrupt_path = tmp_path / "corrupt.pt"
    corrupt_path.write_bytes(b"not a real checkpoint")
    model = _new_model(tiny_model_kwargs)
    loaded = model.load_checkpoint(str(corrupt_path))
    assert loaded is False
    assert model.is_trained is False
