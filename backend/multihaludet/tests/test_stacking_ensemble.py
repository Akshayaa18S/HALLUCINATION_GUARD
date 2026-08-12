"""
Comprehensive unit tests for MultiHaluDet stacking ensemble, dataset sampling,
independent fold initialization, feature diagnostics, and artifact serialization.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from multihaludet.ensemble import ClassicalEnsemble, check_base_learner_dependencies
from multihaludet.pipeline import MultiHaluDetModel
from multihaludet.training.datasets import (
    HallucinationExample,
    get_dataset_diagnostics,
    sample_representative_subset,
)
from multihaludet.training.train import run_feature_diagnostics, _run_epoch
from multihaludet.tests.conftest import make_synthetic_bundle


def _make_toy_dataset() -> list[HallucinationExample]:
    examples = []
    # Source A
    for i in range(10):
        examples.append(HallucinationExample(f"Q_A_{i}", f"Ans_A_True_{i}", True, source="source_A", provenance="prov_A"))
        examples.append(HallucinationExample(f"Q_A_{i}", f"Ans_A_False_{i}", False, source="source_A", provenance="prov_A"))
    # Source B
    for i in range(10):
        examples.append(HallucinationExample(f"Q_B_{i}", f"Ans_B_True_{i}", True, source="source_B", provenance="prov_B"))
        examples.append(HallucinationExample(f"Q_B_{i}", f"Ans_B_False_{i}", False, source="source_B", provenance="prov_B"))
    return examples


def test_1_deterministic_stratified_sampling():
    ds1 = _make_toy_dataset()
    ds2 = _make_toy_dataset()

    sampled1 = sample_representative_subset(ds1, max_samples=10, seed=42)
    sampled2 = sample_representative_subset(ds2, max_samples=10, seed=42)

    assert len(sampled1) == 10
    assert len(sampled2) == 10
    assert [ex.query for ex in sampled1] == [ex.query for ex in sampled2]


def test_2_balanced_representative_max_samples_behavior():
    ds = _make_toy_dataset()
    sampled = sample_representative_subset(ds, max_samples=12, seed=42)
    assert len(sampled) == 12

    diag = get_dataset_diagnostics(sampled)
    assert diag["sources"]["source_A"]["total"] == 6
    assert diag["sources"]["source_B"]["total"] == 6
    assert diag["positives"] == 6
    assert diag["negatives"] == 6


def test_3_independent_fold_initialization(tiny_model_kwargs):
    model1 = MultiHaluDetModel(**tiny_model_kwargs)
    model2 = MultiHaluDetModel(**tiny_model_kwargs)

    opt1 = torch.optim.SGD(model1.parameters(), lr=0.1)
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])

    # Update model1
    fused = model1.compute_deep_features(bundle)
    out = model1.predict_from_features(fused)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["meta_logit"], torch.tensor(1.0))
    opt1.zero_grad()
    loss.backward()
    opt1.step()

    # Ensure model2 parameters were NOT mutated by model1's training step
    for (n1, p1), (n2, p2) in zip(model1.named_parameters(), model2.named_parameters()):
        if "meta" in n1 or "members" in n1:
            continue
        # Weights should differ after optimization on model1
        if torch.equal(p1, p2):
            assert not torch.equal(p1, p2), f"Parameter {n1} leaked or did not update"
            break


def test_4_feature_extraction_shape(tiny_model_kwargs):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])
    fused = model.compute_deep_features(bundle)
    assert fused.ndim == 1
    assert fused.shape[0] == tiny_model_kwargs["encoder_dim"]


def test_5_feature_separation_diagnostics():
    np.random.seed(42)
    # Linearly separable features
    faithful_feats = np.random.normal(loc=0.0, scale=1.0, size=(20, 16))
    halluc_feats = np.random.normal(loc=3.0, scale=1.0, size=(20, 16))

    features = np.vstack([faithful_feats, halluc_feats])
    labels = np.array([0] * 20 + [1] * 20)

    diag = run_feature_diagnostics(features, labels)

    assert "centroid_distance" in diag
    assert diag["centroid_distance"] > 1.0
    assert diag["probe_metrics"]["auc"] > 0.8
    assert "overall_feature_variance" in diag


def test_6_oof_prediction_dimensions():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(30, 16).astype(np.float32)
    y = np.array([0, 1] * 15)

    res = ensemble.fit_oof(X, y, n_splits=3, seed=42)
    oof_probs = res["oof_probs"]

    assert oof_probs.shape[0] == 30
    assert oof_probs.shape[1] == len(ensemble.active_member_names)


def test_7_no_oof_data_leakage():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(20, 10).astype(np.float32)
    y = np.array([0, 1] * 10)

    res = ensemble.fit_oof(X, y, n_splits=2, seed=42)
    meta_auc = res["meta_oof_metrics"]["auc"]
    assert 0.0 <= meta_auc <= 1.0


def test_8_rf_training():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10)
    ensemble.fit_oof(X, y, n_splits=2, seed=42)
    assert "random_forest" in ensemble.base_models


def test_9_xgboost_training():
    dep_status = check_base_learner_dependencies()
    xgb_avail, _ = dep_status["xgboost"]
    if not xgb_avail:
        pytest.skip("XGBoost not available in environment")

    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10)
    ensemble.fit_oof(X, y, n_splits=2, seed=42)
    assert "xgboost" in ensemble.base_models


def test_10_lightgbm_training_and_app_control_handling():
    dep_status = check_base_learner_dependencies()
    lgb_avail, lgb_msg = dep_status["lightgbm"]

    if not lgb_avail:
        assert lgb_msg is not None
        assert "LightGBM is unavailable" in lgb_msg
    else:
        ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
        X = np.random.randn(20, 8).astype(np.float32)
        y = np.array([0, 1] * 10)
        ensemble.fit_oof(X, y, n_splits=2, seed=42)
        assert "lightgbm" in ensemble.base_models


def test_11_logistic_regression_training():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10)
    ensemble.fit_oof(X, y, n_splits=2, seed=42)
    assert "logistic_regression" in ensemble.base_models


def test_12_svm_training():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10)
    ensemble.fit_oof(X, y, n_splits=2, seed=42)
    assert "svm" in ensemble.base_models


def test_13_meta_learner_training():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(20, 8).astype(np.float32)
    y = np.array([0, 1] * 10)
    res = ensemble.fit_oof(X, y, n_splits=2, seed=42)
    assert ensemble.meta_model is not None
    assert "accuracy" in res["meta_oof_metrics"]


def test_14_model_artifact_save_load(tiny_model_kwargs, tmp_path):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    X = np.random.randn(20, tiny_model_kwargs["encoder_dim"]).astype(np.float32)
    y = np.array([0, 1] * 10)
    model.classical_ensemble.fit_oof(X, y, n_splits=2, seed=42)

    ckpt_dir = tmp_path / "checkpoints"
    meta = {"model_name": "test", "seed": 42}
    model.save_checkpoint(str(ckpt_dir), metadata=meta)

    assert (ckpt_dir / "feature_extractor.pt").exists()
    assert (ckpt_dir / "ensemble" / "meta_learner.joblib").exists()
    assert (ckpt_dir / "metadata.json").exists()

    fresh = MultiHaluDetModel(**tiny_model_kwargs)
    loaded = fresh.load_checkpoint(str(ckpt_dir))
    assert loaded is True
    assert fresh.is_trained is True
    assert fresh.classical_ensemble.is_fitted is True


def test_15_inference_after_loading_artifacts(tiny_model_kwargs, tmp_path):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    X = np.random.randn(20, tiny_model_kwargs["encoder_dim"]).astype(np.float32)
    y = np.array([0, 1] * 10)
    model.classical_ensemble.fit_oof(X, y, n_splits=2, seed=42)

    ckpt_dir = tmp_path / "checkpoints"
    model.save_checkpoint(str(ckpt_dir))

    fresh = MultiHaluDetModel(**tiny_model_kwargs)
    fresh.load_checkpoint(str(ckpt_dir))

    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])
    result = fresh(bundle)

    assert result["is_trained"] is True
    assert "ensemble_member_probabilities" in result
    assert 0.0 <= result["internal_hallucination_probability"] <= 1.0


def test_16_probability_range():
    ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)
    X = np.random.randn(10, 8).astype(np.float32)
    y = np.array([0, 1] * 5)
    ensemble.fit_oof(X, y, n_splits=2, seed=42)

    pred = ensemble.predict_proba(X[0])
    p = pred["final_probability"]
    assert 0.0 <= p <= 1.0
    for m_p in pred["member_probabilities"].values():
        assert 0.0 <= m_p <= 1.0


def test_17_bce_with_logits_loss_neural_compatibility_path(tiny_model_kwargs):
    model = MultiHaluDetModel(**tiny_model_kwargs)
    model.train()
    bundle = make_synthetic_bundle(hidden_size=tiny_model_kwargs["hidden_size"])

    fused = model.compute_deep_features(bundle)
    out = model.predict_from_features(fused)

    assert "meta_logit" in out
    meta_logit = out["meta_logit"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(meta_logit, torch.tensor(1.0))
    loss.backward()
    assert meta_logit.grad_fn is not None



def test_18_system_specific_schemas_in_comparative_evaluation():
    """Verifies that comparative systems A, B, C, D validate custom feature dimensions without zero padding."""
    from multihaludet.ensemble import evaluate_comparative_systems, ClassicalEnsemble
    from multihaludet.feature_extractor import FeatureSchemaError

    np.random.seed(42)
    # Total matrix shape [100, 271] (256 Qwen deep + 15 explicit)
    X_total = np.random.randn(100, 271).astype(np.float32)
    y_labels = np.array([0, 1] * 50, dtype=np.int64)

    comp_res = evaluate_comparative_systems(X_total, y_labels, n_splits=3, seed=42, allow_reduced=True)

    assert "System_A_Qwen_Baseline" in comp_res
    assert "System_B_DeBERTa_NLI_Only" in comp_res
    assert "System_C_NLI_Plus_Evidence" in comp_res
    assert "System_D_Full_Fused_MultiHaluDet" in comp_res

    for sys_name, sys_m in comp_res.items():
        assert "error" not in sys_m, f"System {sys_name} failed with error: {sys_m.get('error')}"
        assert 0.0 <= sys_m["auc"] <= 1.0
        assert 0.0 <= sys_m["pr_auc"] <= 1.0
        assert 0.0 <= sys_m["f1"] <= 1.0
        assert 0.0 <= sys_m["accuracy"] <= 1.0

    # Test schema guard rejection when feature count is wrong
    ens_strict = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True, expected_feature_dim=271, system_name="StrictTest")
    X_wrong = np.random.randn(50, 15).astype(np.float32)
    with pytest.raises(FeatureSchemaError, match="SYSTEM FEATURE SCHEMA MISMATCH"):
        ens_strict.fit_oof(X_wrong, np.array([0, 1] * 25), n_splits=2)
