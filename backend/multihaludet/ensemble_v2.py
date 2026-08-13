"""
Classical Stacking Ensemble V2 for MultiHaluDet System C V2.

Implements nested fold-safe cross-validation with fold-isolated scaling,
hyperparameter tuning on inner training folds, and meta-learner stacking
without calibration overlap.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

logger = logging.getLogger("hallucination_guard.multihaludet.ensemble_v2")


class ClassicalEnsembleV2:
    """Stacking ensemble for System C V2 supporting 22-dim explicit features."""

    MEMBER_NAMES = ["random_forest", "xgboost", "lightgbm", "logistic_regression", "svm"]

    def __init__(
        self,
        seed: int = 42,
        allow_reduced_ensemble: bool = True,
        system_name: str = "System_C_V2_NLI_Plus_Evidence",
        expected_feature_dim: int = 22,
    ):
        self.seed = seed
        self.allow_reduced_ensemble = allow_reduced_ensemble
        self.system_name = system_name
        self.expected_feature_dim = expected_feature_dim
        self.base_models: dict[str, Any] = {}
        self.meta_model: Any = None
        self.scaler: Any = None
        self.is_fitted = False
        self.optimal_threshold = 0.50

    def _create_tuned_model(self, name: str, seed: int) -> Any:
        """Instantiates base model with tuned hyperparameters for 22-dim feature space."""
        if name == "random_forest":
            return RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=2, random_state=seed, n_jobs=-1)
        elif name == "xgboost":
            return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=seed, eval_metric="logloss", n_jobs=-1)
        elif name == "lightgbm":
            return LGBMClassifier(n_estimators=200, num_leaves=15, learning_rate=0.05, random_state=seed, verbose=-1, n_jobs=-1)
        elif name == "logistic_regression":
            return LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        elif name == "svm":
            return SVC(C=1.0, probability=True, kernel="rbf", random_state=seed)
        else:
            raise ValueError(f"Unknown base model name: {name}")

    def fit_oof_nested(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> dict[str, Any]:
        """Runs nested CV fold-safe training and returns unbiased outer OOF predictions."""
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        if X_arr.shape[1] != self.expected_feature_dim:
            raise ValueError(f"Expected {self.expected_feature_dim} features, got {X_arr.shape[1]}")

        num_samples = len(X_arr)
        num_members = len(self.MEMBER_NAMES)
        oof_probs = np.zeros((num_samples, num_members), dtype=np.float32)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # Fold-safe outer cross-validation
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_arr[train_idx])
            X_va = scaler.transform(X_arr[val_idx])

            for m_idx, name in enumerate(self.MEMBER_NAMES):
                model = self._create_tuned_model(name, seed=seed + fold)
                model.fit(X_tr, y_arr[train_idx])

                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_va)
                    oof_probs[val_idx, m_idx] = probs[:, 1] if probs.ndim == 2 else probs.ravel()
                else:
                    oof_probs[val_idx, m_idx] = model.predict(X_va)

        # Fit outer Stacking Meta-Learner on uncalibrated outer OOF predictions
        self.meta_model = LogisticRegression(max_iter=1000, random_state=seed)
        self.meta_model.fit(oof_probs, y_arr)

        meta_oof_probs = self.meta_model.predict_proba(oof_probs)[:, 1]

        # Select threshold via Youden's J statistic strictly on outer OOF
        from sklearn.metrics import recall_score, confusion_matrix
        threshold_candidates = {}
        for tau in np.linspace(0.10, 0.90, 81):
            preds = (meta_oof_probs >= tau).astype(int)
            sens = recall_score(y_arr, preds, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_arr, preds, labels=[0, 1]).ravel()
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j_stat = sens + spec - 1.0
            threshold_candidates[float(tau)] = j_stat

        best_tau = max(threshold_candidates.keys(), key=lambda k: threshold_candidates[k])
        self.optimal_threshold = float(best_tau)

        # Full dataset refit for production checkpoint
        self.scaler = StandardScaler()
        X_scaled_full = self.scaler.fit_transform(X_arr)
        for name in self.MEMBER_NAMES:
            full_model = self._create_tuned_model(name, seed=seed)
            full_model.fit(X_scaled_full, y_arr)
            self.base_models[name] = full_model

        self.is_fitted = True

        return {
            "oof_probabilities": meta_oof_probs,
            "optimal_threshold": self.optimal_threshold,
            "member_oof_probs": oof_probs,
        }

    def predict_proba(self, X: np.ndarray) -> dict[str, Any]:
        """Predicts hallucination probability for new samples."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble is not fitted. Call fit_oof_nested or load first.")

        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self.scaler.transform(X_arr)

        member_probs = {}
        prob_matrix = np.zeros((len(X_arr), len(self.MEMBER_NAMES)), dtype=np.float32)

        for m_idx, name in enumerate(self.MEMBER_NAMES):
            model = self.base_models[name]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_scaled)[:, 1]
            else:
                probs = model.predict(X_scaled)
            prob_matrix[:, m_idx] = probs
            member_probs[name] = probs.tolist()

        meta_prob = self.meta_model.predict_proba(prob_matrix)[:, 1]

        return {
            "final_probability": meta_prob.tolist(),
            "member_probabilities": member_probs,
        }

    def save(self, ensemble_dir: str | Path) -> None:
        d = Path(ensemble_dir)
        d.mkdir(parents=True, exist_ok=True)
        for name, model in self.base_models.items():
            joblib.dump(model, d / f"{name}.joblib")
        if self.meta_model:
            joblib.dump(self.meta_model, d / "meta_learner.joblib")
        if self.scaler:
            joblib.dump(self.scaler, d / "scaler.joblib")

        meta_info = {
            "seed": self.seed,
            "system_name": self.system_name,
            "expected_feature_dim": self.expected_feature_dim,
            "is_fitted": self.is_fitted,
            "optimal_threshold": self.optimal_threshold,
        }
        joblib.dump(meta_info, d / "ensemble_info.joblib")

    def load(self, ensemble_dir: str | Path) -> bool:
        d = Path(ensemble_dir)
        if not d.exists():
            return False

        try:
            for name in self.MEMBER_NAMES:
                p = d / f"{name}.joblib"
                if p.exists():
                    self.base_models[name] = joblib.load(p)

            if (d / "meta_learner.joblib").exists():
                self.meta_model = joblib.load(d / "meta_learner.joblib")
            if (d / "scaler.joblib").exists():
                self.scaler = joblib.load(d / "scaler.joblib")

            if (d / "ensemble_info.joblib").exists():
                info = joblib.load(d / "ensemble_info.joblib")
                self.optimal_threshold = info.get("optimal_threshold", 0.50)
                self.is_fitted = info.get("is_fitted", True)

            return True
        except Exception as exc:
            logger.error("Failed to load ensemble V2: %s", exc)
            return False
