"""
Stage 7 of the MultiHaluDet branch: out-of-fold deep-feature generation
+ a learned ensemble meta-learner over several base-learner heads,
which the paper lists as one of its four major stages.

Implementation note (read before trusting scores from this module):
the base paper's ensemble members are classical models (random forest,
XGBoost, LightGBM, logistic regression, SVM) trained out-of-fold on deep
features extracted from a *trained* backbone. Reproducing that training
loop requires labeled hallucination data (HaluEval / TriviaQA per the
paper) and is implemented in multihaludet/training/ - it is NOT run
automatically by this module. Here, each "base learner" is a small,
differently-initialized linear head over the same fused deep-feature
vector (architecturally analogous to a stacking ensemble's base layer,
not literally RF/XGBoost/LightGBM), and the meta-learner is a logistic
regression-style linear layer over their outputs. Until
`load_checkpoint()` is pointed at real trained weights
(multihaludet_checkpoint_path in config/settings.py), every score this
produces reflects an UNTRAINED, randomly-initialized network - useful
for validating the architecture end-to-end, not for real hallucination
judgments. `MultiHaluDetModel.is_trained` reports which case you're in,
and services/pipeline_service.py surfaces it in stage metadata rather
than hiding it.
"""

from __future__ import annotations

import torch
from torch import nn

# Member names kept as the paper's reference algorithm family for
# readability in API responses / explainability output. See the module
# docstring: these are structurally-analogous heads, not literal
# sklearn/XGBoost/LightGBM models.
DEFAULT_MEMBER_NAMES = [
    "random_forest",
    "xgboost",
    "lightgbm",
    "logistic_regression",
    "svm",
]


class BaseLearnerHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, seed: int):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )
        with torch.no_grad():
            for p in self.net.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p, generator=g)
                else:
                    nn.init.zeros_(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # logit


class EnsembleMetaLearner(nn.Module):
    def __init__(
        self,
        in_dim: int,
        num_members: int = 5,
        member_hidden_dim: int = 32,
        member_names: list[str] | None = None,
    ):
        super().__init__()
        self.member_names = (member_names or DEFAULT_MEMBER_NAMES)[:num_members]
        while len(self.member_names) < num_members:
            self.member_names.append(f"member_{len(self.member_names)}")
        self.members = nn.ModuleList(
            [
                BaseLearnerHead(in_dim, member_hidden_dim, seed=1000 + i)
                for i in range(num_members)
            ]
        )
        # Meta-learner: logistic regression over the base-member logits.
        self.meta = nn.Linear(num_members, 1)
        with torch.no_grad():
            self.meta.weight.fill_(1.0 / num_members)
            self.meta.bias.zero_()

    def forward(self, deep_features: torch.Tensor) -> dict[str, torch.Tensor]:
        member_logits = torch.stack([m(deep_features) for m in self.members])  # [num_members]
        member_probs = torch.sigmoid(member_logits)
        meta_logit = self.meta(member_logits.unsqueeze(0)).squeeze()
        final_prob = torch.sigmoid(meta_logit)
        return {
            "member_logits": member_logits,
            "member_probs": member_probs,
            "meta_logit": meta_logit,
            "final_probability": final_prob,
        }


def check_base_learner_dependencies() -> dict[str, tuple[bool, str | None]]:
    """Checks availability of optional/external ML dependencies (XGBoost, LightGBM).

    Returns a dict mapping algorithm name to (is_available, failure_message).
    """
    status: dict[str, tuple[bool, str | None]] = {
        "random_forest": (True, None),
        "logistic_regression": (True, None),
        "svm": (True, None),
    }

    # Check XGBoost
    try:
        import xgboost  # noqa: F401
        status["xgboost"] = (True, None)
    except Exception as exc:
        msg = (
            f"XGBoost is unavailable ({exc}). "
            "Please install it using: python -m pip install xgboost"
        )
        status["xgboost"] = (False, msg)

    # Check LightGBM
    try:
        import lightgbm  # noqa: F401
        status["lightgbm"] = (True, None)
    except Exception as exc:
        msg = (
            f"LightGBM is unavailable ({exc}). "
            "Please install it or check Windows Application Control policies using: python -m pip install lightgbm"
        )
        status["lightgbm"] = (False, msg)

    return status


class ClassicalEnsemble:
    """True base-paper stacking ensemble over fused MultiHaluDet deep features.

    Contains 5 base learners:
    1. RandomForestClassifier
    2. XGBoost XGBClassifier
    3. LightGBM LGBMClassifier
    4. sklearn LogisticRegression
    5. SVM using SVC(probability=True)

    Stacked using a LogisticRegression meta-classifier trained on Out-Of-Fold
    (OOF) predictions.
    """

    MEMBER_NAMES = DEFAULT_MEMBER_NAMES

    def __init__(self, seed: int = 42, allow_reduced_ensemble: bool = False):
        self.seed = seed
        self.allow_reduced_ensemble = allow_reduced_ensemble
        self.dep_status = check_base_learner_dependencies()
        self.active_member_names: list[str] = []

        self.is_fitted = False
        self.is_complete_ensemble = False
        self.mode = "unfitted"

        self.base_models: dict[str, Any] = {}
        self.meta_model: Any = None

    def _init_base_models(self) -> dict[str, Any]:
        from typing import Any
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        models: dict[str, Any] = {}
        missing: list[str] = []

        # 1. Random Forest
        models["random_forest"] = RandomForestClassifier(n_estimators=100, random_state=self.seed)

        # 2. XGBoost
        xgb_avail, xgb_err = self.dep_status["xgboost"]
        if xgb_avail:
            try:
                from xgboost import XGBClassifier
                models["xgboost"] = XGBClassifier(
                    eval_metric="logloss", random_state=self.seed, n_estimators=100
                )
            except Exception as exc:
                xgb_avail = False
                xgb_err = str(exc)

        if not xgb_avail:
            missing.append(f"xgboost: {xgb_err}")

        # 3. LightGBM
        lgb_avail, lgb_err = self.dep_status["lightgbm"]
        if lgb_avail:
            try:
                from lightgbm import LGBMClassifier
                models["lightgbm"] = LGBMClassifier(
                    random_state=self.seed, verbose=-1, n_estimators=100
                )
            except Exception as exc:
                lgb_avail = False
                lgb_err = str(exc)

        if not lgb_avail:
            missing.append(f"lightgbm: {lgb_err}")

        # 4. Logistic Regression
        models["logistic_regression"] = LogisticRegression(max_iter=1000, random_state=self.seed)

        # 5. SVM
        models["svm"] = SVC(probability=True, random_state=self.seed)

        if missing and not self.allow_reduced_ensemble:
            err_msg = (
                "Cannot initialize full 5-model base ensemble due to missing/blocked dependencies:\n"
                + "\n".join(missing)
                + "\n\nStandard production training requires all 5 algorithms. Use python -m pip install <package> "
                "or run with allow_reduced_ensemble=True for explicitly labeled development/test mode."
            )
            raise RuntimeError(err_msg)

        return models

    def fit_oof(
        self,
        X: Any,
        y: Any,
        n_splits: int = 5,
        seed: int = 42,
    ) -> dict[str, Any]:
        """Performs true Out-Of-Fold (OOF) stacking:

        1. Splits training data into K folds.
        2. Fits each base model on K-1 folds, predicts probabilities on validation fold.
        3. Assembles OOF probability matrix OOF_X shape [num_examples, num_active_members].
        4. Trains LogisticRegression meta-classifier on (OOF_X, y).
        5. Retrains all base models on full (X, y).
        """
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.int64)

        if len(X_arr) < n_splits:
            raise ValueError(f"Number of samples ({len(X_arr)}) must be >= n_splits ({n_splits})")

        self.scaler = StandardScaler()
        X_arr = self.scaler.fit_transform(X_arr)

        self.base_models = self._init_base_models()
        self.active_member_names = [name for name in self.MEMBER_NAMES if name in self.base_models]
        num_members = len(self.active_member_names)

        self.is_complete_ensemble = (num_members == len(self.MEMBER_NAMES))
        self.mode = "complete_production_ensemble" if self.is_complete_ensemble else "reduced_dev_mode"

        oof_probs = np.zeros((len(X_arr), num_members), dtype=np.float32)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

        # 1. Generate OOF predictions
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_arr, y_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va, y_va = X_arr[val_idx], y_arr[val_idx]

            for m_idx, name in enumerate(self.active_member_names):
                # Instantiate fresh fold copy of the model
                fold_model = self._create_fresh_model(name)
                fold_model.fit(X_tr, y_tr)

                if hasattr(fold_model, "predict_proba"):
                    probs = fold_model.predict_proba(X_va)
                    if probs.ndim == 2 and probs.shape[1] > 1:
                        oof_probs[val_idx, m_idx] = probs[:, 1]
                    else:
                        oof_probs[val_idx, m_idx] = probs.ravel()
                else:
                    oof_probs[val_idx, m_idx] = fold_model.predict(X_va)

        # 2. Fit meta-learner on OOF predictions
        self.meta_model = LogisticRegression(max_iter=1000, random_state=seed)
        self.meta_model.fit(oof_probs, y_arr)

        # Evaluate OOF Meta learner
        meta_oof_probs = self.meta_model.predict_proba(oof_probs)[:, 1]
        meta_oof_preds = (meta_oof_probs >= 0.5).astype(int)

        oof_metrics = {
            "accuracy": float(accuracy_score(y_arr, meta_oof_preds)),
            "precision": float(precision_score(y_arr, meta_oof_preds, zero_division=0)),
            "recall": float(recall_score(y_arr, meta_oof_preds, zero_division=0)),
            "f1": float(f1_score(y_arr, meta_oof_preds, zero_division=0)),
            "auc": float(roc_auc_score(y_arr, meta_oof_probs)) if len(set(y_arr)) > 1 else 0.5,
        }

        # Evaluate base models OOF
        base_oof_metrics: dict[str, dict[str, float]] = {}
        for m_idx, name in enumerate(self.active_member_names):
            m_probs = oof_probs[:, m_idx]
            m_preds = (m_probs >= 0.5).astype(int)
            base_oof_metrics[name] = {
                "accuracy": float(accuracy_score(y_arr, m_preds)),
                "precision": float(precision_score(y_arr, m_preds, zero_division=0)),
                "recall": float(recall_score(y_arr, m_preds, zero_division=0)),
                "f1": float(f1_score(y_arr, m_preds, zero_division=0)),
                "auc": float(roc_auc_score(y_arr, m_probs)) if len(set(y_arr)) > 1 else 0.5,
            }

        # 3. Retrain base models on full data
        for name in self.active_member_names:
            full_model = self._create_fresh_model(name)
            full_model.fit(X_arr, y_arr)
            self.base_models[name] = full_model

        self.is_fitted = True
        return {
            "meta_oof_metrics": oof_metrics,
            "base_oof_metrics": base_oof_metrics,
            "oof_probs": oof_probs,
            "is_complete_ensemble": self.is_complete_ensemble,
            "mode": self.mode,
        }

    def _create_fresh_model(self, name: str) -> Any:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        if name == "random_forest":
            return RandomForestClassifier(n_estimators=100, random_state=self.seed)
        elif name == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(eval_metric="logloss", random_state=self.seed, n_estimators=100)
        elif name == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(random_state=self.seed, verbose=-1, n_estimators=100)
        elif name == "logistic_regression":
            return LogisticRegression(max_iter=1000, random_state=self.seed)
        elif name == "svm":
            return SVC(probability=True, random_state=self.seed)
        else:
            raise ValueError(f"Unknown base learner name: {name}")

    def predict_proba(self, X: Any) -> dict[str, Any]:
        """Predicts probabilities using base models + meta learner."""
        import numpy as np

        if not self.is_fitted:
            raise RuntimeError("ClassicalEnsemble is not fitted yet. Call fit_oof() or load() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        single_input = (X_arr.ndim == 1)
        if single_input:
            X_arr = X_arr.reshape(1, -1)

        if hasattr(self, "scaler") and self.scaler is not None:
            X_arr = self.scaler.transform(X_arr)

        num_samples = len(X_arr)
        num_members = len(self.active_member_names)
        member_matrix = np.zeros((num_samples, num_members), dtype=np.float32)

        member_probs_dict: dict[str, list[float]] = {}
        for m_idx, name in enumerate(self.active_member_names):
            model = self.base_models[name]
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_arr)
                if probs.ndim == 2 and probs.shape[1] > 1:
                    p = probs[:, 1]
                else:
                    p = probs.ravel()
            else:
                p = model.predict(X_arr).astype(np.float32)

            member_matrix[:, m_idx] = p
            member_probs_dict[name] = [float(v) for v in p]

        meta_probs = self.meta_model.predict_proba(member_matrix)[:, 1]

        if single_input:
            return {
                "member_probabilities": {name: member_probs_dict[name][0] for name in self.active_member_names},
                "final_probability": float(meta_probs[0]),
                "is_complete_ensemble": self.is_complete_ensemble,
                "mode": self.mode,
            }

        return {
            "member_probabilities": member_probs_dict,
            "final_probability": [float(v) for v in meta_probs],
            "is_complete_ensemble": self.is_complete_ensemble,
            "mode": self.mode,
        }

    def save(self, ensemble_dir: str | Any) -> None:
        """Saves base models and meta-learner into joblib files inside ensemble_dir."""
        from pathlib import Path
        import joblib

        d = Path(ensemble_dir)
        d.mkdir(parents=True, exist_ok=True)

        for name, model in self.base_models.items():
            joblib.dump(model, d / f"{name}.joblib")

        if self.meta_model is not None:
            joblib.dump(self.meta_model, d / "meta_learner.joblib")

        if hasattr(self, "scaler") and self.scaler is not None:
            joblib.dump(self.scaler, d / "scaler.joblib")

        meta_info = {
            "seed": self.seed,
            "active_member_names": self.active_member_names,
            "is_fitted": self.is_fitted,
            "is_complete_ensemble": self.is_complete_ensemble,
            "mode": self.mode,
        }
        joblib.dump(meta_info, d / "ensemble_info.joblib")

    def load(self, ensemble_dir: str | Any) -> bool:
        """Loads base models and meta-learner from joblib files inside ensemble_dir."""
        from pathlib import Path
        import joblib

        d = Path(ensemble_dir)
        if not d.exists() or not d.is_dir():
            return False

        meta_info_path = d / "ensemble_info.joblib"
        if not meta_info_path.exists():
            return False

        try:
            meta_info = joblib.load(meta_info_path)
            self.seed = meta_info.get("seed", 42)
            self.active_member_names = meta_info.get("active_member_names", [])
            self.is_complete_ensemble = meta_info.get("is_complete_ensemble", False)
            self.mode = meta_info.get("mode", "unknown")

            self.base_models = {}
            for name in self.active_member_names:
                m_path = d / f"{name}.joblib"
                if not m_path.exists():
                    return False
                self.base_models[name] = joblib.load(m_path)

            meta_path = d / "meta_learner.joblib"
            if not meta_path.exists():
                return False
            self.meta_model = joblib.load(meta_path)

            scaler_path = d / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                self.scaler = None

            self.is_fitted = True
            return True
        except Exception as exc:  # pragma: no cover
            self.is_fitted = False
            return False

