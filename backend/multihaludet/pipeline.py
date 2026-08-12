"""
Assembles the full MultiHaluDet forward pass:

    GenerationBundle
        -> layer_sampling.build_sequential_features / build_global_features
        -> MultiScaleAttention
        -> LayerWeightedTransformerEncoder
        -> SelfAttentionPooling
        -> GlobalBranch (on global features)
        -> GatedFusion
        -> EnsembleMetaLearner
        -> internal hallucination probability + explainability metadata
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from multihaludet.attention import LayerWeightedTransformerEncoder, MultiScaleAttention, SelfAttentionPooling
from multihaludet.ensemble import ClassicalEnsemble, EnsembleMetaLearner
from multihaludet.fusion import GatedFusion, GlobalBranch
from multihaludet.generation_backend import GenerationBundle
from multihaludet.layer_sampling import GLOBAL_FEATURE_DIM_FOR, build_global_features, build_sequential_features

logger = logging.getLogger("hallucination_guard.multihaludet.pipeline")


class MultiHaluDetModel(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_sampled_layers: int = 6,
        attention_scales: list[int] | None = None,
        encoder_dim: int = 256,
        encoder_heads: int = 4,
        encoder_layers: int = 2,
        global_top_k: int = 5,
        global_hidden_dim: int = 64,
        ensemble_members: int = 5,
        decision_threshold: float = 0.20,
    ):
        super().__init__()
        scales = attention_scales or [1, 2, 4]
        self.hidden_size = hidden_size
        self.num_sampled_layers = num_sampled_layers
        self.global_top_k = global_top_k
        self.encoder_dim = encoder_dim
        self.decision_threshold = decision_threshold

        self.multi_scale_attention = MultiScaleAttention(hidden_size, encoder_dim, scales)
        self.layer_weighted_encoder = LayerWeightedTransformerEncoder(
            encoder_dim, encoder_heads, encoder_layers, max_layers=max(num_sampled_layers, 64)
        )
        self.self_attention_pooling = SelfAttentionPooling(encoder_dim)

        global_in_dim = GLOBAL_FEATURE_DIM_FOR(global_top_k)
        self.global_branch = GlobalBranch(global_in_dim, global_hidden_dim)
        self.gated_fusion = GatedFusion(encoder_dim, global_hidden_dim, fused_dim=encoder_dim)

        self.ensemble = EnsembleMetaLearner(encoder_dim, num_members=ensemble_members)
        self.classical_ensemble = ClassicalEnsemble(seed=42, allow_reduced_ensemble=True)

        self.is_trained: bool = False
        self.is_complete_ensemble: bool = False
        self.checkpoint_path: str | None = None
        self.metadata: dict[str, Any] = {}

    def forward(self, bundle: GenerationBundle) -> dict[str, Any]:
        """Inference entry point (used by the live API / evaluate.py).
        Wraps the differentiable core in no_grad, since callers here only
        want scores, not gradients. Training uses `compute_deep_features`
        / `predict_from_features` directly (see those methods) to keep
        the autograd graph alive instead."""
        with torch.no_grad():
            return self._forward_result(bundle)

    def _forward_result(self, bundle: GenerationBundle) -> dict[str, Any]:
        if bundle.is_empty():
            return self._neutral_result(reason="empty_generation")

        fused, extra = self._compute_fused(bundle)
        
        # Use classical ensemble if trained & loaded, otherwise fallback to neural head
        if self.classical_ensemble.is_fitted:
            from multihaludet.feature_extractor import ExplicitFeatureExtractor, FeatureSchemaError, EXPECTED_TOTAL_FEATURE_DIM, verify_feature_dim
            extractor = ExplicitFeatureExtractor()
            fused_np = fused.detach().cpu().numpy()
            if fused_np.ndim == 1:
                fused_np = fused_np.reshape(1, -1)
            fused_norm = np.linalg.norm(fused_np, ord=2, axis=-1, keepdims=True)
            fused_norm = np.where(fused_norm > 1e-8, fused_norm, 1.0)
            fused_np = fused_np / fused_norm
            explicit_vec = extractor.extract_feature_vector(getattr(bundle, "query", "") or "", bundle.text or "").reshape(1, -1)

            is_test_mode = getattr(self.classical_ensemble, "allow_reduced_ensemble", False)
            scaler = getattr(self.classical_ensemble, "scaler", None)

            if scaler is not None and hasattr(scaler, "n_features_in_"):
                expected_scaler_dim = scaler.n_features_in_
                if not is_test_mode and expected_scaler_dim != EXPECTED_TOTAL_FEATURE_DIM:
                    raise FeatureSchemaError(
                        f"Loaded scaler expects {expected_scaler_dim} features, but canonical schema v3.2 requires {EXPECTED_TOTAL_FEATURE_DIM}. "
                        "Legacy/dual feature checkpoints are strictly forbidden in publication runs."
                    )


                if is_test_mode and expected_scaler_dim == fused_np.shape[1]:
                    combined_features = fused_np
                else:
                    combined_features = np.concatenate([fused_np, explicit_vec], axis=-1)
            else:
                combined_features = np.concatenate([fused_np, explicit_vec], axis=-1)

            if not is_test_mode:
                verify_feature_dim(combined_features.shape[1], context="MultiHaluDetModel forward pass")

            ensemble_res = self.classical_ensemble.predict_proba(combined_features)
            member_probs_dict = ensemble_res["member_probabilities"]
            final_raw = ensemble_res["final_probability"]
            if isinstance(final_raw, (list, tuple, np.ndarray)):
                final_probability = float(final_raw[0])
            else:
                final_probability = float(final_raw)
            member_names = list(member_probs_dict.keys())
            member_probs = [float(p[0] if isinstance(p, (list, tuple, np.ndarray)) else p) for p in member_probs_dict.values()]
            is_complete = bool(ensemble_res.get("is_complete_ensemble", False))
            mode = str(ensemble_res.get("mode", "classical_ensemble"))
        else:
            ensemble_out = self.ensemble(fused)
            final_probability = float(ensemble_out["final_probability"].item())
            member_probs = ensemble_out["member_probs"].tolist()
            member_names = self.ensemble.member_names
            member_probs_dict = {name: float(p) for name, p in zip(member_names, member_probs)}
            is_complete = False
            mode = "neural_head_untrained" if not self.is_trained else "neural_head"

        seq_features = extra["seq_features"]
        global_features = extra["global_features"]
        scale_gate = extra["scale_gate"]
        layer_importance = extra["layer_importance"]
        pooling_weights = extra["pooling_weights"]
        gate = extra["gate"]

        member_arr = np.array(member_probs, dtype=np.float32)
        agreement = 1.0 - float(member_arr.std()) if len(member_arr) > 0 else 1.0
        decisiveness = float(abs(final_probability - 0.5) * 2.0)
        confidence = max(0.0, min(1.0, 0.5 * agreement + 0.5 * decisiveness))

        return {
            "internal_hallucination_probability": round(final_probability, 4),
            "internal_confidence": round(confidence, 4),
            "is_hallucination": final_probability >= self.decision_threshold,
            "decision_threshold": self.decision_threshold,
            "is_trained": self.is_trained,
            "is_complete_ensemble": is_complete,
            "ensemble_mode": mode,
            "checkpoint_path": self.checkpoint_path,
            "selected_layers": seq_features.selected_layers,
            "num_total_layers": int(bundle.layer_step_hidden.shape[0]),
            "generated_tokens": int(bundle.step_logits.shape[0]),
            "ensemble_member_names": member_names,
            "ensemble_member_probabilities": {
                name: round(float(p[0] if isinstance(p, (list, tuple, np.ndarray)) else p), 4)
                for name, p in member_probs_dict.items()
            },
            "meta_learner_probability": round(final_probability, 4),
            "layer_importance_weights": [round(float(w), 4) for w in layer_importance.tolist()],
            "self_attention_pooling_weights": [round(float(w), 4) for w in pooling_weights.tolist()],
            "multi_scale_gate_weights": [round(float(w), 4) for w in scale_gate.tolist()],
            "global_branch_gate_mean": round(float(gate.mean().item()), 4),
            "global_feature_names": global_features.names,
            "global_feature_values": [round(float(v), 4) for v in global_features.values.tolist()],
            "deep_features": fused.tolist(),
            "metadata": self.metadata,
        }

    def _compute_fused(self, bundle: GenerationBundle) -> tuple[torch.Tensor, dict[str, Any]]:
        seq_features = build_sequential_features(bundle, self.num_sampled_layers)
        global_features = build_global_features(bundle, self.global_top_k, self.num_sampled_layers)

        # Ensure tensors are created on the same device as model parameters
        dev = next(self.parameters()).device if list(self.parameters()) else torch.device("cpu")
        layer_traj = torch.from_numpy(seq_features.layer_trajectories).to(dev)  # [S, T, H]
        global_vec = torch.from_numpy(global_features.values).to(dev)  # [G]

        multiscale_out, scale_gate = self.multi_scale_attention(layer_traj)  # [S, P], [num_scales]
        encoded, layer_importance = self.layer_weighted_encoder(multiscale_out)  # [S, P], [S]
        pooled_seq, pooling_weights = self.self_attention_pooling(encoded)  # [P], [S]

        global_out = self.global_branch(global_vec.unsqueeze(0)).squeeze(0)  # [global_hidden]
        fused, gate = self.gated_fusion(pooled_seq, global_out)  # [P]

        return fused, {
            "seq_features": seq_features,
            "global_features": global_features,
            "scale_gate": scale_gate,

            "layer_importance": layer_importance,
            "pooling_weights": pooling_weights,
            "gate": gate,
        }

    # -- training entry points (gradients flow; caller owns optimizer) ---

    def compute_deep_features(self, bundle: GenerationBundle) -> torch.Tensor:
        if bundle.is_empty():
            raise ValueError(
                "compute_deep_features() requires a non-empty GenerationBundle "
                "(bundle.is_empty() was True) - check the (query, response) "
                "pair produced tokens before training on it."
            )
        fused, _ = self._compute_fused(bundle)
        return fused

    def predict_from_features(self, fused: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.ensemble(fused)

    def _neutral_result(self, reason: str) -> dict[str, Any]:
        return {
            "internal_hallucination_probability": 0.5,
            "internal_confidence": 0.0,
            "is_trained": self.is_trained,
            "is_complete_ensemble": False,
            "ensemble_mode": "untrained",
            "checkpoint_path": self.checkpoint_path,
            "selected_layers": [],
            "num_total_layers": 0,
            "generated_tokens": 0,
            "ensemble_member_names": self.ensemble.member_names,
            "ensemble_member_probabilities": {name: 0.5 for name in self.ensemble.member_names},
            "meta_learner_probability": 0.5,
            "layer_importance_weights": [],
            "self_attention_pooling_weights": [],
            "multi_scale_gate_weights": [],
            "global_branch_gate_mean": 0.0,
            "global_feature_names": [],
            "global_feature_values": [],
            "deep_features": [],
            "metadata": self.metadata,
            "note": reason,
        }

    # -- checkpointing -----------------------------------------------------

    def save_checkpoint(self, path: str, metadata: dict[str, Any] | None = None) -> None:
        """Saves PyTorch feature extractor weights, classical ensemble models,
        and metadata.json to a checkpoint directory or file."""
        import json

        p = Path(path)
        if p.suffix == ".pt":
            # Save single PyTorch file for backwards compatibility
            p.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": self.state_dict()}, p)
            # Also save classical ensemble and metadata in same parent directory if present
            if self.classical_ensemble.is_fitted:
                self.classical_ensemble.save(p.parent / "ensemble")
            if metadata:
                self.metadata = metadata
                with (p.parent / "metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
        else:
            # Save to checkpoint directory structure
            p.mkdir(parents=True, exist_ok=True)
            torch.save({"state_dict": self.state_dict()}, p / "feature_extractor.pt")

            if self.classical_ensemble.is_fitted:
                self.classical_ensemble.save(p / "ensemble")

            if metadata:
                self.metadata = metadata
                with (p / "metadata.json").open("w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

    def load_checkpoint(self, path: str | None) -> bool:
        """Returns True iff a checkpoint was actually loaded."""
        import json

        if not path or not Path(path).exists():
            logger.info("No MultiHaluDet checkpoint at %s", path)
            self.is_trained = False
            self.is_complete_ensemble = False
            self.checkpoint_path = None
            return False

        p = Path(path)
        try:
            if p.is_dir():
                fe_path = p / "feature_extractor.pt"
                if not fe_path.exists():
                    logger.warning("Directory %s does not contain feature_extractor.pt", path)
                    self.is_trained = False
                    return False

                ckpt = torch.load(fe_path, map_location="cpu", weights_only=False)
                self.load_state_dict(ckpt["state_dict"], strict=False)

                ens_dir = p / "ensemble"
                if ens_dir.exists():
                    self.classical_ensemble.load(ens_dir)

                meta_path = p / "metadata.json"
                if meta_path.exists():
                    with meta_path.open(encoding="utf-8") as f:
                        self.metadata = json.load(f)

                self.is_trained = True
                self.is_complete_ensemble = self.classical_ensemble.is_complete_ensemble
                self.checkpoint_path = str(path)
                logger.info("Loaded MultiHaluDet complete checkpoint directory from %s", path)
                return True
            else:
                ckpt = torch.load(p, map_location="cpu")
                ckpt_state = ckpt.get("state_dict", {})

                # Adapt input projection matrix if hidden_size differs between checkpoints
                w_key = "multi_scale_attention.input_proj.weight"
                if w_key in ckpt_state:
                    ckpt_in_dim = ckpt_state[w_key].shape[1]
                    model_in_dim = self.multi_scale_attention.input_proj.weight.shape[1]
                    if ckpt_in_dim != model_in_dim:
                        logger.warning(
                            "Adapting checkpoint feature projection layer (ckpt input_dim=%d, model input_dim=%d)",
                            ckpt_in_dim, model_in_dim
                        )
                        old_w = ckpt_state[w_key]
                        if model_in_dim > ckpt_in_dim:
                            new_w = torch.zeros((old_w.shape[0], model_in_dim), dtype=old_w.dtype)
                            new_w[:, :ckpt_in_dim] = old_w
                            ckpt_state[w_key] = new_w
                        else:
                            ckpt_state[w_key] = old_w[:, :model_in_dim]

                self.load_state_dict(ckpt_state, strict=False)

                # Check if sibling ensemble/ directory exists
                ens_dir = p.parent / "ensemble"
                if ens_dir.exists():
                    self.classical_ensemble.load(ens_dir)

                meta_path = p.parent / "metadata.json"
                if meta_path.exists():
                    with meta_path.open(encoding="utf-8") as f:
                        self.metadata = json.load(f)

                self.is_trained = True
                self.is_complete_ensemble = getattr(self.classical_ensemble, "is_complete_ensemble", True)
                self.checkpoint_path = str(path)
                logger.info("Loaded MultiHaluDet checkpoint file from %s", path)
                return True
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("Failed to load MultiHaluDet checkpoint %s: %s", path, exc)
            self.is_trained = False
            self.is_complete_ensemble = False
            self.checkpoint_path = None
            return False

