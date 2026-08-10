"""
Stage 2 of the MultiHaluDet branch: turn a raw GenerationBundle (all
layers x all generation steps) into

  1. a *sequential representation*: a small, evenly-spaced-by-depth
     subset of layers (early/middle/late), each a [T, H] trajectory
     across generation steps - this is what multi-scale attention and
     the layer-weighted Transformer encoder operate on; and

  2. a *global feature vector*: scalar statistics over the whole
     generation (confidence, entropy, layer-norm trajectory, agreement
     between shallow and deep representations) that don't need the
     Transformer at all - this is the paper's "global branch" input.

Pure numpy - no torch dependency here, so this stage is unit-testable
without the heavy model loaded.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multihaludet.generation_backend import GenerationBundle


def select_layers(num_layers_plus_embed: int, num_samples: int) -> list[int]:
    """Evenly-spaced layer indices across depth, always including the
    embedding output (index 0, "shallowest") and the final layer
    ("deepest"). This is the paper's "retain early, middle and late
    network-stage information" requirement, made deterministic so the
    same architecture is probed the same way every call."""
    n = num_layers_plus_embed
    k = max(2, min(num_samples, n))
    if k >= n:
        return list(range(n))
    # linspace over [0, n-1] inclusive, deduplicated, always keeps the
    # endpoints (shallowest embedding + deepest transformer layer).
    idx = np.linspace(0, n - 1, num=k)
    return sorted({int(round(i)) for i in idx})


@dataclass
class SequentialFeatures:
    selected_layers: list[int]
    # [num_selected_layers, T, H] float32
    layer_trajectories: np.ndarray


@dataclass
class GlobalFeatures:
    names: list[str]
    values: np.ndarray  # float32 [len(names)]


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.clip(np.sum(e, axis=axis, keepdims=True), 1e-9, None)


def build_sequential_features(
    bundle: GenerationBundle, num_sampled_layers: int
) -> SequentialFeatures:
    total_layers = bundle.layer_step_hidden.shape[0]
    if total_layers == 0:
        return SequentialFeatures(selected_layers=[], layer_trajectories=np.zeros((0, 0, bundle.hidden_size), dtype=np.float32))
    layers = select_layers(total_layers, num_sampled_layers)
    trajectories = bundle.layer_step_hidden[layers, :, :]  # [S, T, H]
    return SequentialFeatures(selected_layers=layers, layer_trajectories=trajectories.astype(np.float32))


def build_global_features(
    bundle: GenerationBundle, top_k: int = 5, num_sampled_layers: int = 6
) -> GlobalFeatures:
    T = bundle.step_logits.shape[0]
    names: list[str] = []
    values: list[float] = []

    if T == 0:
        # Degenerate (empty generation) - return a zero vector with the
        # same schema so downstream shapes stay fixed.
        names = _global_feature_names(top_k)
        return GlobalFeatures(names=names, values=np.zeros(len(names), dtype=np.float32))

    step_logits = bundle.step_logits.astype(np.float32)

    # Standardize step_logits to top-64 logits if full vocab is present,
    # matching precomputed training bundle dimensions.
    if step_logits.ndim == 2 and step_logits.shape[1] > 64:
        k = 64
        topk_idx = np.argpartition(step_logits, -k, axis=-1)[:, -k:]
        rows = np.arange(step_logits.shape[0])[:, None]
        sorted_sub_idx = np.argsort(-step_logits[rows, topk_idx], axis=-1)
        step_logits = step_logits[rows, topk_idx[rows, sorted_sub_idx]]
    layer_step_hidden = bundle.layer_step_hidden.astype(np.float32)

    total_layers = layer_step_hidden.shape[0]
    if total_layers > num_sampled_layers:
        sampled_layers = select_layers(total_layers, num_sampled_layers)
        layer_step_hidden = layer_step_hidden[sampled_layers, :, :]

    probs = _softmax(step_logits, axis=-1)  # [T, V]
    sorted_probs = np.sort(probs, axis=-1)[:, ::-1]  # descending, [T, V]
    top_k = min(top_k, sorted_probs.shape[-1])
    topk_probs = sorted_probs[:, :top_k]  # [T, k]

    # 1. top-k token probabilities (mean over steps, per rank)
    topk_mean = topk_probs.mean(axis=0)  # [k]
    for rank in range(top_k):
        names.append(f"top{rank+1}_prob_mean")
        values.append(float(topk_mean[rank]))

    # 2. probability differences (confidence margin: top1 - top2, per step)
    margin = topk_probs[:, 0] - (topk_probs[:, 1] if top_k > 1 else 0.0)
    names += ["margin_mean", "margin_min", "margin_std"]
    values += [float(margin.mean()), float(margin.min()), float(margin.std())]

    # 3. entropy / logit statistics
    if hasattr(bundle, "step_entropy") and bundle.step_entropy is not None and len(bundle.step_entropy) == T:
        entropy = bundle.step_entropy.astype(np.float32)
    else:
        entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, None)), axis=-1)  # [T]
    names += ["entropy_mean", "entropy_max", "entropy_std"]
    values += [float(entropy.mean()), float(entropy.max()), float(entropy.std())]

    logits_flat = step_logits
    names += ["logit_mean", "logit_std", "logit_max"]
    values += [float(logits_flat.mean()), float(logits_flat.std()), float(logits_flat.max())]

    # 4. layer-norm trajectory statistics: ||h_l|| for the FINAL generated
    # token's representation, across depth - captures whether the model's
    # internal representation is still "settling" by the last layer
    # (associated in probing literature with less-grounded generations).
    final_step = layer_step_hidden[:, -1, :]  # [L+1, H]
    layer_norms = np.linalg.norm(final_step, axis=-1)  # [L+1]
    if layer_norms.shape[0] >= 2:
        depth_axis = np.arange(layer_norms.shape[0])
        slope = float(np.polyfit(depth_axis, layer_norms, 1)[0])
    else:
        slope = 0.0
    names += ["layer_norm_mean", "layer_norm_std", "layer_norm_final", "layer_norm_depth_slope"]
    values += [
        float(layer_norms.mean()),
        float(layer_norms.std()),
        float(layer_norms[-1]),
        slope,
    ]

    # 5. anchor descriptors: cosine similarity between the shallowest
    # (embedding) and deepest (final layer) representation of the final
    # token - a large drift can indicate the model "changed its mind"
    # about the token deep in the network.
    shallow = final_step[0]
    deep = final_step[-1]
    denom = (np.linalg.norm(shallow) * np.linalg.norm(deep)) or 1e-9
    cos_sim = float(np.dot(shallow, deep) / denom)
    names.append("anchor_shallow_deep_cosine")
    values.append(cos_sim)

    # 6. interaction features (cheap, paper-flavored: confidence x
    # entropy, confidence x drift, response length interactions).
    names += ["interaction_margin_entropy", "interaction_margin_cosine", "response_length_norm"]
    values += [
        float(margin.mean() * entropy.mean()),
        float(margin.mean() * cos_sim),
        float(np.log1p(T)),
    ]

    return GlobalFeatures(names=names, values=np.array(values, dtype=np.float32))


def _global_feature_names(top_k: int) -> list[str]:
    names = [f"top{r+1}_prob_mean" for r in range(top_k)]
    names += ["margin_mean", "margin_min", "margin_std"]
    names += ["entropy_mean", "entropy_max", "entropy_std"]
    names += ["logit_mean", "logit_std", "logit_max"]
    names += ["layer_norm_mean", "layer_norm_std", "layer_norm_final", "layer_norm_depth_slope"]
    names.append("anchor_shallow_deep_cosine")
    names += ["interaction_margin_entropy", "interaction_margin_cosine", "response_length_norm"]
    return names


GLOBAL_FEATURE_DIM_FOR = lambda top_k: len(_global_feature_names(top_k))  # noqa: E731
