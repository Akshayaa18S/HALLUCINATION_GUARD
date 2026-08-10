"""
Stages 3-5 of the MultiHaluDet branch, operating on the sequential
representation built in layer_sampling.py:

  MultiScaleAttention          - per selected layer, pool the
                                  generation-step trajectory at several
                                  temporal resolutions and combine them
                                  with learned position-wise gating.
  LayerWeightedTransformer     - treats the (now depth-ordered) per-layer
                                  pooled vectors as a sequence, gives each
                                  layer a learnable importance weight,
                                  and runs a small Transformer encoder
                                  over them.
  SelfAttentionPooling         - collapses the encoder's per-layer output
                                  sequence into one vector via learned
                                  attention (instead of a plain mean).

All torch. Input shape convention: [S, T, H] (S = selected layers,
T = generation steps, H = model hidden size), unbatched (this branch
processes one generation at a time - batching can be added later by
adding a leading batch dim throughout if needed).
"""

from __future__ import annotations

import torch
from torch import nn


class MultiScaleAttention(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, scales: list[int]):
        super().__init__()
        self.scales = scales
        self.input_proj = nn.Linear(in_dim, proj_dim)
        # One attention-pooling head per scale (shared across layers).
        self.scale_pool_scores = nn.ModuleList(
            [nn.Linear(proj_dim, 1) for _ in scales]
        )
        # Position-wise gate over the concatenated per-scale vectors,
        # producing one softmax weight per scale.
        self.gate = nn.Sequential(
            nn.Linear(proj_dim * len(scales), len(scales)),
        )

    def _pool_at_scale(self, x: torch.Tensor, scale: int, score_layer: nn.Linear) -> torch.Tensor:
        """x: [S, T, P] -> attention-pooled [S, P] after avg-pooling the
        time axis into windows of size `scale`."""
        S, T, P = x.shape
        if T == 0:
            return torch.zeros(S, P, device=x.device, dtype=x.dtype)
        pad = (-T) % scale
        if pad:
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
        windows = x.shape[1] // scale
        if windows == 0:
            pooled_seq = x.mean(dim=1, keepdim=True)  # [S, 1, P]
        else:
            pooled_seq = x[:, : windows * scale, :].view(S, windows, scale, P).mean(dim=2)
        scores = score_layer(pooled_seq).squeeze(-1)  # [S, windows]
        weights = torch.softmax(scores, dim=-1)
        return torch.einsum("sw,swp->sp", weights, pooled_seq)

    def forward(self, layer_trajectories: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """layer_trajectories: [S, T, H] -> (fused [S, proj_dim], scale_gate_weights [len(scales)])"""
        x = self.input_proj(layer_trajectories)  # [S, T, P]
        per_scale = [
            self._pool_at_scale(x, scale, self.scale_pool_scores[i])
            for i, scale in enumerate(self.scales)
        ]  # list of [S, P]
        stacked = torch.stack(per_scale, dim=1)  # [S, num_scales, P]
        concat = stacked.flatten(start_dim=1)  # [S, num_scales * P]
        gate_logits = self.gate(concat)  # [S, num_scales]
        gate_weights = torch.softmax(gate_logits, dim=-1)  # [S, num_scales]
        fused = torch.einsum("sn,snp->sp", gate_weights, stacked)  # [S, P]
        # Mean gate across layers, purely for explainability reporting.
        mean_gate = gate_weights.mean(dim=0)
        return fused, mean_gate


class LayerWeightedTransformerEncoder(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_layers: int, max_layers: int = 64):
        super().__init__()
        self.layer_importance = nn.Parameter(torch.zeros(max_layers))
        self.layer_pos_embedding = nn.Parameter(torch.randn(max_layers, dim) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads, batch_first=True, dim_feedforward=dim * 2
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, layer_vectors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """layer_vectors: [S, P] -> (encoded [S, P], layer_weights [S])"""
        S = layer_vectors.shape[0]
        importance = torch.softmax(self.layer_importance[:S], dim=0)  # [S]
        weighted = layer_vectors * importance.unsqueeze(-1) * S  # rescale so mean magnitude is stable
        weighted = weighted + self.layer_pos_embedding[:S]
        encoded = self.encoder(weighted.unsqueeze(0)).squeeze(0)  # [S, P]
        return encoded, importance


class SelfAttentionPooling(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [S, P] -> (pooled [P], attention_weights [S])"""
        scores = self.score(x).squeeze(-1)  # [S]
        weights = torch.softmax(scores, dim=0)
        pooled = torch.einsum("s,sp->p", weights, x)
        return pooled, weights
