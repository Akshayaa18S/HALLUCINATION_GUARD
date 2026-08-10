"""
Stage 6 of the MultiHaluDet branch: the paper's "global branch" (an MLP
over the global feature vector: top-k probs, prob diffs, entropy/logit
stats, layer-norm trajectory stats, anchor descriptors, interaction
features) plus a gated fusion with the sequential/attention branch's
pooled output, producing the deep-feature vector the ensemble
meta-learner classifies.
"""

from __future__ import annotations

import torch
from torch import nn


class GlobalBranch(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, global_features: torch.Tensor) -> torch.Tensor:
        return self.net(global_features)


class GatedFusion(nn.Module):
    """Projects the sequential-branch pooled vector and the global-branch
    vector to a shared fused dimension, and combines them with a learned,
    per-dimension gate (rather than a fixed concat or average), so the
    model can lean on whichever branch is more informative per example."""

    def __init__(self, seq_dim: int, global_dim: int, fused_dim: int):
        super().__init__()
        self.seq_proj = nn.Linear(seq_dim, fused_dim)
        self.global_proj = nn.Linear(global_dim, fused_dim)
        self.gate = nn.Sequential(
            nn.Linear(seq_dim + global_dim, fused_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(fused_dim)

    def forward(self, seq_vector: torch.Tensor, global_vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        seq_p = self.seq_proj(seq_vector)
        global_p = self.global_proj(global_vector)
        gate = self.gate(torch.cat([seq_vector, global_vector], dim=-1))
        fused = self.norm(gate * seq_p + (1.0 - gate) * global_p)
        return fused, gate
