"""2-layer MLP adapter mapping perceiver-resampler output to LLM embedding dim.

One instance per modality. Matches PETRG-3D's "Adapter" block: a small MLP
between the per-modality perceiver sampler and the LLM input-embedding
space. Independent weights for CT and PET allow each modality to develop
its own projection without cross-talk.
"""

from __future__ import annotations

import torch
from torch import nn


class VisionProjection(nn.Module):
    """``Linear(in_dim, hidden_dim) → GELU → Linear(hidden_dim, out_dim)``.

    If ``hidden_dim`` is omitted, defaults to ``out_dim`` (so the MLP looks
    like ``in → out → out`` -- this matches the LLaVA / PETRG convention of
    an "MLP-2" projector).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim if hidden_dim is not None else out_dim

        self.fc1 = nn.Linear(in_dim, self.hidden_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.fc2 = nn.Linear(self.hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, N, in_dim) -> (B, N, out_dim)``."""
        return self.fc2(self.drop(self.act(self.fc1(x))))

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, hidden_dim={self.hidden_dim}, "
            f"out_dim={self.out_dim}"
        )
