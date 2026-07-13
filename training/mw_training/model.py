"""Small SOUL policy networks and affordance masking."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .dataset import FEATURE_DIM, N_NEIGHBORS, N_TOOLS


class PolicyMLP(nn.Module):
    """Two-layer shared MLP with tool and neighbor-pointer heads.

    The twelve affordance bits are appended to the encoded observation before
    entering the network. They are also applied as a hard output mask; the
    duplicate input lets the policy choose intelligently among valid tools
    while the output mask provides the safety guarantee.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM + N_TOOLS,
        hidden_dim: int = 768,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.tool_head = nn.Linear(hidden_dim, N_TOOLS)
        self.target_head = nn.Linear(hidden_dim, N_NEIGHBORS)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(obs)
        return self.tool_head(hidden), self.target_head(hidden)


def _mask_bits(afforded_mask: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Extract bits using ONNX-supported integer arithmetic (not bitwise ops)."""
    if device is None:
        device = afforded_mask.device
    divisors = 2 ** torch.arange(N_TOOLS, device=device)
    return torch.remainder(
        torch.div(afforded_mask.to(device=device).long().unsqueeze(-1), divisors, rounding_mode="floor"),
        2,
    )


def model_input(obs: torch.Tensor, afforded_mask: torch.Tensor) -> torch.Tensor:
    """Append the schema's affordance mask as twelve binary input features."""
    return torch.cat((obs, _mask_bits(afforded_mask, obs.device).to(obs.dtype)), dim=-1)


class MaskedPolicy(nn.Module):
    """Export/inference wrapper that applies the hard action mask."""

    def __init__(self, policy: PolicyMLP):
        super().__init__()
        self.policy = policy

    def forward(self, obs: torch.Tensor, afforded_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tool_logits, target_logits = self.policy(model_input(obs, afforded_mask))
        return masked_logits(tool_logits, afforded_mask), target_logits


@dataclass(frozen=True)
class TransformerConfig:
    """Optional config for a future tokenized observation encoder."""

    d_model: int = 128
    nhead: int = 4
    layers: int = 2
    ff_dim: int = 512

def masked_logits(logits: torch.Tensor, afforded_mask: torch.Tensor) -> torch.Tensor:
    """Set unafforded tool logits to ``-inf`` without changing input tensors."""
    if logits.shape[-1] != N_TOOLS:
        raise ValueError(f"expected {N_TOOLS} tool logits, got {logits.shape[-1]}")
    mask = _mask_bits(afforded_mask, logits.device).bool()
    # A zero mask is a valid cold/LOD observation; leave all logits finite so
    # argmax remains deterministic rather than producing an all-NaN softmax.
    no_tool = ~mask.any(dim=-1, keepdim=True)
    safe_mask = mask | no_tool
    return logits.masked_fill(~safe_mask, float("-inf"))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
