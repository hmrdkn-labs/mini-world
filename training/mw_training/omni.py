"""Manifest-conditioned OMNI SOUL policy and behavior-cloning dataset.

OMNI does not allocate a fixed action-classification head.  A shared observation
trunk produces one context vector; each manifest descriptor row is projected by
the descriptor trunk and paired with that context to produce one score.  Thus
adding/reordering tools changes only the descriptor tensor (and affordance
mask), not model parameters.  The target pointer head addresses the eight
observed-neighbor slots and the parameter head emits a small continuous vector.

ONNX contract (dynamic batch and tool count): ``obs`` is ``[B, 129]``,
``tool_descriptors`` is ``[B, T, D]``, ``afforded`` is ``[B, T]`` float mask;
outputs are ``tool_scores [B,T]``, ``target_logits [B,8]``, and
``params [B,P]``.  Scores for unafforded rows are ``-inf``.  Descriptor rows
are data, not learned output classes, so the same export serves any manifest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from .dataset import (
    EXPERTISE_DIM,
    FEATURE_DIM,
    IGNORE_INDEX,
    N_NEIGHBORS,
    TOOL_NAMES,
    compute_norm_stats,
    encode_record,
    record_expertise,
)

DEFAULT_DESCRIPTOR_DIM = 16
DEFAULT_PARAM_DIM = 4


def _stable_unit(text: str) -> float:
    """Map descriptor text to a deterministic feature without Python hash randomization."""
    digest = hashlib.blake2s(text.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "little") / 2**32


def descriptor_rows(
    manifest: Sequence[Mapping[str, object] | str],
    descriptor_dim: int = DEFAULT_DESCRIPTOR_DIM,
) -> torch.Tensor:
    """Encode a manifest into deterministic descriptor rows ``[T, D]``.

    A string is shorthand for a tool name.  Mapping descriptors may provide
    ``name`` and ``args`` where each arg has ``kind``; unknown fields are
    intentionally ignored so scenario packs can evolve independently.
    """
    if descriptor_dim < 8:
        raise ValueError("descriptor_dim must be at least 8")
    rows: list[list[float]] = []
    for index, raw in enumerate(manifest):
        item: Mapping[str, object] = {"name": raw} if isinstance(raw, str) else raw
        name = str(item.get("name", f"tool_{index}"))
        args = item.get("args", ())
        args = args if isinstance(args, Sequence) and not isinstance(args, (str, bytes)) else ()
        kinds = [str(a.get("kind", "")) for a in args if isinstance(a, Mapping)]
        row = [
            index / max(len(manifest) - 1, 1),
            len(args) / 8.0,
            sum(k in {"entity", "entity_ref", "pointer"} for k in kinds) / 8.0,
            sum(k in {"scalar", "number", "float", "int"} for k in kinds) / 8.0,
            sum(k in {"enum", "string", "item"} for k in kinds) / 8.0,
            _stable_unit(name),
            _stable_unit(name.lower()),
            float("move" in name.lower()),
            float("speak" in name.lower() or "talk" in name.lower()),
            float("idle" in name.lower() or "wait" in name.lower()),
            float("target" in name.lower() or any(k in {"entity", "entity_ref", "pointer"} for k in kinds)),
            float("param" in name.lower() or bool(kinds)),
            (index % 7) / 7.0,
            (index % 11) / 11.0,
            _stable_unit("args:" + ",".join(kinds)),
            1.0,
        ]
        rows.append(row[:descriptor_dim] + [0.0] * max(0, descriptor_dim - len(row)))
    if not rows:
        raise ValueError("manifest must contain at least one tool")
    return torch.tensor(rows, dtype=torch.float32)


class OmniPolicy(nn.Module):
    """Observation, expertise, and descriptor-conditioned per-tool policy."""

    def __init__(
        self,
        obs_dim: int = FEATURE_DIM,
        descriptor_dim: int = DEFAULT_DESCRIPTOR_DIM,
        hidden_dim: int = 96,
        target_slots: int = N_NEIGHBORS,
        param_dim: int = DEFAULT_PARAM_DIM,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.descriptor_dim = descriptor_dim
        self.hidden_dim = hidden_dim
        self.target_slots = target_slots
        self.param_dim = param_dim
        self.obs_trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
        )
        self.expertise_trunk = nn.Sequential(
            nn.Linear(EXPERTISE_DIM, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.descriptor_trunk = nn.Sequential(
            nn.Linear(descriptor_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.score_head = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        self.target_head = nn.Linear(hidden_dim, target_slots)
        self.param_head = nn.Linear(hidden_dim, param_dim)

    def forward(
        self,
        obs: torch.Tensor,
        tool_descriptors: torch.Tensor,
        afforded: torch.Tensor,
        expertise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tool_descriptors.ndim != 3 or afforded.ndim != 2:
            raise ValueError("tool_descriptors must be [B,T,D] and afforded [B,T]")
        if expertise is None:
            expertise = torch.zeros(
                (obs.shape[0], EXPERTISE_DIM),
                dtype=obs.dtype,
                device=obs.device,
            )
            expertise[:, 1] = 1.0
        if expertise.ndim != 2 or expertise.shape != (obs.shape[0], EXPERTISE_DIM):
            raise ValueError(f"expertise must be [B,{EXPERTISE_DIM}]")
        context = self.obs_trunk(obs) + self.expertise_trunk(expertise)
        descriptors = self.descriptor_trunk(tool_descriptors)
        context_rows = context.unsqueeze(1).expand(-1, descriptors.shape[1], -1)
        scores = self.score_head(torch.cat((context_rows, descriptors), dim=-1)).squeeze(-1)
        mask = afforded.to(dtype=torch.bool)
        # Keep an all-zero row finite for cold observations; argmax then remains deterministic.
        safe_mask = mask | ~mask.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~safe_mask, float("-inf"))
        return scores, self.target_head(context), self.param_head(context)


@dataclass
class OmniDataset(Dataset):
    """Tensor dataset for manifest-conditioned teacher actions."""

    obs: torch.Tensor
    tool_descriptors: torch.Tensor
    afforded: torch.Tensor
    expertise: torch.Tensor
    tool: torch.Tensor
    target: torch.Tensor
    params: torch.Tensor

    def __len__(self) -> int:
        return int(self.tool.shape[0])

    def __getitem__(self, index: int):
        return tuple(
            x[index]
            for x in (
                self.obs,
                self.tool_descriptors,
                self.afforded,
                self.expertise,
                self.tool,
                self.target,
                self.params,
            )
        )

    @classmethod
    def from_trajectory(
        cls,
        records: Iterable[Mapping],
        manifest: Sequence[Mapping[str, object] | str],
        norm=None,
    ) -> "OmniDataset":
        base = list(records)
        if not base:
            raise ValueError("no trajectory records")
        names = [str(x.get("name", x)) if isinstance(x, Mapping) else str(x) for x in manifest]
        expertise = torch.tensor(np.stack([record_expertise(record) for record in base]), dtype=torch.float32)
        name_to_id = {name: i for i, name in enumerate(names)}
        encoded = [encode_record(record) for record in base]
        raw_obs = np.stack([row[0] for row in encoded])
        if norm is None:
            norm = compute_norm_stats(raw_obs)
        obs = torch.from_numpy(((raw_obs - norm.mean) / norm.std).astype(np.float32, copy=False))
        masks = torch.tensor([row[1] for row in encoded], dtype=torch.long)
        tools = torch.tensor([name_to_id[str(record["decision"]["tool"])] for record in base], dtype=torch.long)
        targets = torch.tensor([row[3] for row in encoded], dtype=torch.long)
        params = torch.tensor(
            [_parameter_vector(record["decision"].get("params", {})) for record in base],
            dtype=torch.float32,
        )
        desc = descriptor_rows(manifest)
        return cls(
            obs,
            desc.unsqueeze(0).expand(len(base), -1, -1).clone(),
            _mask_rows(masks, len(desc)),
            expertise,
            tools,
            targets,
            params,
        )

    @classmethod
    def from_fixtures(
        cls,
        path: str | Path,
        manifest: Sequence[Mapping[str, object] | str] = TOOL_NAMES,
    ) -> "OmniDataset":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        records = payload.get("records", payload) if isinstance(payload, Mapping) else payload
        if not records:
            raise ValueError("fixture file has no records")
        desc = descriptor_rows(manifest)
        names = [str(x.get("name", x)) if isinstance(x, Mapping) else str(x) for x in manifest]
        rows, masks, expertise, tools, targets = [], [], [], [], []
        for record in records:
            logits = np.asarray(record["tool_logits"], dtype=np.float32)
            mask = int(record["mask"])
            available = [i for i in range(min(len(logits), len(names))) if mask & (1 << i)]
            tool = max(available, key=lambda i: float(logits[i])) if available else len(names) - 1
            target_logits = np.asarray(record.get("target_logits", [0.0] * N_NEIGHBORS))
            present = int(record.get("present", 0))
            target = int(target_logits.argmax()) if present else IGNORE_INDEX
            rows.append(record.get("features", record["raw_features"]))
            masks.append(mask)
            expertise.append(record_expertise(record))
            tools.append(tool)
            targets.append(target)
        obs = torch.tensor(np.asarray(rows, dtype=np.float32))
        return cls(
            obs,
            desc.unsqueeze(0).expand(len(rows), -1, -1).clone(),
            _mask_rows(torch.tensor(masks), len(desc)),
            torch.tensor(np.stack(expertise), dtype=torch.float32),
            torch.tensor(tools),
            torch.tensor(targets),
            torch.zeros((len(rows), DEFAULT_PARAM_DIM)),
        )


def _mask_rows(masks: torch.Tensor, tool_count: int) -> torch.Tensor:
    shifts = torch.arange(tool_count, dtype=torch.long)
    return torch.remainder(
        torch.div(masks.long().unsqueeze(-1), 2**shifts, rounding_mode="floor"),
        2,
    ).float()

def _parameter_vector(raw: object) -> list[float]:
    if not isinstance(raw, Mapping):
        return [0.0] * DEFAULT_PARAM_DIM
    values = [float(value) for _, value in sorted(raw.items()) if isinstance(value, (int, float))]
    return (values + [0.0] * DEFAULT_PARAM_DIM)[:DEFAULT_PARAM_DIM]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
