"""JSONL schema-v2 loader for mini-world trajectory exports.

The encoder mirrors ``crates/mw-sim/src/trajectory.rs`` and never consumes
decision or outcome fields (which would leak the label).  ``score_margin`` is
retained as evaluation-only metadata and is deliberately absent from the
feature vector.

Feature layout (129 float32 values): persona traits (5), persona need_weights
(3), agent_slot (1), observation tick (1), self_stats (3), self_pos (2),
self_cell_class one-hot (5), eight neighbors × thirteen values ``present, dist2,
opinion, faction, kind, id_slot, rel_x, rel_y, cell_class one-hot`` (104), four
event buckets (4), and goal (1). Values are bounded integer fields scaled by
these constants: 1000, 1000, 50, 10000, 1000, 16, 16, 512, 1000, 4, 4, 50,
16, 16, 100, and 8 respectively. Missing neighbor ids are encoded as zero.
Categorical cell classes use five binary one-hot features.
The separate ``afforded_mask`` tensor is the 12-bit action mask; it is not
included in the observation features, then applied to tool logits at
training/inference.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

TOOL_NAMES = (
    "move", "eat", "sleep", "work", "speak", "give", "pickup", "drop",
    "use", "follow", "flee", "idle",
)
TOOL_TO_ID = {name: i for i, name in enumerate(TOOL_NAMES)}
N_TOOLS = len(TOOL_NAMES)
N_NEIGHBORS = 8
N_CELL_CLASSES = 5
FEATURE_DIM = 5 + 3 + 1 + 1 + 3 + 2 + N_CELL_CLASSES + N_NEIGHBORS * 13 + 4 + 1
EXPERTISE_LEVELS = ("novice", "capable", "expert")
EXPERTISE_TO_ID = {name: i for i, name in enumerate(EXPERTISE_LEVELS)}
EXPERTISE_DIM = len(EXPERTISE_LEVELS)
DEFAULT_EXPERTISE = "capable"

def expertise_id(value: object = DEFAULT_EXPERTISE) -> int:
    """Return the stable novice/capable/expert id for an explicit rank."""
    if isinstance(value, bool):
        raise ValueError(f"invalid expertise rank: {value!r}")
    if isinstance(value, int):
        if 0 <= value < EXPERTISE_DIM:
            return value
    else:
        normalized = str(value).strip().lower()
        if normalized in EXPERTISE_TO_ID:
            return EXPERTISE_TO_ID[normalized]
    raise ValueError(f"unknown expertise rank: {value!r}")


def expertise_vector(value: object = DEFAULT_EXPERTISE) -> np.ndarray:
    """Encode an explicit rank as a deterministic one-hot float vector."""
    vector = np.zeros(EXPERTISE_DIM, dtype=np.float32)
    vector[expertise_id(value)] = 1.0
    return vector


def record_expertise(record: Mapping) -> np.ndarray:
    """Read only structured expertise metadata, defaulting legacy rows to capable."""
    raw = record.get("expertise")
    if isinstance(raw, Mapping):
        raw = raw.get("level", raw.get("expertise_level"))
    if raw is None:
        raw = record.get("expertise_level", record.get("expertise_rank"))
    return expertise_vector(DEFAULT_EXPERTISE if raw is None else raw)
IGNORE_INDEX = -100
NO_SECOND_CANDIDATE = 2**31 - 1


def _scaled(value: float, divisor: float) -> float:
    return float(value) / divisor


def _one_hot(value: int, size: int, field: str) -> list[float]:
    if not 0 <= value < size:
        raise ValueError(f"{field} outside one-hot range: {value!r}")
    return [1.0 if i == value else 0.0 for i in range(size)]


def encode_record(record: Mapping) -> tuple[np.ndarray, int, int, int]:
    """Encode one schema-v2 record as ``(obs, mask, tool, target_index)``.

    ``target_index`` is the neighbor slot (0..7), not the global entity id;
    absent/non-neighbor targets return ``IGNORE_INDEX`` for an ignored pointer
    loss. ``score_margin`` is intentionally not returned: callers use the
    dataset's metadata tensor for evaluation stratification only.
    ``ValueError`` is raised for unsupported schema versions or tools.
    """
    if int(record.get("schema_version", -1)) != 2:
        raise ValueError(f"unsupported trajectory schema: {record.get('schema_version')!r}")
    persona = record["persona"]
    obs = record["obs"]
    features: list[float] = []
    features.extend(_scaled(v, 1000.0) for v in persona["traits"])
    features.extend(_scaled(v, 1000.0) for v in persona["need_weights"])
    features.append(_scaled(record["agent_slot"], 50.0))
    features.append(_scaled(obs["tick"], 10000.0))
    features.extend(_scaled(v, 1000.0) for v in obs["self_stats"])
    features.extend(_scaled(v, 16.0) for v in obs["self_pos"])
    features.extend(_one_hot(int(obs["self_cell_class"]), N_CELL_CLASSES, "self_cell_class"))
    for n in obs["neighbors"]:
        features.extend((
            float(bool(n["present"])),
            _scaled(n["dist2"], 512.0),
            _scaled(n["opinion"], 1000.0),
            _scaled(n["faction"], 4.0),
            _scaled(n["kind"], 4.0),
            _scaled(n["id_slot"] or 0, 50.0),
            _scaled(n["rel_pos"][0], 16.0),
            _scaled(n["rel_pos"][1], 16.0),
        ))
        features.extend(_one_hot(int(n["cell_class"]), N_CELL_CLASSES, "neighbor.cell_class"))
    features.extend(_scaled(v, 100.0) for v in obs["events"])
    features.append(_scaled(obs["goal"], 8.0))
    if len(features) != FEATURE_DIM:
        raise ValueError(f"encoder produced {len(features)} features, expected {FEATURE_DIM}")
    mask = int(record["afforded_mask"]) & ((1 << N_TOOLS) - 1)
    # The exporter duplicates the mask in obs; reject disagreement rather than
    # silently training against a different affordance contract.
    if int(obs["tool_mask"]) != int(record["afforded_mask"]):
        raise ValueError("trajectory afforded_mask and obs.tool_mask disagree")
    try:
        tool = TOOL_TO_ID[record["decision"]["tool"]]
    except KeyError as exc:
        raise ValueError(f"unknown decision tool: {record['decision'].get('tool')!r}") from exc
    target_index = IGNORE_INDEX
    target_slot = record["decision"].get("target_slot")
    if target_slot is not None:
        for i, n in enumerate(obs["neighbors"]):
            if n.get("present") and n.get("id_slot") == target_slot:
                target_index = i
                break
    return np.asarray(features, dtype=np.float32), mask, tool, target_index


def iter_jsonl(paths: str | Path | Sequence[str | Path]) -> Iterator[dict]:
    """Yield decoded records from one or more JSONL files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]
    for path in paths:
        with Path(path).open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON in {path}:{line_number}") from exc


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray

    def as_dict(self) -> dict[str, list[float]]:
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, value: Mapping) -> "NormStats":
        return cls(np.asarray(value["mean"], dtype=np.float32), np.asarray(value["std"], dtype=np.float32))


def compute_norm_stats(features: np.ndarray) -> NormStats:
    mean = features.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return NormStats(mean, std)


@dataclass
class TrajectoryDataset(Dataset):
    """Tensor dataset containing one deterministic seed split."""

    obs: torch.Tensor
    afforded_mask: torch.Tensor
    tool: torch.Tensor
    target: torch.Tensor
    # Evaluation metadata only; never concatenated into model_input.
    score_margin: torch.Tensor
    seeds: tuple[int, ...]
    norm: NormStats | None = None

    def __len__(self) -> int:
        return int(self.tool.shape[0])

    def __getitem__(self, index: int):
        return self.obs[index], self.afforded_mask[index], self.tool[index], self.target[index]

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping],
        *,
        seeds: Sequence[int] | None = None,
        norm: NormStats | None = None,
    ) -> "TrajectoryDataset":
        selected = set(int(s) for s in seeds) if seeds is not None else None
        encoded: list[np.ndarray] = []
        masks: list[int] = []
        tools: list[int] = []
        targets: list[int] = []
        score_margins: list[int] = []
        seen_seeds: set[int] = set()
        for record in records:
            seed = int(record["seed"])
            if selected is not None and seed not in selected:
                continue
            x, mask, tool, target = encode_record(record)
            encoded.append(x)
            masks.append(mask)
            tools.append(tool)
            targets.append(target)
            score_margins.append(int(record["decision"]["score_margin"]))
            seen_seeds.add(seed)
        if not encoded:
            raise ValueError(f"no trajectory records selected for seeds={sorted(selected) if selected is not None else None}")
        array = np.stack(encoded)
        if norm is None:
            norm = compute_norm_stats(array)
        array = (array - norm.mean) / norm.std
        return cls(
            obs=torch.from_numpy(array.astype(np.float32, copy=False)),
            afforded_mask=torch.tensor(masks, dtype=torch.long),
            tool=torch.tensor(tools, dtype=torch.long),
            target=torch.tensor(targets, dtype=torch.long),
            score_margin=torch.tensor(score_margins, dtype=torch.long),
            seeds=tuple(sorted(seen_seeds)),
            norm=norm,
        )


def load_jsonl(
    paths: str | Path | Sequence[str | Path],
    *,
    seeds: Sequence[int] | None = None,
    norm: NormStats | None = None,
) -> TrajectoryDataset:
    return TrajectoryDataset.from_records(iter_jsonl(paths), seeds=seeds, norm=norm)
