"""Train and evaluate the seven deterministic OMNI distillation tiers.

The ladder is deliberately explicit: each tier has one hidden width, one artifact
path, and one seed.  The output directory is self describing so Rust can select a
tier without guessing from a model filename.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import torch

from .dataset import NormStats, TOOL_NAMES, compute_norm_stats, iter_jsonl, encode_record
from .dataset import EXPERTISE_LEVELS
from .omni import OmniDataset, OmniPolicy, count_parameters
from .train_omni import OmniTrainConfig, export_omni_onnx, train_omni

SPLIT_SEED = 20260715
MODEL_SEED = 20260715
PARTITION_RATIOS = (0.70, 0.15, 0.15)
PARTITION_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class LadderTier:
    index: int
    name: str
    hidden_dim: int
    seed: int


# These widths intentionally span just above 300K through about 20M parameters.
TIERS: tuple[LadderTier, ...] = tuple(
    LadderTier(index, f"tier-{index}", width, MODEL_SEED)
    for index, width in enumerate((296, 448, 640, 896, 1280, 1792, 2560))
)


def tier_parameter_counts() -> dict[str, int]:
    return {
        tier.name: count_parameters(OmniPolicy(hidden_dim=tier.hidden_dim))
        for tier in TIERS
    }


def _group_key(record: dict) -> tuple[str, str]:
    """Return the stable world-seed/trajectory identity for one record."""
    world_seed = record.get("world_seed", record.get("seed"))
    if world_seed is None:
        raise ValueError("record has no world seed")
    trajectory = next(
        (
            record[name]
            for name in ("trajectory_id", "trajectory", "episode_id", "episode", "run")
            if name in record
        ),
        0,
    )
    return str(world_seed), str(trajectory)


def _split(
    records: Sequence[dict],
    *,
    split_seed: int = SPLIT_SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Deterministically assign whole world-seed/trajectory groups.

    Group sizes make exact percentages impossible for some small fixtures.  The
    greedy assignment minimizes the resulting ratio error and leaves
    under-sized partitions empty rather than splitting a group.
    """
    groups: dict[tuple[str, str], list[dict]] = {}
    for record in records:
        groups.setdefault(_group_key(record), []).append(record)
    for group in groups.values():
        group.sort(key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":")))
    if not groups:
        raise ValueError("ladder split requires records")
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(
            f"{split_seed}:{key[0]}:{key[1]}".encode("utf-8")
        ).hexdigest(),
    )
    targets = [max(1.0, len(records) * ratio) for ratio in PARTITION_RATIOS]
    counts = [0, 0, 0]
    selected: list[list[dict]] = [[], [], []]
    for key in ordered:
        group = groups[key]
        partition = min(range(3), key=lambda index: counts[index] / targets[index])
        selected[partition].extend(group)
        counts[partition] += len(group)
    return selected[0], selected[1], selected[2]

def _matched_group_key(record: dict) -> str:
    """Return the source-state identity shared by all three expertise rows."""
    for value in (
        record.get("matched_group_id"),
        record.get("matched_group"),
        record.get("expertise", {}).get("group_id") if isinstance(record.get("expertise"), dict) else None,
        record.get("group_id"),
        record.get("source_state_id"),
    ):
        if value is not None:
            if isinstance(value, dict):
                value = value.get("id") or value.get("state_id")
            if value is not None:
                return str(value)
    raise ValueError("matched record has no grouping identity")


def _split_matched(
    records: Sequence[dict],
    *,
    split_seed: int,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split matched expertise triplets without separating their levels."""
    groups: dict[str, list[dict]] = {}
    for record in records:
        groups.setdefault(_matched_group_key(record), []).append(record)
    expected = set(EXPERTISE_LEVELS)
    for key, group in groups.items():
        levels = [str(record.get("expertise_level", record.get("expertise", {}).get("level", ""))) for record in group]
        if len(group) != 3 or set(levels) != expected or len(set(levels)) != 3:
            raise ValueError(f"matched group {key!r} is not one novice/capable/expert triplet")
        group.sort(key=lambda record: json.dumps(record, sort_keys=True, separators=(",", ":")))
    if not groups:
        raise ValueError("matched split requires records")
    ordered = sorted(
        groups,
        key=lambda key: hashlib.sha256(f"{split_seed}:{key}".encode("utf-8")).hexdigest(),
    )
    targets = [max(1.0, len(groups) * ratio) for ratio in PARTITION_RATIOS]
    counts = [0, 0, 0]
    selected: list[list[dict]] = [[], [], []]
    for key in ordered:
        partition = min(range(3), key=lambda index: counts[index] / targets[index])
        selected[partition].extend(groups[key])
        counts[partition] += 1
    return selected[0], selected[1], selected[2]



def _shared_norm(records: Iterable[dict]) -> NormStats:
    encoded = [encode_record(record)[0] for record in records]
    if not encoded:
        raise ValueError("cannot compute normalization for empty dataset")
    import numpy as np

    return compute_norm_stats(np.stack(encoded))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _predict(model: OmniPolicy, data: OmniDataset, device: torch.device, batch_size: int) -> torch.Tensor:
    predictions: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            end = min(start + batch_size, len(data))
            scores, _, _ = model(
                data.obs[start:end].to(device),
                data.tool_descriptors[start:end].to(device),
                data.afforded[start:end].to(device),
                data.expertise[start:end].to(device),
            )
            predictions.append(scores.argmax(-1).cpu())
    return torch.cat(predictions) if predictions else torch.empty(0, dtype=torch.long)


def evaluate_tail_behavior(
    model: OmniPolicy,
    heldout: OmniDataset,
    device: torch.device,
    *,
    rows: int = 128,
    feature_stride: int = 4,
    batch_size: int = 64,
) -> dict[str, float | int]:
    """Measure test action match and deterministic ±4σ/tail-clamp behavior.

    Probe rows set one normalized feature at a time to ±4σ.  The clamp metric
    clips every heldout feature to the same [-4, 4] standardized range, making
    behavior at observed tails explicit rather than silently relying on finite
    logits.
    """
    n = min(rows, len(heldout))
    base = OmniDataset(
        heldout.obs[:n].clone(), heldout.tool_descriptors[:n].clone(),
        heldout.afforded[:n].clone(), heldout.expertise[:n].clone(), heldout.tool[:n].clone(),
        heldout.target[:n].clone(), heldout.params[:n].clone(),
    )
    baseline = _predict(model, base, device, batch_size)
    truth = base.tool
    clipped = base.obs.clamp(-4.0, 4.0)
    clamped = OmniDataset(
        clipped, base.tool_descriptors, base.afforded, base.expertise, base.tool, base.target, base.params
    )
    clamp_prediction = _predict(model, clamped, device, batch_size)
    dimensions = list(range(0, heldout.obs.shape[1], max(1, feature_stride)))
    probe_obs: list[torch.Tensor] = []
    for dimension in dimensions:
        for sign in (-1.0, 1.0):
            row = base.obs.clone()
            row[:, dimension] = sign * 4.0
            probe_obs.append(row)
    probes = OmniDataset(
        torch.cat(probe_obs),
        base.tool_descriptors.repeat(len(probe_obs), 1, 1),
        base.afforded.repeat(len(probe_obs), 1),
        base.expertise.repeat(len(probe_obs), 1),
        base.tool.repeat(len(probe_obs)),
        base.target.repeat(len(probe_obs)),
        base.params.repeat(len(probe_obs), 1),
    )
    probe_prediction = _predict(model, probes, device, batch_size).view(len(dimensions), 2, n)
    probe_truth = probes.tool.view(len(dimensions), 2, n)
    minus = probe_prediction[:, 0].reshape(-1)
    plus = probe_prediction[:, 1].reshape(-1)
    minus_truth = probe_truth[:, 0].reshape(-1)
    plus_truth = probe_truth[:, 1].reshape(-1)
    raw_tail = (base.obs.abs() > 4.0).float().mean().item()
    return {
        "test_match_rate": float((baseline == truth).float().mean()),
        "tail_clamp_match_rate": float((clamp_prediction == truth).float().mean()),
        "tail_clamp_changed_fraction": float((clamp_prediction != baseline).float().mean()),
        "raw_tail_feature_fraction": float(raw_tail),
        "plus4_match_rate": float((plus == plus_truth).float().mean()),
        "minus4_match_rate": float((minus == minus_truth).float().mean()),
        "plus4_prediction_change_fraction": float((plus != baseline.repeat(len(dimensions))).float().mean()),
        "minus4_prediction_change_fraction": float((minus != baseline.repeat(len(dimensions))).float().mean()),
        "tail_probe_rows": len(probes),
        "tail_probe_dimensions": len(dimensions),
    }


def _partition_manifest(records: Sequence[dict]) -> list[dict[str, str]]:
    return [
        {"world_seed": world_seed, "trajectory": trajectory}
        for world_seed, trajectory in sorted({_group_key(record) for record in records})
    ]


def train_matched_reference(
    records: Sequence[dict],
    out_dir: str | Path,
    *,
    split_seed: int,
    model_seeds: Sequence[int],
    hidden_dim: int = 296,
    epochs: int = 24,
    batch_size: int = 64,
    device: torch.device | None = None,
    dataset_sha256: str | None = None,
) -> dict:
    """Train the smallest corrected reference on shared matched triplet splits."""
    if len(model_seeds) < 3:
        raise ValueError("matched reference requires at least three model seeds")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_records, val_records, test_records = _split_matched(records, split_seed=split_seed)
    norm = _shared_norm(train_records)
    norm_payload = {**norm.as_dict(), "fit_partition": "train", "fit_count": len(train_records)}
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    partitions = {
        name: sorted({_matched_group_key(record) for record in partition})
        for name, partition in zip(PARTITION_NAMES, (train_records, val_records, test_records))
    }
    provenance = {
        "schema_version": 2,
        "dataset": "spark_matched_expertise.jsonl",
        "dataset_sha256": dataset_sha256,
        "dataset_count": len(records),
        "matched_group_count": len(set(_matched_group_key(record) for record in records)),
        "split_seed": split_seed,
        "model_seeds": list(model_seeds),
        "hidden_dim": hidden_dim,
        "parameter_count": count_parameters(OmniPolicy(hidden_dim=hidden_dim)),
        "partition_ratios": dict(zip(PARTITION_NAMES, PARTITION_RATIOS)),
        "partition_counts": {name: len(partition) for name, partition in zip(PARTITION_NAMES, (train_records, val_records, test_records))},
        "partition_group_counts": {name: len(groups) for name, groups in partitions.items()},
        "partition_groups": partitions,
        "triplets_grouped": True,
        "normalization_fit_partition": "train",
        "selection_partition": "val",
        "test_evaluation": "once_after_validation_selection",
        "manifest": list(TOOL_NAMES),
    }
    (destination / "partitions.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    train_data = OmniDataset.from_trajectory(train_records, manifest=TOOL_NAMES, norm=norm)
    val_data = OmniDataset.from_trajectory(val_records, manifest=TOOL_NAMES, norm=norm)
    test_data = OmniDataset.from_trajectory(test_records, manifest=TOOL_NAMES, norm=norm)
    results: list[dict] = []
    for seed in model_seeds:
        config = OmniTrainConfig(
            seed=int(seed), epochs=epochs, batch_size=batch_size, hidden_dim=hidden_dim,
        )
        model, metrics = train_omni(train_data, val_data, test_data, config=config, device=device)
        seed_dir = destination / f"seed-{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        export_omni_onnx(model, seed_dir / "model.onnx")
        (seed_dir / "norm_stats.json").write_text(json.dumps(norm_payload, indent=2) + "\n", encoding="utf-8")
        metrics["provenance"] = provenance
        (seed_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        (seed_dir / "config.json").write_text(
            json.dumps({"train_config": asdict(config), "provenance": provenance}, indent=2) + "\n",
            encoding="utf-8",
        )
        hashes = {name: _sha256(seed_dir / name) for name in ("model.onnx", "norm_stats.json", "metrics.json", "config.json")}
        (seed_dir / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
        metrics["hashes"] = hashes
        results.append(metrics)
    aggregate = {
        "provenance": provenance,
        "seeds": list(model_seeds),
        "metrics": {
            metric: {str(seed): result.get(metric) for seed, result in zip(model_seeds, results)}
            for metric in ("train_match_rate", "val_match_rate", "test_match_rate", "train_loss", "val_loss", "test_loss")
        },
        "artifacts": {str(seed): results[index]["hashes"] for index, seed in enumerate(model_seeds)},
    }
    (destination / "aggregate.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    return aggregate

def train_ladder(
    records: Sequence[dict],
    out_dir: str | Path,
    *,
    epochs: int = 24,
    batch_size: int = 64,
    device: torch.device | None = None,
) -> dict:
    """Train every tier with train-only normalization and untouched test data."""
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_records, val_records, test_records = _split(records)
    norm = _shared_norm(train_records)
    norm_payload = {
        **norm.as_dict(),
        "fit_partition": "train",
        "fit_count": len(train_records),
    }
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    partitions = {
        "train": _partition_manifest(train_records),
        "val": _partition_manifest(val_records),
        "test": _partition_manifest(test_records),
    }
    provenance = {
        "split_seed": SPLIT_SEED,
        "model_seed": MODEL_SEED,
        "partition_ratios": dict(zip(PARTITION_NAMES, PARTITION_RATIOS)),
        "partition_counts": {
            "train": len(train_records),
            "val": len(val_records),
            "test": len(test_records),
        },
        "partition_groups": partitions,
        "selection_partition": "val",
        "test_evaluation": "after_validation_selection",
        "normalization_fit_partition": "train",
        "manifest": list(TOOL_NAMES),
    }
    summary: dict[str, object] = {
        "schema_version": 2,
        "device": str(device),
        "dataset_count": len(records),
        **provenance,
        "tiers": [],
    }
    train_data = OmniDataset.from_trajectory(train_records, manifest=TOOL_NAMES, norm=norm)
    val_data = (
        OmniDataset.from_trajectory(val_records, manifest=TOOL_NAMES, norm=norm)
        if val_records else train_data
    )
    test_data = (
        OmniDataset.from_trajectory(test_records, manifest=TOOL_NAMES, norm=norm)
        if test_records else None
    )
    for tier in TIERS:
        tier_dir = destination / tier.name
        tier_dir.mkdir(parents=True, exist_ok=True)
        config = OmniTrainConfig(
            seed=tier.seed,
            epochs=epochs,
            batch_size=batch_size,
            hidden_dim=tier.hidden_dim,
        )
        model, metrics = train_omni(
            train_data,
            val_data,
            test_data,
            config=config,
            device=device,
        )
        model_path = tier_dir / "model.onnx"
        export_omni_onnx(model, model_path)
        (tier_dir / "norm_stats.json").write_text(
            json.dumps(norm_payload, indent=2) + "\n", encoding="utf-8"
        )
        metrics["tier"] = asdict(tier)
        metrics["provenance"] = provenance
        if test_data is not None:
            metrics["tail_behavior"] = evaluate_tail_behavior(model.to(device), test_data, device)
        model.cpu()
        metrics_path = tier_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        config_path = tier_dir / "config.json"
        config_path.write_text(
            json.dumps({"train_config": asdict(config), "provenance": provenance}, indent=2) + "\n",
            encoding="utf-8",
        )
        hashes = {
            name: _sha256(tier_dir / name)
            for name in ("model.onnx", "norm_stats.json", "metrics.json", "config.json")
        }
        (tier_dir / "sha256.json").write_text(json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
        metrics["hashes"] = hashes
        summary["tiers"].append(metrics)
        (destination / "ladder.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="training/artifacts/spark_omni.jsonl")
    parser.add_argument("--out-dir", default="training/artifacts/ladder")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--matched", action="store_true")
    parser.add_argument("--split-seed", type=int, default=20260719)
    parser.add_argument("--model-seeds", default="20260719,20260720,20260721")
    parser.add_argument("--hidden-dim", type=int, default=296)
    args = parser.parse_args(argv)
    records = list(iter_jsonl(args.data))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.matched:
        summary = train_matched_reference(
            records,
            args.out_dir,
            split_seed=args.split_seed,
            model_seeds=tuple(int(seed) for seed in args.model_seeds.split(",") if seed),
            hidden_dim=args.hidden_dim,
            epochs=args.epochs,
            batch_size=args.batch_size,
            device=device,
            dataset_sha256=_sha256(Path(args.data)),
        )
        print(json.dumps({"device": str(device), "seeds": summary["seeds"]}, sort_keys=True))
    else:
        summary = train_ladder(records, args.out_dir, epochs=args.epochs, batch_size=args.batch_size, device=device)
        print(json.dumps({"device": summary["device"], "tiers": tier_parameter_counts()}, sort_keys=True))


if __name__ == "__main__":
    main()
