"""Train and export the mini-world schema-v2 discrete SOUL policy."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Iterable, Sequence

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .dataset import (
    FEATURE_DIM,
    NO_SECOND_CANDIDATE,
    TOOL_NAMES,
    NormStats,
    TrajectoryDataset,
    iter_jsonl,
    load_jsonl,
)
from .model import MaskedPolicy, PolicyMLP, TransformerConfig, count_parameters, masked_logits, model_input

OPSET_VERSION = 17


@dataclass
class TrainConfig:
    seed: int = 20260713
    epochs: int = 30
    batch_size: int = 2048
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 5
    hidden_dim: int = 768
    num_workers: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # MPS has no complete deterministic-algorithms implementation; this still
    # fixes all host-side and PyTorch RNG streams used by this loop.
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _loss_and_predictions(
    model: PolicyMLP,
    batch,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    obs, masks, tools, targets = (x.to(device) for x in batch)
    raw_tool, target_logits = model(model_input(obs, masks))
    tool_logits = masked_logits(raw_tool, masks)
    tool_loss = nn.functional.cross_entropy(tool_logits, tools)
    target_rows = targets != -100
    if target_rows.any():
        target_loss = nn.functional.cross_entropy(target_logits[target_rows], targets[target_rows])
    else:
        target_loss = tool_loss.new_zeros(())
    return tool_loss + 0.1 * target_loss, tool_logits.argmax(dim=-1), tools


@torch.no_grad()
def evaluate(model: PolicyMLP, dataset: TrajectoryDataset, device: torch.device, batch_size: int) -> dict:
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss = 0.0
    total = 0
    correct = 0
    confident_total = 0
    confident_correct = 0
    tie_total = 0
    no_second_total = 0
    confusion = np.zeros((len(TOOL_NAMES), len(TOOL_NAMES)), dtype=np.int64)
    offset = 0
    for batch in loader:
        loss, pred, truth = _loss_and_predictions(model, batch, device)
        count = int(truth.numel())
        total_loss += float(loss.item()) * count
        total += count
        correct += int((pred == truth).sum().item())
        margins = dataset.score_margin[offset : offset + count]
        confident = margins > 0
        ties = margins == 0
        confident_total += int(confident.sum().item())
        confident_correct += int(((pred.cpu() == truth.cpu()) & confident).sum().item())
        tie_total += int(ties.sum().item())
        no_second_total += int((margins == NO_SECOND_CANDIDATE).sum().item())
        offset += count
        for t, p in zip(truth.cpu().tolist(), pred.cpu().tolist()):
            confusion[t, p] += 1
    return {
        "loss": total_loss / max(total, 1),
        "match_rate": correct / max(total, 1),
        "count": total,
        "confident_match_rate": confident_correct / max(confident_total, 1),
        "confident_count": confident_total,
        "tie_fraction": tie_total / max(total, 1),
        "tie_count": tie_total,
        "no_second_candidate_count": no_second_total,
        "confusion": confusion,
    }


def train_policy(
    train_data: TrajectoryDataset,
    heldout_data: TrajectoryDataset,
    config: TrainConfig | None = None,
    *,
    device: torch.device | None = None,
) -> tuple[PolicyMLP, dict, NormStats]:
    config = config or TrainConfig()
    seed_everything(config.seed)
    device = device or select_device()
    model = PolicyMLP(hidden_dim=config.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=config.num_workers,
    )
    best_state = None
    best_score = float("-inf")
    stale = 0
    history: list[dict] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in loader:
            optimizer.zero_grad(set_to_none=True)
            loss, _, truth = _loss_and_predictions(model, batch, device)
            loss.backward()
            optimizer.step()
            count = int(truth.numel())
            total_loss += float(loss.item()) * count
            total += count
        train_metrics = evaluate(model, train_data, device, config.batch_size)
        heldout_metrics = evaluate(model, heldout_data, device, config.batch_size)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(total, 1),
            "train_match": train_metrics["match_rate"],
            "heldout_loss": heldout_metrics["loss"],
            "heldout_match": heldout_metrics["match_rate"],
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_loss={row['train_loss']:.4f} "
            f"heldout_loss={row['heldout_loss']:.4f} "
            f"train_match={row['train_match']:.3f} heldout_match={row['heldout_match']:.3f}",
            flush=True,
        )
        if heldout_metrics["match_rate"] > best_score:
            best_score = heldout_metrics["match_rate"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("training produced no checkpoint")
    model.load_state_dict(best_state)
    final_train = evaluate(model, train_data, device, config.batch_size)
    final_eval = evaluate(model, heldout_data, device, config.batch_size)
    metrics = {
        "params": count_parameters(model),
        "device": str(device),
        "train_match_rate": final_train["match_rate"],
        "heldout_match_rate": final_eval["match_rate"],
        "heldout_match_at_1": final_eval["match_rate"],
        "confident_subset_match": final_eval["confident_match_rate"],
        "confident_subset_count": final_eval["confident_count"],
        "tie_fraction": final_eval["tie_fraction"],
        "tie_count": final_eval["tie_count"],
        "no_second_candidate_count": final_eval["no_second_candidate_count"],
        "train_loss": final_train["loss"],
        "heldout_loss": final_eval["loss"],
        "train_count": final_train["count"],
        "heldout_count": final_eval["count"],
        "per_tool_confusion": {
            name: {pred: int(final_eval["confusion"][i, j]) for j, pred in enumerate(TOOL_NAMES)}
            for i, name in enumerate(TOOL_NAMES)
        },
        "history": history,
        "config": asdict(config),
        "train_seeds": list(train_data.seeds),
        "heldout_seeds": list(heldout_data.seeds),
        "opset": OPSET_VERSION,
    }
    return model.cpu(), metrics, train_data.norm  # type: ignore[return-value]


def export_onnx(model: PolicyMLP, path: str | Path) -> None:
    """Export the masked policy with dynamic batch and pinned opset."""
    wrapper = MaskedPolicy(model.eval())
    obs = torch.zeros((1, FEATURE_DIM), dtype=torch.float32)
    mask = torch.full((1,), (1 << len(TOOL_NAMES)) - 1, dtype=torch.long)
    torch.onnx.export(
        wrapper,
        (obs, mask),
        str(path),
        input_names=["obs", "afforded_mask"],
        output_names=["tool_logits", "target_logits"],
        dynamic_axes={"obs": {0: "batch"}, "afforded_mask": {0: "batch"}, "tool_logits": {0: "batch"}, "target_logits": {0: "batch"}},
        opset_version=OPSET_VERSION,
        dynamo=False,
    )


def _load_splits(paths: Sequence[str], train_seeds: Sequence[int], heldout_seeds: Sequence[int]):
    train = load_jsonl(paths, seeds=train_seeds)
    heldout = load_jsonl(paths, seeds=heldout_seeds, norm=train.norm)
    return train, heldout


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", default=["data/traj-s1.jsonl", "data/traj-s2.jsonl", "data/traj-s3.jsonl", "data/traj-s4.jsonl", "data/traj-s5.jsonl"])
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--train-seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--heldout-seeds", nargs="+", type=int, default=[5])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, heldout = _load_splits(args.data, args.train_seeds, args.heldout_seeds)
    config = TrainConfig(epochs=args.epochs, batch_size=args.batch_size, hidden_dim=args.hidden_dim, patience=args.patience)
    model, metrics, norm = train_policy(train, heldout, config)
    if norm is None:
        raise RuntimeError("training split did not produce normalization statistics")
    export_onnx(model, out / "model.onnx")
    (out / "norm_stats.json").write_text(json.dumps(norm.as_dict(), indent=2) + "\n", encoding="utf-8")
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"params={metrics['params']} device={metrics['device']} train_action_match={metrics['train_match_rate']:.3%} heldout_action_match={metrics['heldout_match_rate']:.3%}")
    print(
        f"heldout_match_at_1={metrics['heldout_match_at_1']:.3%} "
        f"confident_subset_match={metrics['confident_subset_match']:.3%} "
        f"confident_subset_count={metrics['confident_subset_count']} "
        f"tie_fraction={metrics['tie_fraction']:.3%} tie_count={metrics['tie_count']}"
    )
    print("per_tool_confusion=" + json.dumps(metrics["per_tool_confusion"], sort_keys=True))
    print(f"artifacts={out}")


if __name__ == "__main__":
    main()
