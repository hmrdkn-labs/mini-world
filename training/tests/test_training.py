from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from mw_training.dataset import FEATURE_DIM, IGNORE_INDEX, TOOL_NAMES, encode_record, load_jsonl
from mw_training.model import MaskedPolicy, PolicyMLP, masked_logits, model_input
from mw_training.train import export_onnx


def tiny_record(seed: int = 1, tool: str = "move") -> dict:
    neighbors = [
        {
            "present": True,
            "dist2": 1,
            "opinion": 0,
            "faction": 0,
            "kind": 0,
            "id_slot": 7,
            "pos": [1, 0],
            "rel_pos": [1, 0],
            "cell_class": 2,
        },
        *[
            {
                "present": False,
                "dist2": 0,
                "opinion": 0,
                "faction": 0,
                "kind": 0,
                "id_slot": None,
                "pos": [0, 0],
                "rel_pos": [0, 0],
                "cell_class": 0,
            }
            for _ in range(7)
        ],
    ]
    return {
        "schema_version": 2,
        "seed": seed,
        "tick": 3,
        "agent_slot": 0,
        "persona": {"traits": [1, 2, 3, 4, 5], "need_weights": [6, 7, 8]},
        "obs": {
            "tick": 3,
            "self_stats": [900, 800, 700],
            "self_pos": [0, 0],
            "self_cell_class": 1,
            "neighbors": neighbors,
            "events": [1, 2, 0, 0],
            "tool_mask": 0b1011,
            "goal": 1,
        },
        "afforded_mask": 0b1011,
        "decision": {
            "tool": tool,
            "target_slot": 7 if tool == "speak" else None,
            "params": {},
            "score_margin": 0 if tool == "idle" else 12,
        },
        "outcome": {"events": [], "need_deltas": [0, 0, 0]},
        "replay": False,
    }


def test_loader_round_trip_tiny_fixture(tmp_path: Path):
    rows = [tiny_record(1), tiny_record(2, "idle")]
    path = tmp_path / "tiny.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    dataset = load_jsonl(path, seeds=[1, 2])
    assert len(dataset) == 2
    assert dataset.obs.shape == (2, FEATURE_DIM)
    assert dataset.afforded_mask.tolist() == [0b1011, 0b1011]
    assert dataset.tool.tolist() == [0, 11]
    assert dataset.target.tolist() == [IGNORE_INDEX, IGNORE_INDEX]
    assert dataset.score_margin.tolist() == [12, 0]
    assert FEATURE_DIM == 129
    # A second parse yields byte-for-byte equivalent tensors and split metadata.
    again = load_jsonl(path, seeds=[2])
    assert again.seeds == (2,)
    assert torch.equal(dataset.obs[1:], again.obs)


def test_score_margin_and_absolute_neighbor_pos_do_not_leak_into_features():
    record = tiny_record()
    original, *_ = encode_record(record)
    record["decision"]["score_margin"] = 0
    record["obs"]["neighbors"][0]["pos"] = [99, -99]
    changed, *_ = encode_record(record)
    np.testing.assert_array_equal(original, changed)


def test_masking_correctness():
    logits = torch.tensor([[10.0, 9.0, 8.0, 7.0] + [0.0] * 8])
    mask = torch.tensor([0b0100])
    masked = masked_logits(logits, mask)
    assert masked.argmax(dim=-1).item() == 2
    for i in range(len(TOOL_NAMES)):
        if i != 2:
            assert masked[0, i].item() == float("-inf")


def test_onnx_export_reload_parity(tmp_path: Path):
    torch.manual_seed(4)
    model = PolicyMLP(hidden_dim=32).eval()
    path = tmp_path / "policy.onnx"
    export_onnx(model, path)
    obs = torch.randn(3, FEATURE_DIM)
    masks = torch.tensor([0b1011, 0b1000001, 0b111111111111], dtype=torch.long)
    with torch.no_grad():
        expected = MaskedPolicy(model)(obs, masks)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"obs": obs.numpy(), "afforded_mask": masks.numpy()})
    np.testing.assert_allclose(expected[0].numpy(), got[0], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(expected[1].numpy(), got[1], rtol=1e-4, atol=1e-4)
