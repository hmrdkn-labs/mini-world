#!/usr/bin/env python3
"""Generate small Python/tract parity fixtures from a deterministic trajectory export."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "training" / "artifacts"
MODEL = ARTIFACTS / "model.onnx"
OUT = ARTIFACTS / "fixtures.json"
N = 32


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mw-fixtures-") as tmp:
        trajectory = Path(tmp) / "trajectory.jsonl"
        subprocess.run(
            [
                "cargo",
                "run",
                "--release",
                "-p",
                "mw-sim",
                "--",
                "trajectories",
                "--seed",
                "1",
                "--agents",
                "4",
                "--ticks",
                "8",
                "--out",
                str(trajectory),
                "--habits",
                "off",
            ],
            cwd=ROOT,
            check=True,
        )
        records = [json.loads(line) for line in trajectory.read_text().splitlines()[:N]]
    if len(records) != N:
        raise RuntimeError(f"trajectory export produced {len(records)} records, expected {N}")

    # Import the canonical training encoder rather than duplicating its layout.
    import sys

    sys.path.insert(0, str(ROOT / "training"))
    from mw_training.dataset import encode_record

    features = []
    masks = []
    present = []
    for record in records:
        row, mask, _, _ = encode_record(record)
        features.append(row)
        masks.append(mask)
        present.append(
            sum((1 << i) for i, n in enumerate(record["obs"]["neighbors"]) if n["present"])
        )

    raw_features = [row.copy() for row in features]
    stats = json.loads((ARTIFACTS / "norm_stats.json").read_text())
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    features = [((row - mean) / std).astype(np.float32) for row in features]

    session = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    inputs = session.get_inputs()
    if len(inputs) != 2:
        raise RuntimeError(f"expected two model inputs, got {[x.name for x in inputs]}")
    # Model exports use the feature matrix and affordance mask as their two inputs.
    feed = {
        inputs[0].name: np.asarray(features, dtype=np.float32),
        inputs[1].name: np.asarray(masks, dtype=np.int64),
    }
    outputs = session.run(None, feed)
    if len(outputs) < 2:
        raise RuntimeError(f"expected two model outputs, got {len(outputs)}")

    def finite(values: np.ndarray) -> list[float]:
        # JSON has no Infinity literal; tract's hard-masked logits are also
        # canonicalized to the same large negative sentinel in the Rust test.
        return np.where(np.isfinite(values), values, -1.0e9).astype(np.float32).tolist()

    payload = {
        "schema_version": 1,
        "records": [
            {
                "raw_features": raw_features[i].astype(np.float32).tolist(),
                "tick": records[i]["tick"],
                "agent_slot": records[i]["agent_slot"],
                "mask": masks[i],
                "present": present[i],
                "features": features[i].astype(np.float32).tolist(),
                "tool_logits": finite(np.asarray(outputs[0][i], dtype=np.float32)),
                "target_logits": finite(np.asarray(outputs[1][i], dtype=np.float32)),
            }
            for i in range(N)
        ],
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    print(f"wrote {N} fixtures to {OUT}")


if __name__ == "__main__":
    main()
