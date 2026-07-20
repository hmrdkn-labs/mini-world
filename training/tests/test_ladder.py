from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from mw_training.dataset import (
    EXPERTISE_DIM,
    TOOL_NAMES,
    expertise_vector,
)
from mw_training.ladder import (
    TIERS,
    _shared_norm,
    _split,
    evaluate_tail_behavior,
    tier_parameter_counts,
)
from mw_training.omni import OmniDataset, OmniPolicy
from mw_training.train_omni import OmniTrainConfig, export_omni_onnx, train_omni

def test_expertise_encoding_and_omni_input_shape():
    rows = records()[:3]
    rows[0]["expertise"] = "novice"
    rows[1]["expertise_rank"] = "expert"
    dataset = OmniDataset.from_trajectory(rows, manifest=TOOL_NAMES)
    np.testing.assert_array_equal(dataset.expertise[0].numpy(), expertise_vector("novice"))
    np.testing.assert_array_equal(dataset.expertise[1].numpy(), expertise_vector("expert"))
    np.testing.assert_array_equal(dataset.expertise[2].numpy(), expertise_vector("capable"))
    assert dataset.expertise.shape == (3, EXPERTISE_DIM)
    model = OmniPolicy(hidden_dim=16).eval()
    with torch.no_grad():
        scores, targets, params = model(
            dataset.obs,
            dataset.tool_descriptors,
            dataset.afforded,
            dataset.expertise,
        )
    assert scores.shape == (3, len(TOOL_NAMES))
    assert targets.shape[0] == params.shape[0] == 3
    novice_scores = model(
        dataset.obs[:1],
        dataset.tool_descriptors[:1],
        dataset.afforded[:1],
        torch.from_numpy(expertise_vector("novice")).unsqueeze(0),
    )[0]
    expert_scores = model(
        dataset.obs[:1],
        dataset.tool_descriptors[:1],
        dataset.afforded[:1],
        torch.from_numpy(expertise_vector("expert")).unsqueeze(0),
    )[0]
    assert not torch.equal(novice_scores, expert_scores)


def test_omni_onnx_export_and_runtime_inputs(tmp_path: Path):
    dataset = OmniDataset.from_fixtures(Path(__file__).parents[1] / "artifacts" / "fixtures.json")
    model = OmniPolicy(hidden_dim=16).eval()
    path = tmp_path / "omni.onnx"
    export_omni_onnx(model, path)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [item.name for item in session.get_inputs()] == [
        "obs", "tool_descriptors", "afforded", "expertise"
    ]
    outputs = session.run(
        None,
        {
            "obs": dataset.obs[:2].numpy(),
            "tool_descriptors": dataset.tool_descriptors[:2].numpy(),
            "afforded": dataset.afforded[:2].numpy(),
            "expertise": dataset.expertise[:2].numpy(),
        },
    )
    assert [output.shape[0] for output in outputs] == [2, 2, 2]

def records() -> list[dict]:
    path = Path(__file__).parents[1] / "artifacts" / "spark_omni.jsonl"
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream][:20]


def test_ladder_has_seven_explicitly_sized_tiers():
    counts = tier_parameter_counts()
    assert len(TIERS) == 7
    assert list(counts.values())[0] > 300_000
    assert list(counts.values())[-1] < 21_000_000
    assert list(counts.values()) == sorted(counts.values())


def test_group_split_is_deterministic_and_leak_free():
    rows = []
    for index, row in enumerate(records()):
        copy = dict(row)
        copy["world_seed"] = index // 2
        copy["trajectory_id"] = index % 5
        rows.append(copy)
    first = _split(rows)
    second = _split(list(reversed(rows)))
    assert first == second
    groups = [
        {(r["world_seed"], r["trajectory_id"]) for r in partition}
        for partition in first
    ]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    assert all(partition for partition in first)


def test_small_fixture_keeps_one_group_intact():
    train, val, test = _split(records())
    assert len(train) == 20
    assert not val
    assert not test


def test_normalization_uses_training_records_only():
    rows = records()
    train = rows[:10]
    validation = rows[10:11]
    import numpy as np
    from mw_training.dataset import compute_norm_stats, encode_record

    expected = compute_norm_stats(np.stack([encode_record(row)[0] for row in train]))
    norm = _shared_norm(train)
    np.testing.assert_allclose(norm.mean, expected.mean)
    np.testing.assert_allclose(norm.std, expected.std)
    assert not np.allclose(norm.mean, _shared_norm(rows).mean)


def test_tier_model_seed_is_not_width_derived():
    assert len({tier.seed for tier in TIERS}) == 1


def test_test_evaluation_is_last_and_not_used_for_selection(monkeypatch):
    data = OmniDataset.from_fixtures(Path(__file__).parents[1] / "artifacts" / "fixtures.json")
    fields = tuple(data.__dict__.values())
    train = OmniDataset(*(field[:4] for field in fields))
    val = OmniDataset(*(field[4:6] for field in fields))
    test = OmniDataset(*(field[6:8] for field in fields))
    calls = []

    def fake_evaluate(model, dataset, device, batch_size):
        calls.append(dataset)
        return {"loss": 0.0, "match_rate": 0.5, "count": len(dataset)}

    monkeypatch.setattr("mw_training.train_omni._evaluate", fake_evaluate)
    _, metrics = train_omni(
        train,
        val,
        test,
        config=OmniTrainConfig(epochs=2, batch_size=2, hidden_dim=8, patience=8),
    )
    assert calls[-1] is test
    assert calls[-2] is val
    assert metrics["selection_metric"] == "val_match_rate"
    assert metrics["test_evaluated_after_selection"] is True

def test_tail_evaluator_reports_finite_deterministic_metrics():
    rows = records()
    norm = _shared_norm(rows)
    dataset = OmniDataset.from_trajectory(rows, manifest=TOOL_NAMES, norm=norm)
    model = OmniPolicy(hidden_dim=32)
    first = evaluate_tail_behavior(model, dataset, torch.device("cpu"), rows=4, feature_stride=16, batch_size=8)
    second = evaluate_tail_behavior(model, dataset, torch.device("cpu"), rows=4, feature_stride=16, batch_size=8)
    assert first == second
    assert first["tail_probe_rows"] == 72
    assert 0.0 <= first["plus4_match_rate"] <= 1.0
    assert 0.0 <= first["minus4_match_rate"] <= 1.0
    assert 0.0 <= first["tail_clamp_changed_fraction"] <= 1.0
