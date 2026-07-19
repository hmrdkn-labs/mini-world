from __future__ import annotations

import json
from pathlib import Path

from mw_training.omni import OmniDataset
from mw_training.spark_dataset import (
    assemble_labels,
    build_teacher_prompt,
    collect_states,
    summarize_states,
    write_prompts,
)


def state(seed: int = 4) -> dict:
    neighbors = [
        {
            "present": i == 0,
            "dist2": 1 if i == 0 else 0,
            "opinion": -100 if i == 0 else 0,
            "faction": 1,
            "kind": 0,
            "id_slot": 1 if i == 0 else None,
            "pos": [1, 0],
            "rel_pos": [1, 0],
            "cell_class": 0,
        }
        for i in range(8)
    ]
    return {
        "schema_version": 2,
        "seed": seed,
        "tick": 2,
        "agent_slot": 0,
        "persona": {"traits": [900, 700, 400, 200, 100], "need_weights": [500, 800, 300]},
        "obs": {
            "tick": 2,
            "self_stats": [200, 800, 700],
            "self_pos": [2, 3],
            "self_cell_class": 1,
            "neighbors": neighbors,
            "events": [0, 1, 0, 0],
            "tool_mask": 1,
            "goal": 1,
        },
        "afforded_mask": 1,
        "decision": {"tool": "move", "target_slot": None, "params": {}},
        "outcome": {"events": [], "need_deltas": [0, 0, 0]},
        "replay": False,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_prompt_contains_persona_and_structured_semantic_observation():
    prompt = build_teacher_prompt(state())
    assert "Persona sketch:" in prompt
    assert "afforded_tools" in prompt
    assert '"self_stats"' in prompt
    assert '"tool":"<afforded tool>"' in prompt


def test_assembler_kernel_filters_illegal_labels_and_is_trainable(tmp_path: Path):
    states = tmp_path / "states.jsonl"
    labels = tmp_path / "labels.jsonl"
    output = tmp_path / "assembled.jsonl"
    row = state()
    write_jsonl(states, [row])
    write_jsonl(
        labels,
        [
            {"id": "4:2:0", "tool": "move", "target": None, "arg": {"dx": 1}, "why": "escape"},
            {"id": "4:2:0", "tool": "idle", "target": None, "arg": None, "why": "illegal"},
        ],
    )
    report = assemble_labels(states, labels, output)
    assert report["labels"] == 2
    assert report["kept"] == 1
    assert report["illegal_dropped"] == 1
    assembled = [json.loads(line) for line in output.read_text().splitlines()]
    assert assembled[0]["decision"]["tool"] == "move"
    dataset = OmniDataset.from_trajectory(assembled, ["move", "idle"])
    assert len(dataset) == 1


def test_prompt_metadata_and_distribution_report(tmp_path: Path):
    states = tmp_path / "states.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    rows = [state(1), state(2)]
    rows[1]["obs"]["self_cell_class"] = 2
    write_jsonl(states, rows)
    report = write_prompts(states, prompts)
    assert report["prompts"] == 2
    payload = json.loads(prompts.read_text().splitlines()[0])
    assert payload["id"] == "1:2:0"
    assert payload["state"]["obs"]["neighbors"][0]["opinion"] == -100
    summary = summarize_states(rows)
    assert summary["states"] == 2
    assert summary["locations"]["home"] == 1
    assert summary["locations"]["bakery"] == 1
    assert summary["threat_presence"]["present"] == 2

def test_collect_uses_even_stress_profile_mix(tmp_path, monkeypatch):
    calls = []

    def fake_export(root, seed, agents, ticks, out, *, profile="healthy", fraction=25):
        calls.append((profile, fraction))
        rows = [state(seed + index) for index in range(agents * 2)]
        out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr("mw_training.spark_dataset._run_export", fake_export)
    output = tmp_path / "states.jsonl"
    report = collect_states(output, states=8, agents=2, ticks_per_seed=2)
    assert len(output.read_text().splitlines()) == 8
    assert calls == [
        ("healthy", 25),
        ("scarcity", 25),
        ("hostile", 50),
        ("exhausted", 25),
    ]
    assert report["profiles"] == {
        "healthy": 2,
        "scarcity": 2,
        "hostile": 2,
        "exhausted": 2,
    }
