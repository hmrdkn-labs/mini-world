from __future__ import annotations
import pytest

import json
from pathlib import Path

from mw_training.omni import OmniDataset
from mw_training.spark_dataset import (
    _derive_expertise,
    analyze_teacher_consistency,
    assemble_labels,
    build_teacher_prompt,
    collect_states,
    derive_expertise_targets,
    export_consistency_sample,
    ground_labels,
    label_with_spark,
    materialize_matched_expertise,
    sample_teacher_states,
    summarize_states,
    write_consistency_prompts,
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


def test_collect_bounds_episode_ticks_to_requested_quota(tmp_path, monkeypatch):
    ticks_seen = []

    def fake_export(root, seed, agents, ticks, out, *, profile="healthy", fraction=25):
        ticks_seen.append(ticks)
        rows = [state(seed + index) for index in range(agents)]
        out.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    monkeypatch.setattr("mw_training.spark_dataset._run_export", fake_export)
    report = collect_states(tmp_path / "states.jsonl", states=4, agents=2, ticks_per_seed=300)
    assert ticks_seen == [1, 1, 1, 1]
    assert report["plan"]["healthy"]["episode_ticks"] == 1


def test_collect_rejects_pathological_plan_before_export(tmp_path, monkeypatch):
    def fail_export(*args, **kwargs):
        raise AssertionError("pathological plan reached simulator")

    monkeypatch.setattr("mw_training.spark_dataset._run_export", fail_export)
    with pytest.raises(ValueError, match="collection plan exceeds"):
        collect_states(tmp_path / "states.jsonl", states=100_000, agents=50)

def test_label_checkpoint_retries_429_and_resumes_exactly_once(tmp_path: Path, monkeypatch):
    rows = [state(seed) for seed in range(1, 5)]
    prompts = tmp_path / "prompts.jsonl"
    labels = tmp_path / "labels.jsonl"
    assembled = tmp_path / "assembled.jsonl"
    prompt_rows = [
        {"id": f"{row['seed']}:{row['tick']}:{row['agent_slot']}", "prompt": "label this"}
        for row in rows
    ]
    write_jsonl(prompts, prompt_rows)
    first_id = prompt_rows[0]["id"]
    labels.write_text(
        json.dumps({"id": first_id, "tool": "move", "why": "existing"}) + "\n",
        encoding="utf-8",
    )

    class RateLimited(RuntimeError):
        status_code = 429
        headers = {"Retry-After": "0"}

    class FakeCaller:
        supports_batch = True

        def __init__(self, interrupt: bool, rate_limit: bool = True):
            self.calls = []
            self.interrupt = interrupt
            self.rate_limited = not rate_limit

        def __call__(self, batch, model=None):
            ids = [item["id"] for item in batch]
            self.calls.append(ids)
            if not self.rate_limited:
                self.rate_limited = True
                raise RateLimited("HTTP 429")
            if self.interrupt and ids == [prompt_rows[-1]["id"]]:
                raise RuntimeError("interrupted")
            return [{"id": item["id"], "tool": "move"} for item in batch]

    delays = []
    monkeypatch.setattr("mw_training.spark_dataset.time.sleep", delays.append)
    first = FakeCaller(interrupt=True)
    try:
        label_with_spark(
            prompts,
            labels,
            caller=first,
            batch_size=2,
            retries=1,
            retry_backoff=0,
            retry_jitter=0,
            resume=True,
        )
    except RuntimeError as error:
        assert str(error) == "interrupted"
    assert first.calls == [
        [prompt_rows[1]["id"], prompt_rows[2]["id"]],
        [prompt_rows[1]["id"], prompt_rows[2]["id"]],
        [prompt_rows[3]["id"]],
    ]
    assert delays == [0.0]
    checkpoint = [json.loads(line) for line in labels.read_text().splitlines()]
    assert [item["id"] for item in checkpoint] == [first_id, *[row["id"] for row in prompt_rows[1:3]]]

    resumed = FakeCaller(interrupt=False, rate_limit=False)
    report = label_with_spark(
        prompts,
        labels,
        caller=resumed,
        batch_size=2,
        retries=1,
        retry_backoff=0,
        retry_jitter=0,
        resume=True,
    )
    final = [json.loads(line) for line in labels.read_text().splitlines()]
    ids = [item["id"] for item in final]
    assert report["written"] == 1
    assert len(ids) == len(set(ids)) == 4
    assert set(ids) == {row["id"] for row in prompt_rows}
    assert resumed.calls == [[prompt_rows[-1]["id"]]]

    states = tmp_path / "states.jsonl"
    write_jsonl(states, rows)
    assembly_report = assemble_labels(states, labels, assembled)
    assert assembly_report["kept"] == 4
def test_consistency_sampling_and_repeat_ids_are_deterministic(tmp_path: Path):
    rows = []
    for seed in range(1, 7):
        row = state(seed)
        row["afforded_mask"] = 1 << (seed % 3)
        row["persona"]["traits"][0] += seed
        rows.append(row)
    first = sample_teacher_states(rows, 4, seed=7)
    second = sample_teacher_states(list(reversed(rows)), 4, seed=7)
    assert [_state["seed"] for _state in first] == [_state["seed"] for _state in second]

    states = tmp_path / "states.jsonl"
    prompts = tmp_path / "prompts.jsonl"
    write_jsonl(states, first)
    report = write_consistency_prompts(states, prompts, repeats=3)
    payloads = [json.loads(line) for line in prompts.read_text().splitlines()]
    first_id = f"{first[0]['seed']}:2:0"
    assert report["prompts"] == 12
    assert [item["id"] for item in payloads[:3]] == [f"{first_id}#r{repeat}" for repeat in range(3)]
    assert len({item["prompt"] for item in payloads[:3]}) == 1


def test_consistency_metrics_filter_illegal_labels_and_report_ambiguity(tmp_path: Path):
    rows = [state(4), state(5)]
    rows[1]["afforded_mask"] = (1 << 0) | (1 << 2)
    states = tmp_path / "states.jsonl"
    labels = tmp_path / "labels.jsonl"
    report_path = tmp_path / "report.json"
    write_jsonl(states, rows)
    write_jsonl(
        labels,
        [
            {"id": "4:2:0#r0", "tool": "move", "target": None, "arg": None},
            {"id": "4:2:0#r1", "tool": "move", "target": None, "arg": None},
            {"id": "5:2:0#r0", "tool": "move", "target": None, "arg": None},
            {"id": "5:2:0#r1", "tool": "sleep", "target": None, "arg": None},
            {"id": "5:2:0#r2", "tool": "idle", "target": None, "arg": None},
            {"id": "missing:0#r0", "tool": "move", "target": None, "arg": None},
        ],
    )
    report = analyze_teacher_consistency(states, labels, report_path)
    assert report["legal_labels"] == 4
    assert report["illegal_labels"] == 1
    assert report["unknown_labels"] == 1
    assert report["exact_tool_agreement"] == 0.5
    assert report["pairwise_agreement"] == 0.5
    assert report["per_state_action_entropy"]["4:2:0"] == 0.0
    assert report["per_state_action_entropy"]["5:2:0"] == 1.0
    assert report["class_counts"] == {"move": 3, "sleep": 1}
    assert report["ambiguity_slices"]["tool"]["move,sleep"]["ambiguous_states"] == 1
    assert json.loads(report_path.read_text())["legal_labels"] == 4


def test_consistency_sample_export_has_stable_checkpoint_inputs(tmp_path: Path):
    states = tmp_path / "states.jsonl"
    output = tmp_path / "sample.jsonl"
    write_jsonl(states, [state(seed) for seed in range(1, 5)])
    report = export_consistency_sample(states, output, count=2, seed=11)
    assert report["states"] == 2
    assert report["requested"] == 2
    exported = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["seed"] for row in exported] == sorted(row["seed"] for row in exported)


def replayable_state(seed: int = 4) -> dict:
    row = state(seed)
    row["afforded_mask"] = (1 << 0) | (1 << 1) | (1 << 11)
    row["state_snapshot"] = {
        "positions": [[2, 3], [1, 0]],
        "needs": [[200, 800, 700], [1000, 1000, 1000]],
        "inventory": [[1, 0], [0, 0]],
        "ground": [[0, 0], [0, 0]],
    }
    row["replay_provenance"] = {
        "version": 1,
        "profile": "healthy",
        "fraction": 25,
        "agent_count": 2,
        "initial_positions": [[0, 0], [1, 0]],
        "replay_log": [],
        "replay_hash": 1234,
    }
    return row


def test_grounding_retains_repeats_and_drops_malformed_without_fake_scores(tmp_path: Path):
    states, labels, output = (tmp_path / name for name in ("states", "labels", "out"))
    write_jsonl(states, [replayable_state()])
    write_jsonl(
        labels,
        [
            {"id": "4:2:0#r0", "tool": "eat", "arg": None, "why": "food"},
            {"id": "4:2:0#r1", "tool": "move", "arg": {"dx": 2, "dy": 0}, "why": "bad"},
            {"id": "4:2:0#r2", "tool": "idle", "arg": None, "why": "wait"},
        ],
    )
    report = ground_labels(states, labels, output)
    assert report["grounded_states"] == 1
    assert report["candidate_labels"] == 3
    row = json.loads(output.read_text().splitlines()[0])
    assert len(row["grounding"]["candidates"]) == 3
    assert row["grounding"]["candidates"][1]["legal"] is False
    assert row["grounding"]["candidates"][1]["score"] is None
    assert row["grounding"]["replay_hash"] == 1234


def test_grounding_ties_are_deterministic_and_byte_repeatable(tmp_path: Path):
    states, labels, first, second = (
        tmp_path / name for name in ("states", "labels", "first", "second")
    )
    write_jsonl(states, [replayable_state(9)])
    write_jsonl(
        labels,
        [
            {"id": "9:2:0#r1", "tool": "idle", "arg": None},
            {"id": "9:2:0#r0", "tool": "move", "arg": {"dx": 0, "dy": 0}},
        ],
    )
    first_report = ground_labels(states, labels, first)
    second_report = ground_labels(states, labels, second)
    assert first_report["grounded_states"] == second_report["grounded_states"] == 1
    assert first.read_bytes() == second.read_bytes()
    row = json.loads(first.read_text().splitlines()[0])
    assert row["grounding"]["expertise_rank"] == "capable"
    assert row["grounding"]["tie_break"].startswith("personality_only")


def test_grounding_reports_exact_replay_blocker_for_legacy_sample(tmp_path: Path):
    states, labels, output = (tmp_path / name for name in ("states", "labels", "out"))
    write_jsonl(states, [state()])
    write_jsonl(labels, [{"id": "4:2:0#r0", "tool": "move", "arg": None}])
    report = ground_labels(states, labels, output)
    assert report["grounded_states"] == 0
    assert report["coverage"] == 0.0
    assert report["blockers"]["missing_replay_provenance"] == 1


def _expertise_candidate(candidate_id: str, score: int, *, legal: bool = True, tool: str = "move") -> dict:
    return {
        "id": candidate_id,
        "repeat": 0,
        "tool": tool,
        "target_slot": None,
        "params": {},
        "rationale": None,
        "legal": legal,
        "error": None if legal else "tool_not_afforded",
        "score": score if legal else None,
        "components": {} if legal else None,
        "horizon": 8,
    }


def test_expertise_targets_use_objective_rank_not_prompt_or_action_order():
    record = replayable_state()
    record["persona"]["traits"] = [1, 2, 3, 4, 5]
    candidates = [
        _expertise_candidate("4:2:0#r0", 5_000, tool="move"),
        _expertise_candidate("4:2:0#r1", 100, tool="sleep"),
        _expertise_candidate("4:2:0#r2", 2_000, tool="idle"),
    ]
    result = _derive_expertise(record, candidates)
    targets = result["targets"]
    assert targets["expert"]["score"] == 5_000
    assert targets["capable"]["score"] >= targets["novice"]["score"]
    assert targets["novice"]["score"] < targets["capable"]["score"]
    assert all(target["legal"] for target in targets.values())
    assert all(target["id"].startswith("4:2:0#") for target in targets.values())


def test_expertise_ignores_illegal_high_score_and_rejects_prompt_expertise():
    record = replayable_state()
    record["expertise"] = "expert"
    candidates = [
        _expertise_candidate("4:2:0#r0", 100, tool="move"),
        _expertise_candidate("4:2:0#r1", 9_999_999, legal=False, tool="sleep"),
        _expertise_candidate("4:2:0#r2", 2_000, tool="idle"),
    ]
    result = _derive_expertise(record, candidates)
    assert result["targets"]["expert"]["score"] == 2_000
    assert all(target["legal"] for target in result["targets"].values())
    assert "expertise" not in result["contract"]


def test_expertise_ties_and_degenerate_sets_are_deterministic():
    record = replayable_state()
    tied = [
        _expertise_candidate("4:2:0#r1", 100, tool="sleep"),
        _expertise_candidate("4:2:0#r0", 100, tool="move"),
    ]
    first = _derive_expertise(record, tied)
    second = _derive_expertise(record, list(reversed(tied)))
    assert first["contract"]["degenerate_reason"] == "all_unique_legal_actions_tied"
    assert first["targets"]["expert"]["id"] == second["targets"]["expert"]["id"]
    assert first["targets"]["novice"]["score"] == first["targets"]["capable"]["score"]

    one = _derive_expertise(record, [_expertise_candidate("4:2:0#r0", 100)])
    assert one["contract"]["degenerate_reason"] == "only_one_unique_legal_action"
    assert {target["id"] for target in one["targets"].values()} == {"4:2:0#r0"}


def test_derive_expertise_reports_reproducible_same_state_coverage(tmp_path: Path):
    source = tmp_path / "grounded.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    row = replayable_state()
    row["grounding"] = {
        "horizon": 8,
        "candidates": [
            _expertise_candidate("4:2:0#r0", 5_000),
            _expertise_candidate("4:2:0#r1", 100, tool="sleep"),
        ],
    }
    write_jsonl(source, [row])
    first_report = derive_expertise_targets(source, first)
    second_report = derive_expertise_targets(source, second)
    first_row = json.loads(first.read_text().splitlines()[0])
    assert first_report["objective_order_violations"] == 0
    assert first_report["output_sha256"] == second_report["output_sha256"]
    assert first_row["grounding"]["expertise_contract"]["same_state"] is True
    assert set(first_row["grounding"]["expertise_targets"]) == {"novice", "capable", "expert"}


def test_materialize_matched_expertise_forms_deterministic_triplets(tmp_path: Path):
    source = tmp_path / "grounded.jsonl"
    reordered = tmp_path / "reordered.jsonl"
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    first_manifest = tmp_path / "first.manifest.json"
    second_manifest = tmp_path / "second.manifest.json"
    rows = []
    for seed in (4, 9):
        row = replayable_state(seed)
        prefix = f"{seed}:2:0"
        row["grounding"] = {
            "horizon": 8,
            "candidates": [
                _expertise_candidate(f"{prefix}#r0", 5_000),
                _expertise_candidate(f"{prefix}#r1", 100, tool="sleep"),
                _expertise_candidate(f"{prefix}#r2", 2_000, tool="idle"),
            ],
            "disagreement": {"legal_candidates": 3},
        }
        rows.append(row)
    write_jsonl(source, rows)
    write_jsonl(
        reordered,
        [
            dict(rows[1], grounding=dict(rows[1]["grounding"], candidates=list(reversed(rows[1]["grounding"]["candidates"])))),
            dict(rows[0], grounding=dict(rows[0]["grounding"], candidates=list(reversed(rows[0]["grounding"]["candidates"])))),
        ],
    )
    first_report = materialize_matched_expertise(source, first, first_manifest)
    second_report = materialize_matched_expertise(reordered, second, second_manifest)
    assert first.read_bytes() == second.read_bytes()
    assert first_report["output"]["sha256"] == second_report["output"]["sha256"]
    assert first_report["counts"]["records"] == 6
    assert first_report["assertions"]["objective_order_violations"] == 0
    assert first_report["assertions"]["complete_triplets"] is True
    records = [json.loads(line) for line in first.read_text().splitlines()]
    assert {record["expertise_level"] for record in records} == {"novice", "capable", "expert"}
    assert {record["matched_group_id"] for record in records} == {"4:2:0", "9:2:0"}
    assert all(record["expertise"]["one_hot_vector"] in ([1, 0, 0], [0, 1, 0], [0, 0, 1]) for record in records)
