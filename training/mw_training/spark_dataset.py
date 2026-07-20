"""Collect semantic trajectories, build Spark prompts, and assemble labels.

The simulator remains the source of truth for observations and affordances.  This
module only transports structured schema-v2 records to a teacher and applies a
small, deterministic kernel filter before writing records for ``train_omni``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import itertools
import json
import hashlib
import math
import os
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from typing import Callable, Iterable, Mapping, Sequence

from .dataset import TOOL_NAMES
from .persona import teacher_prompt_fragment

LOCATION_NAMES = ("empty", "home", "bakery", "well", "field")
NEED_BANDS = ("critical", "low", "medium", "high")

# Keep collection bounded even when callers retain the historical 300-tick
# default. The Rust exporter repeats each selected record's snapshot and replay
# prefix, so episode length must follow the number of requested samples.
MAX_COLLECTION_ESTIMATED_BYTES = 512 * 1024 * 1024
EXPORT_FIXED_BYTES = 16_384
EXPORT_AGENT_BYTES = 512
EXPORT_RECORD_BYTES = 64


def _state_id(record: Mapping) -> str:
    """Return a stable key independent of collection order."""
    return f"{int(record['seed'])}:{int(record['tick'])}:{int(record['agent_slot'])}"


def _persona_fragment(record: Mapping) -> str:
    traits = list(record["persona"]["traits"])
    # Rust's persisted order is aggression, sociability, industriousness, greed,
    # caution; the narrative contract is aggression, sociability, industriousness,
    # caution, greed.  Reorder at this boundary rather than changing schema-v2.
    narrative_traits = [traits[0], traits[1], traits[2], traits[4], traits[3]]
    return teacher_prompt_fragment(
        narrative_traits,
        record["persona"]["need_weights"],
        seed=int(record["seed"]) ^ (int(record["agent_slot"]) << 16),
    )


def _afforded_tools(record: Mapping) -> list[str]:
    mask = int(record["afforded_mask"])
    return [name for i, name in enumerate(TOOL_NAMES) if mask & (1 << i)]


def build_teacher_prompt(record: Mapping) -> str:
    """Build a strict JSON-only teacher request from one semantic state."""
    observation = {
        "tick": record["tick"],
        "agent_slot": record["agent_slot"],
        "persona": record["persona"],
        "observation": record["obs"],
        "afforded_tools": _afforded_tools(record),
    }
    return (
        "You are the village SOUL teacher. Choose one legal next intent for this "
        "state. Let the narrative persona influence the choice, but never invent "
        "a tool outside afforded_tools. Return exactly one JSON object and no "
        "markdown, using this schema: "
        '{"tool":"<afforded tool>","target":<neighbor id_slot or null>,'
        '"arg":<JSON scalar/object or null>,"why":"brief rationale"}.\n\n'
        f"{_persona_fragment(record)}\n"
        f"Allowed tools: {json.dumps(_afforded_tools(record), sort_keys=True)}\n"
        "Structured semantic state (do not flatten or omit fields):\n"
        f"{json.dumps(observation, sort_keys=True, separators=(',', ':'))}"
    )


def _need_band(record: Mapping) -> str:
    minimum = min(int(v) for v in record["obs"]["self_stats"])
    if minimum < 250:
        return "critical"
    if minimum < 500:
        return "low"
    if minimum < 750:
        return "medium"
    return "high"


def summarize_states(records: Iterable[Mapping]) -> dict[str, object]:
    """Summarize semantic diversity without depending on labels."""
    rows = list(records)
    locations = Counter(
        LOCATION_NAMES[int(r["obs"]["self_cell_class"])]
        if 0 <= int(r["obs"]["self_cell_class"]) < len(LOCATION_NAMES)
        else "unknown"
        for r in rows
    )
    needs = Counter(_need_band(r) for r in rows)
    threats = Counter(
        "present"
        if any(n["present"] and int(n["opinion"]) < 0 for n in r["obs"]["neighbors"])
        else "absent"
        for r in rows
    )
    personas = len({_persona_fragment(r).split("\n", 1)[0] for r in rows})
    return {
        "states": len(rows),
        "need_bands": dict(sorted(needs.items(), key=lambda x: NEED_BANDS.index(x[0]))),
        "locations": dict(sorted(locations.items())),
        "threat_presence": dict(sorted(threats.items())),
        "persona_sketches": personas,
    }


def _run_export(
    repo_root: Path,
    seed: int,
    agents: int,
    ticks: int,
    out: Path,
    *,
    profile: str = "healthy",
    fraction: int = 25,
) -> None:
    # CI and local development normally use cargo; MW_SIM_BIN permits a
    # prebuilt simulator when dependencies are being rebuilt concurrently.
    binary = os.environ.get("MW_SIM_BIN")
    command = (
        [binary, "trajectories"]
        if binary
        else ["cargo", "run", "-q", "-p", "mw-sim", "--", "trajectories"]
    )
    command += [
        "--seed",
        str(seed),
        "--agents",
        str(agents),
        "--ticks",
        str(ticks),
        "--out",
        str(out),
        "--habits",
        "off",
        "--profile",
        profile,
        "--fraction",
        str(fraction),
    ]
    subprocess.run(command, cwd=repo_root, check=True)


def _sample_profile(exported: list[dict], count: int, profile: str, agents: int) -> list[dict]:
    if len(exported) <= count:
        return exported
    if profile != "healthy":
        # Stress is intentionally visible at the start. Keep the first few
        # ticks while still spreading across every starting agent.
        horizon = min(len(exported), max(count, agents * 8))
        exported = exported[:horizon]
    if count == 1:
        return [exported[0]]
    indexes = [
        round(i * (len(exported) - 1) / (count - 1))
        for i in range(count)
    ]
    return [exported[index] for index in indexes]


def _bounded_ticks(quota: int, agents: int, requested_ticks: int) -> int:
    """Cap one simulator episode at the ticks needed to fill its quota."""
    if quota <= 0:
        return 0
    return max(1, min(requested_ticks, (quota + agents - 1) // agents))


def _estimated_export_bytes(agents: int, ticks: int) -> int:
    """Mirror the Rust export preflight with a conservative upper bound."""
    records = agents * ticks
    per_record = (
        EXPORT_FIXED_BYTES
        + agents * EXPORT_AGENT_BYTES
        + records * EXPORT_RECORD_BYTES
    )
    return records * per_record


def _sample_profile_file(
    path: Path,
    count: int,
    profile: str,
    agents: int,
) -> list[dict]:
    """Select from JSONL in two bounded passes, without loading an episode."""
    with path.open(encoding="utf-8") as stream:
        total = sum(1 for line in stream if line.strip())
    if total == 0:
        raise RuntimeError(f"simulator exported no records for {profile}")
    horizon = total
    if profile != "healthy":
        # Stress is intentionally visible at the start while still spreading
        # across every starting agent.
        horizon = min(total, max(count, agents * 8))
    selected_count = min(count, horizon)
    if selected_count == 1:
        indexes = {0}
    else:
        indexes = {
            round(i * (horizon - 1) / (selected_count - 1))
            for i in range(selected_count)
        }
    selected: list[dict] = []
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= horizon:
                break
            if index in indexes and line.strip():
                selected.append(json.loads(line))
    return selected


def collect_states(
    output: str | Path,
    *,
    states: int = 1_000,
    agents: int = 50,
    seed: int = 1,
    ticks_per_seed: int = 300,
    repo_root: str | Path | None = None,
) -> dict[str, object]:
    """Collect an even mix of healthy and deterministic stress-start states."""
    if states < 1 or agents < 1 or ticks_per_seed < 1:
        raise ValueError("states, agents, and ticks_per_seed must be positive")
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[2]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    profiles = ("healthy", "scarcity", "hostile", "exhausted")
    base, remainder = divmod(states, len(profiles))
    quotas = [base + (index < remainder) for index in range(len(profiles))]
    plan: dict[str, dict[str, int]] = {}
    estimated_temp_bytes = 0
    for profile, quota in zip(profiles, quotas):
        episode_ticks = _bounded_ticks(quota, agents, ticks_per_seed)
        records_per_run = agents * episode_ticks
        runs = (
            0
            if quota == 0
            else max(1, (quota + records_per_run - 1) // records_per_run)
        )
        estimated_bytes = runs * _estimated_export_bytes(agents, episode_ticks)
        plan[profile] = {
            "quota": quota,
            "episode_ticks": episode_ticks,
            "runs": runs,
            "estimated_bytes": estimated_bytes,
        }
        estimated_temp_bytes += estimated_bytes
    if estimated_temp_bytes > MAX_COLLECTION_ESTIMATED_BYTES:
        raise ValueError(
            "collection plan exceeds "
            f"{MAX_COLLECTION_ESTIMATED_BYTES // (1024 * 1024)} MiB "
            f"(estimated_temp_bytes={estimated_temp_bytes})"
        )

    rows: list[dict] = []
    profile_counts: dict[str, int] = {}
    actual_temp_bytes = 0
    with tempfile.TemporaryDirectory(prefix="mw-spark-") as temporary:
        for profile_index, (profile, quota) in enumerate(zip(profiles, quotas)):
            selected: list[dict] = []
            run = 0
            while len(selected) < quota:
                run_seed = seed + profile_index * 1_000_003 + run * 7919
                path = Path(temporary) / f"trajectory-{profile}-{run}.jsonl"
                profile_fraction = 50 if profile == "hostile" else 25
                episode_ticks = plan[profile]["episode_ticks"]
                _run_export(
                    root,
                    run_seed,
                    agents,
                    episode_ticks,
                    path,
                    profile=profile,
                    fraction=profile_fraction,
                )
                actual_temp_bytes += path.stat().st_size
                need = quota - len(selected)
                selected.extend(_sample_profile_file(path, need, profile, agents))
                run += 1
            rows.extend(selected[:quota])
            profile_counts[profile] = quota
    ids = [_state_id(record) for record in rows]
    if len(ids) != len(set(ids)) or len(rows) != states:
        raise RuntimeError("collector produced duplicate or incomplete state ids")
    with destination.open("w", encoding="utf-8") as stream:
        for record in rows:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    output_bytes = destination.stat().st_size
    report = summarize_states(rows)
    report["seeds"] = len({int(r["seed"]) for r in rows})
    report["profiles"] = profile_counts
    report["plan"] = plan
    report["estimated_temp_bytes"] = estimated_temp_bytes
    report["temp_bytes"] = actual_temp_bytes
    report["output_bytes"] = output_bytes
    return report


def write_prompts(states_path: str | Path, prompts_path: str | Path) -> dict[str, object]:
    """Write one Spark prompt plus enough state metadata to join labels later."""
    records = list(_iter_jsonl(states_path))
    destination = Path(prompts_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            payload = {
                "id": _state_id(record),
                "prompt": build_teacher_prompt(record),
                "state": {
                    "seed": record["seed"],
                    "tick": record["tick"],
                    "agent_slot": record["agent_slot"],
                    "persona": record["persona"],
                    "obs": record["obs"],
                    "afforded_mask": record["afforded_mask"],
                    "afforded_tools": _afforded_tools(record),
                    "replay_provenance": record.get("replay_provenance"),
                    "state_snapshot": record.get("state_snapshot"),
                },
            }
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"prompts": len(records), "path": str(destination)}
def _persona_key(record: Mapping) -> str:
    return json.dumps(record.get("persona", {}), sort_keys=True, separators=(",", ":"))


def _regime_key(record: Mapping) -> str:
    explicit = record.get("regime", record.get("profile"))
    if explicit is not None:
        return str(explicit)
    location = int(record["obs"].get("self_cell_class", -1))
    location_name = (
        LOCATION_NAMES[location] if 0 <= location < len(LOCATION_NAMES) else "unknown"
    )
    threatened = any(
        neighbor.get("present") and int(neighbor.get("opinion", 0)) < 0
        for neighbor in record["obs"].get("neighbors", [])
    )
    return f"{_need_band(record)}:{location_name}:{'threat' if threatened else 'safe'}"


def _consistency_strata(record: Mapping) -> tuple[str, str, str]:
    tools = ",".join(_afforded_tools(record)) or "none"
    return tools, _regime_key(record), _persona_key(record)


def sample_teacher_states(
    records: Iterable[Mapping],
    count: int,
    *,
    seed: int = 0,
) -> list[dict]:
    """Select a deterministic round-robin sample across semantic strata.

    Strata are the afforded-tool set, deterministic regime sketch, and persona.
    Records are sorted by their stable state id before allocation, so collection
    order cannot change the sample.  If there are fewer records than requested,
    every record is returned; no synthetic state or label is created.
    """
    if count < 1:
        raise ValueError("count must be positive")
    rows = [dict(record) for record in records]
    rows.sort(key=_state_id)
    ids = [_state_id(record) for record in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("states contain duplicate stable ids")
    if len(rows) <= count:
        return rows
    strata: dict[tuple[str, str, str], list[dict]] = {}
    for row in rows:
        strata.setdefault(_consistency_strata(row), []).append(row)
    keys = sorted(strata)
    rotation = seed % len(keys)
    keys = keys[rotation:] + keys[:rotation]
    selected: list[dict] = []
    positions = {key: 0 for key in keys}
    while len(selected) < count:
        for key in keys:
            position = positions[key]
            if position >= len(strata[key]):
                continue
            selected.append(strata[key][position])
            positions[key] = position + 1
            if len(selected) == count:
                break
    return sorted(selected, key=_state_id)


def export_consistency_sample(
    states_path: str | Path,
    output: str | Path,
    *,
    count: int,
    seed: int = 0,
) -> dict[str, object]:
    """Write the deterministic teacher-ceiling sample as JSONL."""
    selected = sample_teacher_states(_iter_jsonl(states_path), count, seed=seed)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    return {
        "states": len(selected),
        "requested": count,
        "unique_world_seeds": len({int(row["seed"]) for row in selected}),
        "strata": len({_consistency_strata(row) for row in selected}),
        "path": str(destination),
    }


def write_consistency_prompts(
    states_path: str | Path,
    prompts_path: str | Path,
    *,
    repeats: int = 3,
) -> dict[str, object]:
    """Emit stable, repeated prompts for self-consistency measurement.

    Every repeat has identical semantic prompt text.  Only the transport id and
    repeat metadata differ, allowing stochastic teacher calls without injecting
    an artificial instruction into the teacher distribution.
    """
    if repeats < 2:
        raise ValueError("repeats must be at least two for self-consistency")
    records = list(_iter_jsonl(states_path))
    ids = [_state_id(record) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("states contain duplicate stable ids")
    destination = Path(prompts_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            state_id = _state_id(record)
            prompt = build_teacher_prompt(record)
            state = {
                "seed": record["seed"],
                "tick": record["tick"],
                "agent_slot": record["agent_slot"],
                "persona": record["persona"],
                "obs": record["obs"],
                "afforded_mask": record["afforded_mask"],
                "afforded_tools": _afforded_tools(record),
                "replay_provenance": record.get("replay_provenance"),
                "state_snapshot": record.get("state_snapshot"),
            }
            for repeat in range(repeats):
                payload = {
                    "id": f"{state_id}#r{repeat}",
                    "state_id": state_id,
                    "repeat": repeat,
                    "prompt": prompt,
                    "state": state,
                }
                stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
                count += 1
    return {
        "prompts": count,
        "states": len(records),
        "repeats": repeats,
        "path": str(destination),
    }


def _entropy(values: Iterable[object]) -> float:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values() if count
    )


def _consistency_state_id(label: Mapping) -> str:
    explicit = label.get("state_id")
    if explicit is None and isinstance(label.get("consistency"), Mapping):
        explicit = label["consistency"].get("state_id")
    if explicit is not None:
        return str(explicit)
    key = str(label.get("id", ""))
    match = re.fullmatch(r"(.+?)(?:#r|:repeat:)(\d+)", key)
    return match.group(1) if match else key


def _action_key(fields: Mapping) -> str:
    return json.dumps(
        {
            "tool": str(fields.get("tool", fields.get("intent", ""))).strip().lower(),
            "target": fields.get("target"),
            "arg": fields.get("arg"),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _slice_stats(
    groups: Mapping[str, Sequence[tuple[str, Mapping]]],
) -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    for category, entries in sorted(groups.items()):
        by_state: dict[str, list[str]] = {}
        for state_id, action in entries:
            by_state.setdefault(state_id, []).append(_action_key(action))
        entropies = [_entropy(actions) for actions in by_state.values()]
        ambiguous = sum(len(set(actions)) > 1 for actions in by_state.values())
        result[category] = {
            "states": len(by_state),
            "ambiguous_states": ambiguous,
            "ambiguity_rate": ambiguous / len(by_state) if by_state else 0.0,
            "mean_action_entropy": sum(entropies) / len(entropies) if entropies else 0.0,
        }
    return result


def analyze_teacher_consistency(
    states_path: str | Path,
    labels_path: str | Path,
    output: str | Path | None = None,
) -> dict[str, object]:
    """Analyze repeated labels after applying the same tool legality contract.

    Unknown states and labels whose tool is not afforded are excluded from
    agreement metrics and counted explicitly.  This function never invents a
    missing label or treats an illegal response as a legal class.
    """
    states = {_state_id(row): row for row in _iter_jsonl(states_path)}
    if len(states) == 0:
        raise ValueError("consistency analysis requires at least one state")
    labels = list(_iter_jsonl(labels_path))
    groups: dict[str, list[tuple[str, Mapping]]] = {}
    seen_ids: set[str] = set()
    illegal = unknown = malformed = 0
    class_counts: Counter[str] = Counter()
    for label in labels:
        label_id = str(label.get("id", ""))
        if not label_id:
            malformed += 1
            continue
        if label_id in seen_ids:
            raise ValueError(f"labels contain duplicate id {label_id!r}")
        seen_ids.add(label_id)
        state_id = _consistency_state_id(label)
        record = states.get(state_id)
        if record is None:
            unknown += 1
            continue
        try:
            fields = _label_fields(label)
            tool = str(fields.get("tool", fields.get("intent", ""))).strip().lower()
        except ValueError:
            malformed += 1
            continue
        if tool not in _afforded_tools(record):
            illegal += 1
            continue
        action = dict(fields)
        action["tool"] = tool
        groups.setdefault(state_id, []).append((label_id, action))
        class_counts[tool] += 1

    state_entropies: dict[str, float] = {}
    tool_entropies: dict[str, float] = {}
    legal_action_counts: dict[str, int] = {}
    observed_legal_action_counts: dict[str, int] = {}
    exact_tool_numerator = exact_tool_denominator = 0
    pairwise_action_matches = pairwise_action_total = 0
    pairwise_tool_matches = 0
    for state_id, entries in sorted(groups.items()):
        actions = [_action_key(action) for _, action in entries]
        tools = [str(action["tool"]) for _, action in entries]
        state_entropies[state_id] = _entropy(actions)
        tool_entropies[state_id] = _entropy(tools)
        observed_legal_action_counts[state_id] = len(set(actions))
        legal_action_counts[state_id] = len(_afforded_tools(states[state_id]))
        if len(entries) >= 2:
            exact_tool_denominator += 1
            exact_tool_numerator += int(len(set(tools)) == 1)
            for left, right in itertools.combinations(range(len(entries)), 2):
                pairwise_action_total += 1
                pairwise_action_matches += int(actions[left] == actions[right])
                pairwise_tool_matches += int(tools[left] == tools[right])

    legal_labels = sum(class_counts.values())
    class_distribution = {
        tool: count / legal_labels for tool, count in sorted(class_counts.items())
    }
    count_distribution = Counter(legal_action_counts.values())
    report: dict[str, object] = {
        "states": len(states),
        "labels": len(labels),
        "legal_labels": legal_labels,
        "illegal_labels": illegal,
        "unknown_labels": unknown,
        "malformed_labels": malformed,
        "states_with_labels": len(groups),
        "states_with_repeats": sum(len(entries) >= 2 for entries in groups.values()),
        "missing_states": len(states) - len(groups),
        "exact_tool_agreement": (
            exact_tool_numerator / exact_tool_denominator
            if exact_tool_denominator
            else 0.0
        ),
        "tool_agreement": (
            exact_tool_numerator / exact_tool_denominator
            if exact_tool_denominator
            else 0.0
        ),
        "exact_tool_agreement_states": exact_tool_denominator,
        "pairwise_agreement": (
            pairwise_action_matches / pairwise_action_total
            if pairwise_action_total
            else 0.0
        ),
        "pairwise_tool_agreement": (
            pairwise_tool_matches / pairwise_action_total
            if pairwise_action_total
            else 0.0
        ),
        "pairwise_comparisons": pairwise_action_total,
        "per_state_action_entropy": state_entropies,
        "state_action_entropy": state_entropies,
        "per_state_tool_entropy": tool_entropies,
        "mean_action_entropy": (
            sum(state_entropies.values()) / len(state_entropies)
            if state_entropies
            else 0.0
        ),
        "legal_action_counts": legal_action_counts,
        "legal_action_count": legal_action_counts,
        "observed_legal_action_counts": observed_legal_action_counts,
        "mean_legal_action_count": (
            sum(legal_action_counts.values()) / len(legal_action_counts)
            if legal_action_counts
            else 0.0
        ),
        "legal_action_count_distribution": dict(sorted(count_distribution.items())),
        "legal_action_count_entropy": _entropy(legal_action_counts.values()),
        "observed_legal_action_count_entropy": _entropy(
            observed_legal_action_counts.values()
        ),
        "legal_action_entropy": (
            sum(state_entropies.values()) / len(state_entropies)
            if state_entropies
            else 0.0
        ),
        "class_counts": dict(sorted(class_counts.items())),
        "class_distribution": class_distribution,
    }
    dimensions: dict[str, dict[str, list[tuple[str, Mapping]]]] = {
        "tool": {},
        "regime": {},
        "persona": {},
    }
    for state_id in groups:
        record = states[state_id]
        categories = {
            "tool": ",".join(_afforded_tools(record)) or "none",
            "regime": _regime_key(record),
            "persona": _persona_key(record),
        }
        for dimension, category in categories.items():
            dimensions[dimension].setdefault(category, []).extend(
                (state_id, action) for _, action in groups[state_id]
            )
    report["ambiguity_slices"] = {
        dimension: _slice_stats(groups_by_category)
        for dimension, groups_by_category in dimensions.items()
    }
    if output is not None:
        destination = Path(output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        report["path"] = str(destination)
    return report


# Short aliases keep the public flow discoverable for callers that use
# "sample/export/analyze" terminology.
sample_consistency = sample_teacher_states
export_consistency_prompts = write_consistency_prompts
analyze_consistency = analyze_teacher_consistency




def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{number}") from exc


def _decode_response_items(output: object) -> list[dict]:
    """Decode one or more JSON teacher responses.

    The companion CLI emits text, while test and production adapters may return
    already-decoded mappings or a list of mappings for a batch.  Logs and code
    fences around JSON are tolerated, but every returned item is retained for
    independent validation by ``_validate_label_items``.
    """
    if isinstance(output, Mapping):
        for key in ("labels", "items", "results"):
            if key in output:
                return _decode_response_items(output[key])
        return [dict(output)]
    if isinstance(output, str):
        text = output.strip()
        if text.startswith("```") and text.endswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:-1]).strip()
        try:
            return _decode_response_items(json.loads(text))
        except (json.JSONDecodeError, ValueError):
            decoded: list[dict] = []
            for line in output.splitlines():
                candidate = line.strip().strip("`")
                if not candidate:
                    continue
                try:
                    value = json.loads(candidate)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, (Mapping, list)):
                    decoded.extend(_decode_response_items(value))
            if decoded:
                return decoded
        raise ValueError("Spark response did not contain JSON label objects")
    if isinstance(output, Sequence) and not isinstance(output, (bytes, bytearray)):
        decoded = []
        for value in output:
            if not isinstance(value, Mapping):
                raise ValueError("Spark batch response contained a non-object item")
            decoded.append(dict(value))
        return decoded
    raise ValueError("Spark response did not contain JSON label objects")


def _decode_label(output: str) -> dict:
    """Decode one Spark JSON response, tolerating a surrounding code fence."""
    items = _decode_response_items(output)
    if len(items) != 1:
        raise ValueError(f"Spark response contained {len(items)} label objects, expected one")
    return items[0]


def _validate_label_items(output: object, requested: Sequence[Mapping]) -> list[dict]:
    """Parse and validate every response item against one requested batch."""
    items = _decode_response_items(output)
    expected = {str(item["id"]) for item in requested}
    if not items:
        raise ValueError("Spark response contained no label objects")
    seen: set[str] = set()
    validated: list[dict] = []
    for item in items:
        label = dict(item)
        key = label.get("id")
        if key is None and len(expected) == 1:
            key = next(iter(expected))
        if key is None:
            raise ValueError("every batched Spark label must contain an id")
        key = str(key)
        if key not in expected:
            raise ValueError(f"Spark returned label for unknown id {key!r}")
        if key in seen:
            raise ValueError(f"Spark returned duplicate label id {key!r}")
        fields = _label_fields(label)
        if not any(name in fields for name in ("tool", "intent")):
            raise ValueError(f"Spark label {key!r} has no tool or intent")
        if not str(fields.get("tool", fields.get("intent", ""))).strip():
            raise ValueError(f"Spark label {key!r} has an empty tool or intent")
        seen.add(key)
        label["id"] = key
        validated.append(label)
    missing = expected - seen
    if missing:
        raise ValueError(f"Spark response omitted label ids: {', '.join(sorted(missing))}")
    return validated


class _RateLimitError(RuntimeError):
    """Internal response wrapper retaining a server retry hint."""

    def __init__(self, message: str, retry_after: object = None):
        super().__init__(message)
        self.retry_after = retry_after


def _retry_after_seconds(error: BaseException) -> float | None:
    """Read a numeric Retry-After hint from common exception/response shapes."""
    candidates = [error, getattr(error, "response", None)]
    for candidate in candidates:
        if candidate is None:
            continue
        value = getattr(candidate, "retry_after", None)
        if value is None:
            headers = getattr(candidate, "headers", None)
            if isinstance(headers, Mapping):
                for name, header_value in headers.items():
                    if str(name).lower() == "retry-after":
                        value = header_value
                        break
        if value is None and isinstance(candidate, Mapping):
            for name, header_value in candidate.items():
                if str(name).lower() in {"retry-after", "retry_after"}:
                    value = header_value
                    break
        if value is None:
            continue
        try:
            return max(0.0, float(value))
        except (TypeError, ValueError):
            continue
    return None


def _is_rate_limit_error(error: BaseException) -> bool:
    for candidate in (error, getattr(error, "response", None)):
        if candidate is None:
            continue
        for name in ("status_code", "status", "http_status", "code"):
            try:
                if int(getattr(candidate, name)) == 429:
                    return True
            except (AttributeError, TypeError, ValueError):
                pass
        text = " ".join(
            str(getattr(candidate, name, "")) for name in ("stdout", "stderr", "message")
        )
        text += f" {candidate}"
        if re.search(r"\b429\b|rate[\s_-]*limit|too many requests|throttl", text, re.I):
            return True
    return False


def _retry_delay(
    error: BaseException,
    attempt: int,
    *,
    backoff: float,
    max_backoff: float,
    jitter: float,
) -> float:
    base = min(max_backoff, backoff * (2**attempt))
    hint = _retry_after_seconds(error)
    delay = max(base, hint or 0.0)
    return min(max_backoff, delay + random.uniform(0.0, max(0.0, jitter)))


def _adapter_supports_batch(caller: Callable) -> bool:
    marker = getattr(caller, "supports_batch", None)
    if marker is not None:
        return bool(marker)
    return True


def _call_adapter(caller: Callable, batch: Sequence[Mapping], model: str) -> object:
    """Call a batch-capable adapter using a stable structured interface."""
    method = getattr(caller, "call_batch", None) if len(batch) > 1 else None
    if method is None:
        method = getattr(caller, "call", None) or caller
    try:
        return method(batch, model=model)
    except TypeError as exc:
        # Small fakes and existing adapters often expose only ``batch``.
        if "model" not in str(exc):
            raise
        return method(batch)


def _subprocess_call(
    batch: Sequence[Mapping], *, script: Path, model: str
) -> object:
    if len(batch) != 1:
        raise ValueError("the sanctioned CLI adapter does not support batched prompts")
    item = batch[0]
    result = subprocess.run(
        [
            "node",
            str(script),
            "task",
            "--model",
            model,
            item["prompt"],
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _load_checkpoint(path: Path) -> dict[str, dict]:
    existing: dict[str, dict] = {}
    if not path.is_file():
        return existing
    for label in _iter_jsonl(path):
        if not isinstance(label, Mapping):
            raise ValueError(f"invalid label object in checkpoint {path}")
        key = label.get("id")
        if key is None:
            raise ValueError(f"checkpoint label in {path} has no id")
        key = str(key)
        if key in existing:
            raise ValueError(f"checkpoint contains duplicate label id {key!r}")
        existing[key] = dict(label)
    return existing


def label_with_spark(
    prompts_path: str | Path,
    labels_path: str | Path,
    *,
    model: str = "gpt-5.3-codex-spark",
    limit: int | None = None,
    batch_size: int = 1,
    concurrency: int = 1,
    retries: int = 5,
    retry_backoff: float = 1.0,
    max_backoff: float = 60.0,
    retry_jitter: float = 0.25,
    resume: bool = False,
    caller: Callable | None = None,
) -> dict[str, object]:
    """Label prompts with incremental, resumable, rate-limit-aware checkpoints.

    ``caller`` is an optional adapter accepting ``(batch, model=...)`` and
    returning one JSON object per requested item (as a mapping, list, or JSON
    string).  Adapters are considered batch-capable unless they expose
    ``supports_batch = False``; a ``call_batch`` method is also supported.
    """
    if batch_size < 1 or concurrency < 1 or retries < 0:
        raise ValueError("batch_size and concurrency must be positive; retries cannot be negative")
    if retry_backoff < 0 or max_backoff < 0 or retry_jitter < 0:
        raise ValueError("retry backoff values cannot be negative")
    prompts = list(_iter_jsonl(prompts_path))
    selected = prompts if limit is None else prompts[: max(0, limit)]
    selected_ids = [str(item["id"]) for item in selected]
    if len(selected_ids) != len(set(selected_ids)):
        raise ValueError("prompts contain duplicate label ids")
    destination = Path(labels_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = _load_checkpoint(destination) if resume else {}
    pending = [item for item in selected if str(item["id"]) not in existing]

    script: Path | None = None
    if caller is None:
        plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
        if not plugin_root:
            raise RuntimeError(
                "Spark unavailable: CLAUDE_PLUGIN_ROOT is unset; run the labeling "
                "batch with the sanctioned codex-companion CLI"
            )
        script = Path(plugin_root) / "scripts" / "codex-companion.mjs"
        if not script.is_file():
            raise RuntimeError(f"Spark unavailable: missing sanctioned CLI at {script}")

    supports_batch = caller is not None and _adapter_supports_batch(caller)
    effective_batch_size = batch_size if supports_batch else 1
    batches = [
        pending[index : index + effective_batch_size]
        for index in range(0, len(pending), effective_batch_size)
    ]
    written = 0
    written_ids = set(existing)
    mode = "a" if resume else "w"

    def request(batch: Sequence[Mapping]) -> list[dict]:
        for attempt in range(retries + 1):
            try:
                if caller is None:
                    response = _subprocess_call(batch, script=script, model=model)
                else:
                    response = _call_adapter(caller, batch, model)
                if isinstance(response, Mapping):
                    status = response.get("status_code", response.get("status"))
                    try:
                        rate_limited = int(status) == 429
                    except (TypeError, ValueError):
                        rate_limited = False
                    if rate_limited:
                        hint = next(
                            (
                                value
                                for name, value in response.items()
                                if str(name).lower() in {"retry-after", "retry_after"}
                            ),
                            None,
                        )
                        raise _RateLimitError("Spark adapter returned HTTP 429", hint)
                return _validate_label_items(response, batch)
            except BaseException as error:
                if not _is_rate_limit_error(error) or attempt >= retries:
                    raise
                time.sleep(
                    _retry_delay(
                        error,
                        attempt,
                        backoff=retry_backoff,
                        max_backoff=max_backoff,
                        jitter=retry_jitter,
                    )
                )
        raise AssertionError("unreachable retry loop")

    def append_labels(stream, labels: Sequence[Mapping]) -> int:
        count = 0
        for label in labels:
            key = str(label["id"])
            if key in written_ids:
                raise ValueError(f"duplicate label id {key!r} would be written")
            line = json.dumps({"id": key, **dict(label)}, separators=(",", ":")) + "\n"
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
            written_ids.add(key)
            count += 1
        return count

    with destination.open(mode, encoding="utf-8") as stream:
        if concurrency == 1:
            for batch in batches:
                written += append_labels(stream, request(batch))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as executor:
                futures = [executor.submit(request, batch) for batch in batches]
                try:
                    for future in as_completed(futures):
                        written += append_labels(stream, future.result())
                except BaseException:
                    for future in futures:
                        future.cancel()
                    raise
    return {
        "labels": len(written_ids),
        "written": written,
        "skipped": len(selected) - len(pending),
        "path": str(destination),
        "model": model,
        "batches": len(batches),
    }


def _label_fields(label: Mapping) -> Mapping:
    value = label.get("label", label.get("intent", label))
    if not isinstance(value, Mapping):
        raise ValueError("Spark label must be an object")
    return value


def _target_slot(record: Mapping, target: object) -> int | None:
    if target is None:
        return None
    if isinstance(target, str) and target.isdigit():
        target = int(target)
    if isinstance(target, int):
        for index, neighbor in enumerate(record["obs"]["neighbors"]):
            if neighbor.get("present") and neighbor.get("id_slot") == target:
                return target
        if 0 <= target < len(record["obs"]["neighbors"]):
            neighbor = record["obs"]["neighbors"][target]
            return neighbor.get("id_slot") if neighbor.get("present") else None
    return None

GROUNDING_HORIZON = 8
_NEED_DECAY = (2, 1, 3)
_MAX_NEED = 1000
_ACTION_RESOURCE_DELTA = {
    "eat": -1,
    "use": -1,
    "pickup": 1,
    "drop": -1,
    "give": -1,
}


def _base_label_id(label_id: object) -> tuple[str, int | None]:
    """Split ``seed:tick:slot#rN`` while preserving repeat provenance."""
    value = str(label_id)
    match = re.fullmatch(r"(.+)#r([0-9]+)", value)
    if match:
        return match.group(1), int(match.group(2))
    return value, None


def _grounding_state_ready(record: Mapping) -> tuple[bool, str | None]:
    provenance = record.get("replay_provenance")
    snapshot = record.get("state_snapshot")
    if not isinstance(provenance, Mapping):
        return False, "missing_replay_provenance"
    required = ("version", "profile", "fraction", "agent_count", "initial_positions", "replay_log", "replay_hash")
    missing = [name for name in required if name not in provenance]
    if missing:
        return False, "missing_replay_fields:" + ",".join(missing)
    if not isinstance(snapshot, Mapping):
        return False, "missing_state_snapshot"
    if not all(name in snapshot for name in ("positions", "needs", "inventory", "ground")):
        return False, "incomplete_state_snapshot"
    slot = int(record["agent_slot"])
    try:
        lengths = {len(snapshot[name]) for name in ("positions", "needs", "inventory", "ground")}
    except TypeError:
        return False, "malformed_state_snapshot"
    if len(lengths) != 1 or not lengths or slot < 0 or slot >= next(iter(lengths)):
        return False, "state_snapshot_slot_mismatch"
    if int(provenance["agent_count"]) != next(iter(lengths)):
        return False, "replay_agent_count_mismatch"
    actor_needs = [int(value) for value in snapshot["needs"][slot]]
    observed_needs = [int(value) for value in record["obs"].get("self_stats", ())]
    if actor_needs != observed_needs:
        return False, "snapshot_observation_need_mismatch"
    actor_pos = list(snapshot["positions"][slot])
    if actor_pos != list(record["obs"].get("self_pos", ())):
        return False, "snapshot_observation_position_mismatch"
    return True, None


def _candidate_params(fields: Mapping) -> dict:
    arg = fields.get("arg")
    if isinstance(arg, Mapping):
        return dict(arg)
    return {} if arg is None else {"arg": arg}


def _candidate_record(record: Mapping, label: Mapping, label_id: str) -> dict:
    fields = _label_fields(label)
    tool = str(fields.get("tool", fields.get("intent", ""))).strip().lower()
    params = _candidate_params(fields)
    target = _target_slot(record, fields.get("target"))
    legal = tool in _afforded_tools(record)
    error = None
    if not legal:
        error = "tool_not_afforded"
    elif tool == "move":
        try:
            dx, dy = int(params.get("dx", 0)), int(params.get("dy", 0))
        except (TypeError, ValueError):
            legal, error = False, "malformed_move_params"
        else:
            if abs(dx) > 1 or abs(dy) > 1:
                legal, error = False, "move_out_of_range"
            params = {"dx": dx, "dy": dy}
    elif tool in {"speak", "give", "follow", "flee"} and target is None:
        legal, error = False, "missing_or_invalid_target"
    return {
        "id": label_id,
        "repeat": _base_label_id(label_id)[1],
        "tool": tool,
        "target_slot": target,
        "params": params,
        "rationale": None if fields.get("why") is None else str(fields["why"]),
        "legal": legal,
        "error": error,
    }


def _rollout_score(record: Mapping, candidate: Mapping, horizon: int = GROUNDING_HORIZON) -> dict:
    """Score one legal candidate from the exported, identical pre-action snapshot.

    This mirrors the integer village need transition for a fixed eight-tick
    rollout.  It intentionally consumes no rationale/persona text.
    """
    snapshot = record["state_snapshot"]
    slot = int(record["agent_slot"])
    needs = [int(value) for value in snapshot["needs"][slot]]
    inventory = [int(value) for value in snapshot["inventory"][slot]]
    ground = [int(value) for value in snapshot["ground"][slot]]
    tool = candidate["tool"]
    resource_delta = _ACTION_RESOURCE_DELTA.get(tool, 0)
    if tool == "eat":
        resource_delta = -1 if inventory[0] > 0 else 0  # bakery food is free
        needs[0] = min(_MAX_NEED, needs[0] + 400)
    elif tool == "sleep":
        needs[1] = min(_MAX_NEED, needs[1] + 500)
    elif tool == "speak":
        needs[2] = min(_MAX_NEED, needs[2] + 300)
    elif tool == "use":
        needs[1] = min(_MAX_NEED, needs[1] + 150)
    elif tool == "work":
        needs[1] = max(0, needs[1] - 60)
    elif tool == "pickup":
        if ground[0] > 0:
            ground[0] -= 1
            inventory[0] += 1
    elif tool == "drop":
        if inventory[0] > 0:
            inventory[0] -= 1
    for _ in range(max(0, int(horizon) - 1)):
        needs = [max(0, value - decay) for value, decay in zip(needs, _NEED_DECAY)]
    survival = int(needs[0] > 0)
    components = {
        "survival": survival,
        "hunger": needs[0],
        "energy": needs[1],
        "social": needs[2],
        "resource": resource_delta,
    }
    score = (
        components["survival"] * 1_000_000
        + components["hunger"] * 1_000
        + components["energy"] * 100
        + components["social"] * 100
        + components["resource"] * 10
    )
    return {"score": score, "components": components, "horizon": int(horizon)}


EXPERTISE_LEVELS = ("novice", "capable", "expert")
NEAR_EQUIVALENT_SCORE_DELTA = 1_000
EXPERTISE_CONTRACT_VERSION = 1


def _action_key(candidate: Mapping) -> str:
    """Canonical legal action identity; duplicate labels do not skew ranks."""
    return json.dumps(
        {
            "tool": str(candidate.get("tool", "")),
            "target_slot": candidate.get("target_slot"),
            "params": candidate.get("params") or {},
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def _persona_preference_key(record: Mapping, candidate: Mapping) -> tuple:
    traits = tuple(int(value) for value in record.get("persona", {}).get("traits", ()))
    repeat = candidate.get("repeat")
    return (
        traits,
        -(int(repeat) if repeat is not None else 1_000_000),
        str(candidate.get("id", "")),
    )


def _rank_candidates(candidates: list[dict]) -> None:
    """Attach objective ranks to every candidate without reading prompt text."""
    legal = [item for item in candidates if item.get("legal") and item.get("score") is not None]
    if not legal:
        return
    representatives: dict[str, dict] = {}
    for item in sorted(
        legal,
        key=lambda value: (-int(value["score"]), str(value.get("id", ""))),
    ):
        representatives.setdefault(_action_key(item), item)
    unique = list(representatives.values())
    values = sorted({int(item["score"]) for item in unique})
    best = max(values)
    span = best - min(values)
    for item in legal:
        key = _action_key(item)
        representative = representatives[key]
        score = int(representative["score"])
        score_rank = len(values) - 1 - values.index(score)  # 0 is objectively best
        quantile = values.index(score) / max(1, len(values) - 1)  # 0 worst, 1 best
        bucket = (
            "novice" if quantile < 1 / 3 else
            "capable" if quantile < 2 / 3 else
            "expert"
        )
        item["action_key"] = key
        item["quality_rank"] = score_rank
        item["quality_quantile"] = quantile
        item["expertise_rank"] = bucket
        if item is not representative:
            item["duplicate_of"] = str(representative.get("id", ""))
    for item in legal:
        item["quality_separation"] = bool(len(values) >= 3 and span > NEAR_EQUIVALENT_SCORE_DELTA)

def _target_payload(candidate: Mapping, index: int) -> dict:
    return {
        "candidate_index": index,
        "id": candidate.get("id"),
        "repeat": candidate.get("repeat"),
        "tool": candidate.get("tool"),
        "target_slot": candidate.get("target_slot"),
        "params": dict(candidate.get("params") or {}),
        "why": candidate.get("rationale"),
        "legal": bool(candidate.get("legal")),
        "error": candidate.get("error"),
        "score": candidate.get("score"),
        "components": candidate.get("components"),
        "horizon": candidate.get("horizon"),
        "action_key": candidate.get("action_key"),
        "quality_rank": candidate.get("quality_rank"),
        "quality_quantile": candidate.get("quality_quantile"),
        "quality_separation": candidate.get("quality_separation"),
        "expertise_rank": candidate.get("expertise_rank"),
        "duplicate_of": candidate.get("duplicate_of"),
    }


def _derive_expertise(record: Mapping, candidates: list[dict]) -> dict[str, object] | None:
    """Derive all three matched targets from one identical state."""
    _rank_candidates(candidates)
    legal = [item for item in candidates if item.get("legal") and item.get("score") is not None]
    if not legal:
        return None
    representatives: dict[str, dict] = {}
    for item in sorted(
        legal,
        key=lambda value: (-int(value["score"]), str(value.get("id", ""))),
    ):
        representatives.setdefault(_action_key(item), item)
    unique = list(representatives.values())
    values = sorted({int(item["score"]) for item in unique})
    best_score = max(values)
    near = [
        item
        for item in unique
        if best_score - int(item["score"]) <= NEAR_EQUIVALENT_SCORE_DELTA
    ]
    # This is the pre-s19 capable policy: persona can choose only from the
    # objectively near-equivalent set. Ordering is stable even for duplicate ids.
    capable = max(near, key=lambda item: _persona_preference_key(record, item))
    expert_pool = [item for item in unique if int(item["score"]) == best_score]
    expert = max(expert_pool, key=lambda item: _persona_preference_key(record, item))
    lower = [item for item in unique if int(item["score"]) < int(capable["score"])]
    novice_pool = [item for item in lower if item.get("quality_quantile", 1.0) <= 1 / 3]
    if not novice_pool:
        novice_pool = lower
    novice = (
        max(novice_pool, key=lambda item: (int(item["score"]),) + _persona_preference_key(record, item))
        if novice_pool
        else capable
    )
    targets = {
        "novice": _target_payload(novice, candidates.index(novice)),
        "capable": _target_payload(capable, candidates.index(capable)),
        "expert": _target_payload(expert, candidates.index(expert)),
    }
    degenerate_reason = None
    if len(unique) == 1:
        degenerate_reason = "only_one_unique_legal_action"
    elif len(values) == 1:
        degenerate_reason = "all_unique_legal_actions_tied"
    elif not any(int(item["score"]) < int(capable["score"]) for item in unique):
        degenerate_reason = "capable_is_lowest_quality"
    elif len(values) < 3 or best_score - min(values) <= NEAR_EQUIVALENT_SCORE_DELTA:
        degenerate_reason = "insufficient_quality_separation"
    return {
        "targets": targets,
        "contract": {
            "version": EXPERTISE_CONTRACT_VERSION,
            "levels": list(EXPERTISE_LEVELS),
            "source": "deterministic_legal_candidate_rollout",
            "same_state": True,
            "state_id": _state_id(record),
            "horizon": int(record.get("grounding", {}).get("horizon", GROUNDING_HORIZON)),
            "rank_order": "quality_rank_0_is_best",
            "quantile_order": "0_is_worst_1_is_best",
            "near_equivalent_score_delta": NEAR_EQUIVALENT_SCORE_DELTA,
            "persona_tiebreak": "traits_then_repeat_then_id",
            "duplicate_actions": "one_representative_per_canonical_action",
            "duplicate_representative": "highest_score_then_id",
            "degenerate_reason": degenerate_reason,
        },
    }


def _apply_expertise(record: dict) -> dict:
    grounding = record.get("grounding")
    if not isinstance(grounding, dict):
        return record
    candidates = grounding.get("candidates")
    if not isinstance(candidates, list):
        return record
    expertise = _derive_expertise(record, candidates)
    if expertise is None:
        return record
    grounding.update({
        "expertise_targets": expertise["targets"],
        "expertise_contract": expertise["contract"],
    })
    capable = expertise["targets"]["capable"]
    record["decision"] = {
        "tool": capable["tool"],
        "target_slot": capable["target_slot"],
        "params": capable["params"],
        "why": capable["why"],
    }
    grounding["selected_index"] = capable["candidate_index"]
    grounding["score_components"] = candidates[capable["candidate_index"]].get("components")
    grounding["score"] = capable["score"]
    grounding["expertise_rank"] = "capable"
    grounding["tie_break"] = "personality_only_near_equivalent_outcomes"
    return record


def _expertise_summary(records: Sequence[Mapping]) -> dict[str, object]:
    level_counts: Counter[str] = Counter()
    degenerate: Counter[str] = Counter()
    separable = 0
    objective_order_violations = 0
    for record in records:
        grounding = record.get("grounding", {})
        targets = grounding.get("expertise_targets", {})
        if not isinstance(targets, Mapping) or set(targets) != set(EXPERTISE_LEVELS):
            continue
        scores = [targets[level].get("score") for level in EXPERTISE_LEVELS]
        for level, target in targets.items():
            level_counts[level] += 1
        reason = grounding.get("expertise_contract", {}).get("degenerate_reason")
        if reason:
            degenerate[str(reason)] += 1
        if scores[2] > scores[1] > scores[0]:
            separable += 1
        if not (scores[2] >= scores[1] >= scores[0]):
            objective_order_violations += 1
    return {
        "expertise_level_coverage": dict(sorted(level_counts.items())),
        "separable_states": separable,
        "degenerate_counts": dict(sorted(degenerate.items())),
        "objective_order_violations": objective_order_violations,
        "states_with_targets": sum(level_counts.values()) // len(EXPERTISE_LEVELS),
    }


def derive_expertise_targets(
    grounded_path: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Materialize s19 metadata from grounded rows without replay or Spark."""
    rows = list(_iter_jsonl(grounded_path))
    derived = [_apply_expertise(dict(row)) for row in rows]
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for row in derived:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    report = {
        "states": len(rows),
        "path": str(destination),
        **_expertise_summary(derived),
    }
    report["output_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return report
def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_copy(value: Mapping) -> dict:
    return json.loads(_canonical_json(value))


def _canonical_source_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Canonicalize state and candidate ordering before expertise derivation."""
    canonical: list[dict] = []
    seen: set[str] = set()
    for row in rows:
        state_id = _state_id(row)
        if state_id in seen:
            raise ValueError(f"duplicate source state {state_id}")
        seen.add(state_id)
        copied = _canonical_copy(row)
        grounding = copied.get("grounding")
        if not isinstance(grounding, Mapping) or not isinstance(grounding.get("candidates"), list):
            raise ValueError(f"state {state_id} has no retained candidate evidence")
        grounding["candidates"] = sorted(
            (_canonical_copy(candidate) for candidate in grounding["candidates"]),
            key=_canonical_json,
        )
        canonical.append(copied)
    return sorted(canonical, key=_state_id)


def _matched_invariant(record: Mapping) -> str:
    """Return source-state bytes after removing only level/target projections."""
    copied = _canonical_copy(record)
    copied.pop("decision", None)
    copied.pop("selected_action", None)
    copied.pop("selected_target", None)
    copied.pop("expertise", None)
    copied.pop("expertise_level", None)
    copied.pop("expertise_one_hot", None)
    grounding = copied.get("grounding")
    if isinstance(grounding, dict):
        for key in ("selected_index", "score", "score_components", "expertise_rank", "selected_target"):
            grounding.pop(key, None)
    return _canonical_json(copied)


def _target_decision(target: Mapping) -> dict:
    return {
        "tool": target.get("tool"),
        "target_slot": target.get("target_slot"),
        "params": dict(target.get("params") or {}),
        "why": target.get("why"),
    }


def _materialize_matched_record(
    source: Mapping,
    level: str,
    target: Mapping,
    source_hash: str,
) -> dict:
    state_id = _state_id(source)
    record = _canonical_copy(source)
    decision = _target_decision(target)
    one_hot = {item: int(item == level) for item in EXPERTISE_LEVELS}
    contract = record["grounding"]["expertise_contract"]
    degenerate_reason = contract.get("degenerate_reason")
    strictly_separable = (
        record["grounding"]["expertise_targets"]["expert"]["score"]
        > record["grounding"]["expertise_targets"]["capable"]["score"]
        > record["grounding"]["expertise_targets"]["novice"]["score"]
    )
    group = {
        "id": state_id,
        "state_id": state_id,
        "source_state_id": state_id,
        "source_artifact_sha256": source_hash,
        "levels": list(EXPERTISE_LEVELS),
        "records": len(EXPERTISE_LEVELS),
        "trajectory_split": False,
    }
    metadata = {
        "level": level,
        "one_hot": one_hot,
        "one_hot_vector": [one_hot[item] for item in EXPERTISE_LEVELS],
        "contract_version": EXPERTISE_CONTRACT_VERSION,
        "state_id": state_id,
        "group_id": state_id,
        "source_artifact_sha256": source_hash,
        "degenerate_reason": degenerate_reason,
        "strictly_separable": strictly_separable,
        "target": _canonical_copy(target),
    }
    record.update(
        {
            "state_id": state_id,
            "source_state_id": state_id,
            "group_id": state_id,
            "matched_group_id": state_id,
            "source_artifact_sha256": source_hash,
            "expertise_level": level,
            "expertise_one_hot": one_hot,
            "expertise": metadata,
            "matched_group": group,
            "selected_target": _canonical_copy(target),
            "selected_action": decision,
            "decision": decision,
        }
    )
    grounding = record["grounding"]
    grounding.update(
        {
            "selected_index": target["candidate_index"],
            "score_components": target.get("components"),
            "score": target.get("score"),
            "expertise_rank": level,
            "selected_target": _canonical_copy(target),
        }
    )
    return record


def _validate_matched_records(
    records: Sequence[Mapping],
    source_rows: Sequence[Mapping],
) -> dict[str, object]:
    source_ids = [_state_id(row) for row in source_rows]
    source_by_id = dict(zip(source_ids, source_rows))
    expected_source = {state_id: _canonical_json(row) for state_id, row in source_by_id.items()}
    grouped: dict[str, list[Mapping]] = {}
    invariant_violations = 0
    legal_violations = 0
    trace_violations = 0
    order_violations = 0
    for record in records:
        state_id = str(record.get("state_id", ""))
        grouped.setdefault(state_id, []).append(record)
        if state_id not in expected_source:
            trace_violations += 1
            continue
        grounding = record.get("grounding", {})
        targets = grounding.get("expertise_targets", {})
        level = record.get("expertise_level")
        target = record.get("selected_target")
        candidates = grounding.get("candidates", [])
        try:
            index = int(target["candidate_index"])
            candidate = candidates[index]
        except (KeyError, IndexError, TypeError, ValueError):
            trace_violations += 1
            continue
        if (
            level not in EXPERTISE_LEVELS
            or not target.get("legal")
            or not candidate.get("legal")
            or candidate.get("id") != target.get("id")
            or _action_key(candidate) != target.get("action_key")
            or candidate.get("score") != target.get("score")
        ):
            legal_violations += 1
        if set(targets) != set(EXPERTISE_LEVELS):
            trace_violations += 1
    for state_id in source_ids:
        group = grouped.get(state_id, [])
        levels = [record.get("expertise_level") for record in group]
        if len(group) != len(EXPERTISE_LEVELS) or sorted(levels) != sorted(EXPERTISE_LEVELS):
            trace_violations += 1
            continue
        invariant = _matched_invariant(group[0])
        invariant_violations += sum(
            _matched_invariant(record) != invariant for record in group[1:]
        )
        scores = {
            level: next(
                record["selected_target"]["score"]
                for record in group
                if record["expertise_level"] == level
            )
            for level in EXPERTISE_LEVELS
        }
        if not (scores["expert"] >= scores["capable"] >= scores["novice"]):
            order_violations += 1
    return {
        "groups": grouped,
        "triplet_violations": trace_violations,
        "source_invariant_violations": invariant_violations,
        "legal_target_violations": legal_violations,
        "objective_order_violations": order_violations,
    }


def materialize_matched_expertise(
    grounded_path: str | Path,
    output: str | Path,
    manifest: str | Path | None = None,
) -> dict[str, object]:
    """Emit deterministic novice/capable/expert records for every grounded state."""
    source_path = Path(grounded_path)
    destination = Path(output)
    manifest_path = Path(manifest) if manifest is not None else destination.with_suffix(".manifest.json")
    raw_rows = list(_iter_jsonl(source_path))
    source_rows = _canonical_source_rows(raw_rows)
    canonical_source = "\n".join(_canonical_json(row) for row in source_rows) + ("\n" if source_rows else "")
    source_hash = hashlib.sha256(canonical_source.encode("utf-8")).hexdigest()
    input_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    derived: list[dict] = []
    for source in source_rows:
        ready, reason = _grounding_state_ready(source)
        if not ready:
            raise ValueError(f"state {_state_id(source)} is not replay-traceable: {reason}")
        # Keep the canonical source row immutable: expertise derivation adds
        # projections to grounding and decision, which must not contaminate the
        # source-state bytes used for matched-triplet validation.
        row = _apply_expertise(_canonical_copy(source))
        targets = row.get("grounding", {}).get("expertise_targets", {})
        if set(targets) != set(EXPERTISE_LEVELS):
            raise ValueError(f"state {_state_id(source)} did not derive all expertise targets")
        for level in EXPERTISE_LEVELS:
            derived.append(_materialize_matched_record(row, level, targets[level], source_hash))
    validation = _validate_matched_records(derived, source_rows)
    if any(
        validation[key]
        for key in (
            "triplet_violations",
            "source_invariant_violations",
            "legal_target_violations",
            "objective_order_violations",
        )
    ):
        raise ValueError(f"matched expertise validation failed: {validation}")
    level_counts = Counter(str(record["expertise_level"]) for record in derived)
    tool_distributions: dict[str, dict[str, int]] = {}
    score_pairs = {
        "novice_capable": 0,
        "capable_expert": 0,
        "novice_expert": 0,
    }
    target_pairs = {key: 0 for key in score_pairs}
    degenerate = Counter()
    strictly_separable = 0
    for state_id in sorted(validation["groups"]):
        group = validation["groups"][state_id]
        by_level = {record["expertise_level"]: record for record in group}
        for level, record in by_level.items():
            tool_distributions.setdefault(level, Counter())[str(record["selected_target"]["tool"])] += 1
        scores = {level: by_level[level]["selected_target"]["score"] for level in EXPERTISE_LEVELS}
        actions = {level: by_level[level]["selected_target"]["action_key"] for level in EXPERTISE_LEVELS}
        pairs = (("novice_capable", "novice", "capable"), ("capable_expert", "capable", "expert"), ("novice_expert", "novice", "expert"))
        for key, left, right in pairs:
            score_pairs[key] += int(scores[left] == scores[right])
            target_pairs[key] += int(actions[left] == actions[right])
        contract = by_level["expert"]["grounding"]["expertise_contract"]
        if contract.get("degenerate_reason"):
            degenerate[str(contract["degenerate_reason"])] += 1
        strictly_separable += int(scores["expert"] > scores["capable"] > scores["novice"])
    state_count = len(source_rows)
    pair_denominator = max(1, state_count)
    per_level = {
        level: {
            "records": level_counts[level],
            "tool_distribution": dict(sorted(tool_distributions.get(level, {}).items())),
            "class_distribution": dict(sorted(tool_distributions.get(level, {}).items())),
        }
        for level in EXPERTISE_LEVELS
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in derived:
            stream.write(_canonical_json(record) + "\n")
    output_hash = hashlib.sha256(destination.read_bytes()).hexdigest()
    manifest_data = {
        "schema_version": "mw-v1-s20-matched-expertise-v1",
        "source": {
            "path": str(source_path),
            "source_artifact_sha256": source_hash,
            "source_input_sha256": input_hash,
            "states": state_count,
            "selected_states": state_count,
            "teacher_calls": 0,
        },
        "output": {
            "path": str(destination),
            "sha256": output_hash,
            "records": len(derived),
        },
        "counts": {
            "source_states": state_count,
            "matched_groups": len(validation["groups"]),
            "records": len(derived),
            "records_per_group": len(EXPERTISE_LEVELS),
            "per_level": dict(sorted(level_counts.items())),
        },
        "per_level": per_level,
        "equality_distinctness": {
            "score_equality_rates": {key: score_pairs[key] / pair_denominator for key in score_pairs},
            "score_distinctness_rates": {key: 1 - score_pairs[key] / pair_denominator for key in score_pairs},
            "target_equality_rates": {key: target_pairs[key] / pair_denominator for key in target_pairs},
            "target_distinctness_rates": {key: 1 - target_pairs[key] / pair_denominator for key in target_pairs},
        },
        "degeneracy": {
            "strictly_separable_states": strictly_separable,
            "strictly_separable_rate": strictly_separable / pair_denominator,
            "degenerate_reason_counts": dict(sorted(degenerate.items())),
        },
        "trajectory_integrity": {
            "group_key": "state_id",
            "split_by_trajectory": False,
            "all_triplets_complete": True,
            "source_state_is_identical_within_group": True,
        },
        "assertions": {
            "exact_record_count": len(derived) == 3 * state_count,
            "complete_triplets": validation["triplet_violations"] == 0,
            "legal_replay_traceable_targets": validation["legal_target_violations"] == 0,
            "objective_order_violations": validation["objective_order_violations"],
            "source_invariant_violations": validation["source_invariant_violations"],
            "teacher_calls": 0,
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(_canonical_json(manifest_data) + "\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return {
        **manifest_data,
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_hash,
    }


materialize_expertise_dataset = materialize_matched_expertise


def ground_labels(
    states_path: str | Path,
    labels_path: str | Path,
    output: str | Path,
    *,
    horizon: int = GROUNDING_HORIZON,
) -> dict[str, object]:
    """Ground every retained Spark candidate and emit one stable target/state row."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    states = {_state_id(row): row for row in _iter_jsonl(states_path)}
    unreplayable: Counter[str] = Counter()
    for row in states.values():
        ready, reason = _grounding_state_ready(row)
        if not ready:
            unreplayable[str(reason)] += 1
    groups: dict[str, list[dict]] = {}
    blockers: Counter[str] = Counter(unreplayable)
    unknown = 0
    malformed = 0
    for label in _iter_jsonl(labels_path):
        base_id, _ = _base_label_id(label.get("id", ""))
        record = states.get(base_id)
        if record is None:
            unknown += 1
            continue
        ready, _ = _grounding_state_ready(record)
        if not ready:
            continue
        try:
            candidate = _candidate_record(record, label, str(label.get("id", base_id)))
        except (TypeError, ValueError, KeyError) as exc:
            malformed += 1
            blockers["malformed_candidate:" + type(exc).__name__] += 1
            continue
        if candidate["legal"]:
            candidate.update(_rollout_score(record, candidate, horizon))
        else:
            candidate["score"] = None
            candidate["components"] = None
        groups.setdefault(base_id, []).append(candidate)

    kept: list[dict] = []
    for base_id in sorted(groups):
        record = states[base_id]
        candidates = groups[base_id]
        legal = [item for item in candidates if item["legal"]]
        if not legal:
            blockers["no_legal_candidates"] += 1
            continue
        legal_tools = [item["tool"] for item in candidates if item["legal"]]
        pair_total = len(legal_tools) * (len(legal_tools) - 1) // 2
        pair_agree = sum(
            1
            for left_index, left in enumerate(legal_tools)
            for right in legal_tools[left_index + 1 :]
            if left == right
        )
        assembled = dict(record)
        assembled["replay"] = False
        assembled["grounding"] = {
            "candidates": candidates,
            "horizon": horizon,
            "replay_hash": record["replay_provenance"]["replay_hash"],
            "disagreement": {
                "legal_candidates": len(legal_tools),
                "unique_tools": len(set(legal_tools)),
                "tool_counts": dict(sorted(Counter(legal_tools).items())),
                "pairwise_tool_agreement": pair_agree / pair_total if pair_total else 1.0,
                "tool_entropy_bits": _entropy(legal_tools),
            },
        }
        _apply_expertise(assembled)
        kept.append(assembled)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in kept:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    candidate_count = sum(len(value) for value in groups.values())
    legal_count = sum(sum(bool(item["legal"]) for item in value) for value in groups.values())
    return {
        "states": len(states),
        "replayable_states": len(states) - sum(unreplayable.values()),
        "unreplayable_states": sum(unreplayable.values()),
        "candidate_labels": candidate_count,
        "legal_candidates": legal_count,
        "grounded_states": len(kept),
        "coverage": len(kept) / len(states) if states else 0.0,
        "unknown_labels": unknown,
        "malformed_candidates": malformed,
        "blockers": dict(sorted(blockers.items())),
        "horizon": horizon,
        "path": str(destination),
        **_expertise_summary(kept),
    }


def assemble_labels(
    states_path: str | Path,
    labels_path: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Join labels; ground replay-capable exports, otherwise keep legacy join."""
    state_rows = list(_iter_jsonl(states_path))
    if any("replay_provenance" in row or "state_snapshot" in row for row in state_rows):
        return ground_labels(states_path, labels_path, output)
    states = {_state_id(r): r for r in state_rows}
    labels = list(_iter_jsonl(labels_path))
    kept: list[dict] = []
    illegal = 0
    unknown = 0
    for label in labels:
        key = str(label.get("id", ""))
        record = states.get(key)
        if record is None:
            unknown += 1
            continue
        fields = _label_fields(label)
        tool = str(fields.get("tool", "")).strip().lower()
        if tool not in _afforded_tools(record):
            illegal += 1
            continue
        arg = fields.get("arg")
        if isinstance(arg, Mapping):
            params = dict(arg)
        elif arg is not None:
            params = {"arg": arg}
        else:
            params = {}
        decision = dict(record["decision"])
        decision.update(
            {
                "tool": tool,
                "target_slot": _target_slot(record, fields.get("target")),
                "params": params,
            }
        )
        if fields.get("why") is not None:
            decision["why"] = str(fields["why"])
        assembled = dict(record)
        assembled["decision"] = decision
        assembled["replay"] = False
        kept.append(assembled)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in kept:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    legal_rate = len(kept) / len(labels) if labels else 0.0
    return {
        "labels": len(labels),
        "kept": len(kept),
        "illegal_dropped": illegal,
        "unknown_dropped": unknown,
        "legal_rate": legal_rate,
        "path": str(destination),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect = sub.add_parser("collect")
    collect.add_argument("--out", default="training/artifacts/spark_states.jsonl")
    collect.add_argument("--states", type=int, default=1_000)
    collect.add_argument("--agents", type=int, default=50)
    collect.add_argument("--seed", type=int, default=1)
    collect.add_argument("--ticks-per-seed", type=int, default=300)
    collect.add_argument("--prompts-out", default="training/artifacts/spark_prompts.jsonl")
    prompts = sub.add_parser("prompts")
    prompts.add_argument("--states", default="training/artifacts/spark_states.jsonl")
    prompts.add_argument("--out", default="training/artifacts/spark_prompts.jsonl")
    consistency_sample = sub.add_parser(
        "consistency-sample",
        help="select a deterministic stratified teacher-ceiling sample",
    )
    derive = sub.add_parser(
        "derive-expertise",
        help="derive novice/capable/expert targets from an existing grounded JSONL",
    )
    derive.add_argument("--grounded", required=True)
    derive.add_argument("--out", required=True)
    matched = sub.add_parser(
        "materialize-matched-expertise",
        help="materialize one deterministic novice/capable/expert record per grounded state",
    )
    matched.add_argument(
        "--grounded", default="training/artifacts/spark_ground_omni.jsonl"
    )
    matched.add_argument(
        "--out", default="training/artifacts/spark_matched_expertise.jsonl"
    )
    matched.add_argument("--manifest")
    consistency_sample.add_argument("--states", default="training/artifacts/spark_states.jsonl")
    consistency_sample.add_argument("--out", default="training/artifacts/spark_consistency_states.jsonl")
    consistency_sample.add_argument("--count", type=int, default=100)
    consistency_sample.add_argument("--seed", type=int, default=0)
    consistency_prompts = sub.add_parser(
        "consistency-prompts",
        help="emit repeated prompts for Spark self-consistency",
    )
    consistency_prompts.add_argument(
        "--states", default="training/artifacts/spark_consistency_states.jsonl"
    )
    consistency_prompts.add_argument(
        "--out", default="training/artifacts/spark_consistency_prompts.jsonl"
    )
    consistency_prompts.add_argument("--repeats", type=int, default=3)
    consistency_analyze = sub.add_parser(
        "consistency-analyze",
        help="analyze completed repeated labels after legal filtering",
    )
    consistency_analyze.add_argument(
        "--states", default="training/artifacts/spark_consistency_states.jsonl"
    )
    consistency_analyze.add_argument(
        "--labels", default="training/artifacts/spark_consistency_labels.jsonl"
    )
    consistency_analyze.add_argument(
        "--out", default="training/artifacts/spark_consistency_report.json"
    )
    label = sub.add_parser("label")
    label.add_argument("--prompts", default="training/artifacts/spark_prompts.jsonl")
    label.add_argument("--out", default="training/artifacts/spark_labels.jsonl")
    label.add_argument("--model", default="gpt-5.3-codex-spark")
    label.add_argument("--limit", type=int)
    label.add_argument("--batch-size", type=int, default=1)
    label.add_argument("--concurrency", type=int, default=1)
    label.add_argument("--retries", "--retry-attempts", type=int, default=5)
    label.add_argument("--retry-backoff", type=float, default=1.0)
    label.add_argument("--max-backoff", type=float, default=60.0)
    label.add_argument("--retry-jitter", type=float, default=0.25)
    label.add_argument("--resume", action="store_true")
    assemble = sub.add_parser("assemble")
    assemble.add_argument("--states", default="training/artifacts/spark_states.jsonl")
    assemble.add_argument("--labels", default="training/artifacts/spark_labels.jsonl")
    assemble.add_argument("--out", default="training/artifacts/spark_omni.jsonl")
    args = parser.parse_args(argv)
    if args.command == "collect":
        report = collect_states(
            args.out,
            states=args.states,
            agents=args.agents,
            seed=args.seed,
            ticks_per_seed=args.ticks_per_seed,
        )
        report["prompts"] = write_prompts(args.out, args.prompts_out)
    elif args.command == "prompts":
        report = write_prompts(args.states, args.out)
    elif args.command == "consistency-sample":
        report = export_consistency_sample(
            args.states, args.out, count=args.count, seed=args.seed
        )
    elif args.command == "consistency-prompts":
        report = write_consistency_prompts(args.states, args.out, repeats=args.repeats)
    elif args.command == "consistency-analyze":
        report = analyze_teacher_consistency(args.states, args.labels, args.out)
    elif args.command == "label":
        report = label_with_spark(
            args.prompts,
            args.out,
            model=args.model,
            limit=args.limit,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            retries=args.retries,
            retry_backoff=args.retry_backoff,
            max_backoff=args.max_backoff,
            retry_jitter=args.retry_jitter,
            resume=args.resume,
        )
    elif args.command == "materialize-matched-expertise":
        report = materialize_matched_expertise(args.grounded, args.out, args.manifest)
    elif args.command == "derive-expertise":
        report = derive_expertise_targets(args.grounded, args.out)
    else:
        report = assemble_labels(args.states, args.labels, args.out)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
