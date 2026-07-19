"""Collect semantic trajectories, build Spark prompts, and assemble labels.

The simulator remains the source of truth for observations and affordances.  This
module only transports structured schema-v2 records to a teacher and applies a
small, deterministic kernel filter before writing records for ``train_omni``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence

from .dataset import TOOL_NAMES
from .persona import teacher_prompt_fragment

LOCATION_NAMES = ("empty", "home", "bakery", "well", "field")
NEED_BANDS = ("critical", "low", "medium", "high")


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
    rows: list[dict] = []
    profile_counts: dict[str, int] = {}
    with tempfile.TemporaryDirectory(prefix="mw-spark-") as temporary:
        for profile_index, (profile, quota) in enumerate(zip(profiles, quotas)):
            selected: list[dict] = []
            run = 0
            while len(selected) < quota:
                run_seed = seed + profile_index * 1_000_003 + run * 7919
                path = Path(temporary) / f"trajectory-{profile}-{run}.jsonl"
                profile_fraction = 50 if profile == "hostile" else 25
                _run_export(
                    root,
                    run_seed,
                    agents,
                    ticks_per_seed,
                    path,
                    profile=profile,
                    fraction=profile_fraction,
                )
                with path.open(encoding="utf-8") as stream:
                    exported = [json.loads(line) for line in stream if line.strip()]
                if not exported:
                    raise RuntimeError(f"simulator exported no records for {profile}")
                need = quota - len(selected)
                selected.extend(_sample_profile(exported, need, profile, agents))
                run += 1
            rows.extend(selected[:quota])
            profile_counts[profile] = quota
    with destination.open("w", encoding="utf-8") as stream:
        for record in rows:
            stream.write(json.dumps(record, separators=(",", ":")) + "\n")
    report = summarize_states(rows)
    report["seeds"] = len({int(r["seed"]) for r in rows})
    report["profiles"] = profile_counts
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
                },
            }
            stream.write(json.dumps(payload, separators=(",", ":")) + "\n")
    return {"prompts": len(records), "path": str(destination)}


def _iter_jsonl(path: str | Path) -> Iterable[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSON at {path}:{number}") from exc


def _decode_label(output: str) -> dict:
    """Decode Spark's JSON response, tolerating a surrounding code fence."""
    for line in reversed(output.splitlines()):
        candidate = line.strip().strip("`")
        if not candidate:
            continue
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return dict(value)
    raise ValueError("Spark response did not contain a JSON object")


def label_with_spark(
    prompts_path: str | Path,
    labels_path: str | Path,
    *,
    model: str = "gpt-5.3-codex-spark",
    limit: int | None = None,
) -> dict[str, object]:
    """Run the sanctioned read-only Spark CLI over prompts.

    No fallback labels are invented: missing auth/CLI raises an actionable error
    while the already-written prompt artifact remains ready for the conductor.
    """
    plugin_root = os.environ.get("CLAUDE_PLUGIN_ROOT")
    if not plugin_root:
        raise RuntimeError(
            "Spark unavailable: CLAUDE_PLUGIN_ROOT is unset; run the labeling "
            "batch with the sanctioned codex-companion CLI"
        )
    script = Path(plugin_root) / "scripts" / "codex-companion.mjs"
    if not script.is_file():
        raise RuntimeError(f"Spark unavailable: missing sanctioned CLI at {script}")
    prompts = list(_iter_jsonl(prompts_path))
    selected = prompts if limit is None else prompts[: max(0, limit)]
    destination = Path(labels_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=destination.parent, delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            for item in selected:
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
                label = _decode_label(result.stdout)
                json.dump({"id": item["id"], **label}, temporary, separators=(",", ":"))
                temporary.write("\n")
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)
    return {"labels": len(selected), "path": str(destination), "model": model}


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


def assemble_labels(
    states_path: str | Path,
    labels_path: str | Path,
    output: str | Path,
) -> dict[str, object]:
    """Join Spark labels, drop illegal tools, and emit train_omni JSONL."""
    states = {_state_id(r): r for r in _iter_jsonl(states_path)}
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
    label = sub.add_parser("label")
    label.add_argument("--prompts", default="training/artifacts/spark_prompts.jsonl")
    label.add_argument("--out", default="training/artifacts/spark_labels.jsonl")
    label.add_argument("--model", default="gpt-5.3-codex-spark")
    label.add_argument("--limit", type=int)
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
    elif args.command == "label":
        report = label_with_spark(
            args.prompts, args.out, model=args.model, limit=args.limit
        )
    else:
        report = assemble_labels(args.states, args.labels, args.out)
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
