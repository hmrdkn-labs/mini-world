from __future__ import annotations

import json
from pathlib import Path

from mw_training.ladder import _matched_group_key, _split_matched


def test_matched_split_keeps_each_expertise_triplet_together() -> None:
    path = Path(__file__).parents[1] / "artifacts" / "spark_matched_expertise.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    partitions = _split_matched(rows, split_seed=20260719)
    locations: dict[str, int] = {}
    for partition_index, partition in enumerate(partitions):
        for row in partition:
            group = _matched_group_key(row)
            previous = locations.setdefault(group, partition_index)
            assert previous == partition_index
    assert len(locations) == 1000
    assert [len(partition) for partition in partitions] == [2100, 450, 450]
