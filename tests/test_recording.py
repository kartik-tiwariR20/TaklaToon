import csv
import json
from types import SimpleNamespace
import numpy as np
from taklatoon import TaklaToonEnv, Action
from taklatoon.recording import TransitionRecorder
from taklatoon.cli import replay, inspect_run


def test_log_replay_randomized_task(tmp_path):
    path = tmp_path / "run"
    env = TransitionRecorder(TaklaToonEnv(randomize=True, max_steps=8), path, "test")
    try:
        env.reset(seed=31)
        for _ in range(8): env.step(Action.RIGHT)
    finally:
        env.close()
    manifest = json.loads((path / "manifest.json").read_text())
    transitions = [json.loads(s) for s in (path / "transitions.jsonl").read_text().splitlines()]
    assert len(manifest["observation_names"]) == 21
    assert len(transitions) == 8
    assert transitions[-1]["truncated"] and not transitions[-1]["terminated"]
    assert "physics" not in transitions[0]
    replay(SimpleNamespace(directory=path, episode=0, headless=True))
    inspect_run(SimpleNamespace(directory=path))


def test_reset_and_interruption_are_honest_episode_boundaries(tmp_path):
    path = tmp_path / "run"
    env = TransitionRecorder(TaklaToonEnv(), path)
    env.reset(seed=1)
    env.step(0)
    env.reset(seed=2)
    env.step(0)
    env.close()
    env.close()  # Idempotent cleanup.
    with (path / "episodes.csv").open() as stream: rows = list(csv.DictReader(stream))
    assert [r["outcome"] for r in rows] == ["manual_reset", "interrupted"]
    assert all(r["terminated"] == "False" and r["truncated"] == "False" for r in rows)
    assert [r["steps"] for r in rows] == ["1", "1"]
    inspect_run(SimpleNamespace(directory=path))


def test_replay_detects_corrupted_transition(tmp_path):
    import pytest
    path = tmp_path / "run"
    env = TransitionRecorder(TaklaToonEnv(max_steps=1), path)
    env.reset(seed=1)
    env.step(2)
    env.close()
    file = path / "transitions.jsonl"
    row = json.loads(file.read_text())
    row["reward"] += 1
    file.write_text(json.dumps(row) + "\n")
    with pytest.raises(AssertionError): replay(SimpleNamespace(directory=path, episode=0, headless=True))

