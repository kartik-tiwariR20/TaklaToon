"""Append-only transitions, honest episode boundaries, and separate evaluator data."""
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import csv
import json
import platform
import uuid
import gymnasium as gym
from .env import OBSERVATION_NAMES, Action


def new_run_path(root="runs", prefix="session"):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path(root) / f"{prefix}-{stamp}-{uuid.uuid4().hex[:6]}"


class TransitionRecorder(gym.Wrapper):
    """Never inserts evaluator-only information into agent observations or info."""
    def __init__(self, env, directory, policy="manual"):
        super().__init__(env)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.policy = policy
        self.control_source = policy
        self._transitions = (self.directory / "transitions.jsonl").open("w", encoding="utf-8")
        self._metadata = (self.directory / "evaluator.jsonl").open("w", encoding="utf-8")
        self._episodes = (self.directory / "episodes.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._episodes, fieldnames=["episode_id", "seed", "steps", "return",
            "outcome", "success", "terminated", "truncated", "policy"])
        self._writer.writeheader()
        self.episode_id = -1
        self._active = False
        self._closed = False
        self._obs = None
        self._seed = None
        manifest = {"schema_version": "taklatoon.transitions.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "python": platform.python_version(), "platform": platform.platform(), "policy": policy,
            "versions": {p: version(p) for p in ("taklatoon", "pygame", "pymunk", "gymnasium", "numpy")},
            "observation_names": list(OBSERVATION_NAMES), "actions": {a.name: a.value for a in Action},
            "state_units": "Normalized features; see OBSERVATIONS.md. Physics uses px, s, px/s.",
            "hidden_values": "evaluator.jsonl is operator-only; do not feed it into an agent."}
        (self.directory / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def reset(self, *, seed=None, options=None):
        if self._closed:
            raise RuntimeError("Recorder is closed")
        # Validate/reset the underlying environment before ending an existing episode.
        obs, info = self.env.reset(seed=seed, options=options)
        if self._active:
            self._finish("manual_reset", False, False, False)
        self.episode_id += 1
        self._seed = seed
        self._obs = obs.copy()
        self._steps, self._return = 0, 0.
        self._active = True
        metadata = {"episode_id": self.episode_id, "seed": seed,
                    **self.env.unwrapped.evaluator_metadata()}
        self._metadata.write(json.dumps(metadata) + "\n")
        self._metadata.flush()
        return obs, info

    def step(self, action):
        if not self._active or self._closed:
            raise RuntimeError("Call reset() before recording a step")
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        self._return += reward
        row = {"episode_id": self.episode_id, "step": self._steps,
               "observation": self._obs.tolist(), "action": int(action),
               "next_observation": obs.tolist(), "reward": reward,
               "terminated": terminated, "truncated": truncated,
               "control_source": self.control_source,
               "event": info["event"], "reward_components": info["reward_components"],
               "sim_time": info["sim_time"]}
        self._transitions.write(json.dumps(row, allow_nan=False) + "\n")
        self._transitions.flush()
        self._obs = obs.copy()
        if terminated or truncated:
            self._finish(info["event"], info["success"], terminated, truncated)
        return obs, reward, terminated, truncated, info

    def _finish(self, outcome, success, terminated, truncated):
        self._writer.writerow({"episode_id": self.episode_id, "seed": self._seed,
            "steps": self._steps, "return": self._return, "outcome": outcome,
            "success": success, "terminated": terminated, "truncated": truncated, "policy": self.policy})
        self._episodes.flush()
        self._active = False

    def close(self):
        if self._closed:
            return
        try:
            if self._active:
                self._finish("interrupted", False, False, False)
        finally:
            self._transitions.close()
            self._metadata.close()
            self._episodes.close()
            self._closed = True
            self.env.close()

