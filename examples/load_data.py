"""Export transition arrays for a future model. Never reads evaluator.jsonl.

Usage: python examples/load_data.py runs/reference --output runs/training.npz
This helper loads one run into memory. For very large datasets use streaming batches.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with (args.directory / "transitions.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if not rows:
        raise ValueError("No transitions to export")
    obs = np.asarray([r["observation"] for r in rows], dtype=np.float32)
    next_obs = np.asarray([r["next_observation"] for r in rows], dtype=np.float32)
    if not np.isfinite(obs).all() or not np.isfinite(next_obs).all():
        raise ValueError("Nonfinite observations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive output prevents accidental replacement of an existing training dataset.
    with args.output.open("xb") as target:
        np.savez_compressed(target, observations=obs, next_observations=next_obs,
            actions=np.asarray([r["action"] for r in rows], dtype=np.int64),
            rewards=np.asarray([r["reward"] for r in rows], dtype=np.float32),
            episode_ids=np.asarray([r["episode_id"] for r in rows], dtype=np.int64),
            terminated=np.asarray([r["terminated"] for r in rows], dtype=bool),
            truncated=np.asarray([r["truncated"] for r in rows], dtype=bool))
    print(f"Exported {len(rows)} transitions to {args.output.resolve()}")
    print("Split by episode/task, not adjacent frames. Delta targets: next_observations - observations.")


if __name__ == "__main__":
    main()

