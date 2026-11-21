# Agent contract: observation, actions, and episode boundaries

## Observation

`observation_space = Box(-10, 10, shape=(21,), dtype=float32)`.
The broad finite bounds accommodate all supported starting presets and control behaviour.
These features use fixed scale factors, not dataset-fitted normalization. A learned model may fit additional preprocessing on training data only.

| Index | Feature | Formula |
|---|---|---|
| 0 | x | body.x / 1200 |
| 1 | y | body.y / 600 |
| 2 | vx | body.vx / 300 |
| 3 | vy | body.vy / 1200 |
| 4 | grounded | 0 or 1 |
| 5 | goal_dx | (goal.x - body.x) / 1200 |
| 6 | goal_dy | (goal.y - body.y) / 600 |
| 7 | jump_held | Whether the previous action requested a jump: 0 or 1 |
| 8 | target_hold_fraction | MotionLab hold progress, min(hold steps / 6, 1) |
| 9–12 | platform 0 | left_dx / 1200, right_dx / 1200, top_dy / 600, mask |
| 13–16 | platform 1 | same four fields |
| 17–20 | platform 2 | same four fields |

All platform differences are relative to the current body center. Platform ordering is fixed per level, from left to right. Missing platforms have all four fields zero, including mask=0. Present platforms have mask=1. The floor bottom is fixed at y=-200; its observable top and horizontal extent are sufficient for these tasks. World side boundaries and character dimensions are fixed public conventions.

Jump latch and target hold progress are included so the controller does not need to infer these internal action/goal states. No hidden gravity, drag, jump strength, preset ID, or seed appears here. Different hidden dynamics can produce the same observation but different future motion: a future model needs task adaptation or history/context. A context-free predictor trained on pooled settings may average incompatible dynamics.

## Actions

`action_space = Discrete(6)`.

| ID | Enum | Horizontal input | Jump requested |
|---|---|---:|---|
| 0 | IDLE | 0 | no |
| 1 | LEFT | -1 | no |
| 2 | RIGHT | +1 | no |
| 3 | JUMP | 0 | yes |
| 4 | LEFT_JUMP | -1 | yes |
| 5 | RIGHT_JUMP | +1 | yes |

Each action lasts 1/30 simulated second at the default settings. Four 1/120-second physics substeps produce it. A jump triggers only if grounded and the preceding action did not request a jump. Release with any non-jump action before the next jump. Manual input converts Space keydown events to the same action IDs.

## Info

`event`, `success`, `step`, `sim_time`, and `reward_components` are public task information. `event` is `reset`, `running`, `success`, `fall`, or `timeout`. The info dictionary contains no evaluator-only parameters.

## Logs

Use `(observation, action, next_observation)` for model training. Preserve `terminated` and `truncated` separately. Do not join adjacent records across episode IDs. Manual resets and interrupted sessions close the episode summary without fabricating a task-terminal observation. Record dataset splits at episode/task level to avoid neighbouring-frame leakage.

## Evaluator-only interface

`env.evaluator_metadata()` supports logging, reproducible replay, and human debugging. The future agent must not call it or inspect simulator bodies/physics internals to preview outcomes. This is a research/data-flow contract, not a security sandbox against malicious Python code.

