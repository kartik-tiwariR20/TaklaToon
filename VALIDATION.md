# Validation of the delivered foundation

The implementation was exercised on Windows with Python 3.13.7. Exact installed versions are in `requirements-tested.txt`.

## Checks performed

- 33 automated pytest cases passed.
- Gymnasium's environment checker passed.
- The registered `TaklaToon-v0` interface was reset and stepped.
- All three tasks completed under all four fixed presets with the simple scripted reference controller.
- MotionLab's normal-preset reference completed in 147 actions. JumpLab and GapRunner each completed in 98 actions.
- Seeded randomized trajectories reproduced on repeated runs.
- Recorded randomized tasks replayed with matching observations, rewards, and terminal flags.
- An intentionally altered logged reward caused replay verification to fail as expected.
- Manual controls, scenario switches, pause, reference toggle, and cleanup ran through the actual application loop using SDL's dummy display and synthetic events.
- Click targets were tested after display resizing.
- RGB rendering returned uint8 images and did not advance the physics state.
- A real sample run contains 294 transitions from three scripted GapRunner completions.

These checks establish environment functionality and basic reproducibility on the tested setup. They do not demonstrate a trained AI, an active-exploration improvement, cross-platform bitwise determinism, or playability of every arbitrary custom physics setting.

## Reproduce the checks

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe run.py check
.\.venv\Scripts\python.exe run.py inspect examples\sample-run
.\.venv\Scripts\python.exe run.py replay examples\sample-run --episode 0 --headless
```

The tests use dummy display handling where appropriate. The delivered screenshot is an actual RGB render from the implemented environment. The live manual-play experience should also be tested by your team through the included VS Code launch profile.

