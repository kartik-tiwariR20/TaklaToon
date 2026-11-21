from dataclasses import replace
from pathlib import Path
import argparse
import json
import csv
import time
import numpy as np
from .config import EnvConfig, PhysicsParams, PRESETS
from .env import TaklaToonEnv, Action, OBSERVATION_NAMES
from .recording import TransitionRecorder, new_run_path
from .policies import scripted_action


def config_from(args):
    return EnvConfig(level=args.level, preset=args.preset, randomize=args.randomize, max_steps=args.max_steps)


def collect(args):
    env = TransitionRecorder(TaklaToonEnv(config=config_from(args)), args.output or new_run_path(prefix="collect"), args.policy)
    successes = 0
    try:
        for ep in range(args.episodes):
            seed = args.seed + ep
            obs, _ = env.reset(seed=seed)
            rng = np.random.default_rng(seed)
            action = Action.IDLE
            while True:
                if args.policy == "scripted":
                    action = scripted_action(obs, args.level)
                elif rng.random() < .3:
                    action = int(rng.integers(0, 6))
                obs, reward, terminated, truncated, info = env.step(action)
                if terminated or truncated:
                    successes += int(info["success"])
                    print(f"Episode {ep}: {info['event']}, {info['step']} actions")
                    break
    finally:
        env.close()
    print(f"Recorded {args.episodes} episodes ({successes} successes) to {env.directory.resolve()}")


def play(args):
    import pygame
    base = TaklaToonEnv(render_mode="human", config=config_from(args))
    wrapper = None if args.no_record else TransitionRecorder(base, args.output or new_run_path(prefix="play"), "manual+optional-scripted")
    env = wrapper or base
    current_seed = args.seed
    obs, _ = env.reset(seed=current_seed)
    renderer = base._renderer
    renderer.seed = current_seed
    renderer.record_path = str(wrapper.directory.resolve()) if wrapper else "Recording disabled"
    print("TaklaToon is running. Close the window or press Esc to exit.")
    if wrapper:
        print("Recording:", wrapper.directory.resolve())
    paused, reference, running = False, args.autoplay, True
    accumulator = 0.
    jump_pending = False
    frames = 0
    clock = pygame.time.Clock()
    done_since = None

    def reset(options=None):
        nonlocal obs, accumulator, jump_pending, done_since
        obs, _ = env.reset(seed=current_seed, options=options)
        accumulator, jump_pending, done_since = 0., False, None
        renderer.seed = current_seed

    def command(name):
        nonlocal current_seed, paused, reference, running
        if name == "quit": running = False
        elif name == "reset": reset()
        elif name == "next":
            current_seed += 1
            reset()
        elif name == "pause": paused = not paused
        elif name == "reference": reference = not reference
        elif name == "hide": renderer.show_physics = not renderer.show_physics
        elif name == "trail": renderer.show_trail = not renderer.show_trail
        elif name == "randomize": reset({"randomize": not base.config.randomize, "physics_override": None})
        elif name.startswith("level:"): reset({"level": name.split(":")[1]})
        elif name.startswith("preset:"):
            reset({"preset": name.split(":")[1], "physics_override": None, "randomize": False})
        elif name.startswith("change:"):
            field = name.split(":")[1]
            choices = {"gravity": [700., 1000., 1400.], "jump_speed": [440., 520., 600.], "ground_drag": [.65, 2.5, 5.]}[field]
            old = getattr(base.physics, field)
            index = min(range(len(choices)), key=lambda i: abs(choices[i]-old))
            physics = replace(base.physics, **{field: choices[(index+1)%len(choices)]})
            reset({"physics_override": physics, "randomize": False})

    keys = {pygame.K_ESCAPE: "quit", pygame.K_r: "reset", pygame.K_n: "next", pygame.K_p: "pause",
            pygame.K_b: "reference", pygame.K_h: "hide", pygame.K_c: "trail", pygame.K_t: "randomize",
            pygame.K_1: "level:motion", pygame.K_2: "level:jump", pygame.K_3: "level:gap",
            pygame.K_F1: "preset:normal", pygame.K_F2: "preset:heavy", pygame.K_F3: "preset:floaty",
            pygame.K_F4: "preset:slippery", pygame.K_g: "change:gravity", pygame.K_j: "change:jump_speed",
            pygame.K_f: "change:ground_drag"}
    try:
        while running:
            elapsed = min(clock.tick(60) / 1000., .25)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked = renderer.click(event.pos)
                    if clicked: command(clicked)
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_SPACE, pygame.K_w, pygame.K_UP): jump_pending = True
                    elif event.key in keys: command(keys[event.key])
            if not running:
                break
            if paused:
                accumulator = 0.
            elif not base._done:
                accumulator += elapsed
                while accumulator + 1e-10 >= 1 / base.config.control_hz and not base._done:
                    pressed = pygame.key.get_pressed()
                    direction = int(pressed[pygame.K_d] or pressed[pygame.K_RIGHT]) - int(pressed[pygame.K_a] or pressed[pygame.K_LEFT])
                    if reference:
                        action = scripted_action(obs, base.config.level)
                    elif jump_pending:
                        action = Action.LEFT_JUMP if direction < 0 else Action.RIGHT_JUMP if direction > 0 else Action.JUMP
                    else:
                        action = Action.LEFT if direction < 0 else Action.RIGHT if direction > 0 else Action.IDLE
                    jump_pending = False
                    if wrapper:
                        wrapper.control_source = "scripted_reference" if reference else "manual"
                    obs, _, _, _, _ = env.step(action)
                    accumulator -= 1 / base.config.control_hz
                    if base._done: done_since = time.monotonic()
            elif reference and done_since is not None and time.monotonic() - done_since > 1.3:
                current_seed += 1
                reset()
            renderer.paused = paused
            renderer.controller = "SCRIPTED REFERENCE / NOT AI" if reference else "MANUAL CONTROL"
            base.render()
            frames += 1
            if args.frames and frames >= args.frames:
                running = False
    finally:
        env.close()


def inspect_run(args):
    root = Path(args.directory)
    rows = list(csv.DictReader((root / "episodes.csv").open(encoding="utf-8")))
    count = 0
    previous = {}
    with (root / "transitions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            ep = row["episode_id"]
            if row["step"] != previous.get(ep, 0) + 1:
                raise ValueError("Nonconsecutive transition steps")
            previous[ep] = row["step"]
            for key in ("observation", "next_observation"):
                if len(row[key]) != len(OBSERVATION_NAMES) or not np.isfinite(row[key]).all():
                    raise ValueError("Invalid observation")
            if not np.isfinite(row["reward"]): raise ValueError("Nonfinite reward")
            count += 1
    result = {"episodes": len(rows), "transitions": count,
              "successes": sum(r["success"] == "True" for r in rows),
              "outcomes": {k: sum(r["outcome"] == k for r in rows) for k in sorted({r["outcome"] for r in rows})}}
    if sum(int(r["steps"]) for r in rows) != count:
        raise ValueError("Episode summary does not match transitions (is the run still open?)")
    print(json.dumps(result, indent=2))


def replay(args):
    root = Path(args.directory)
    metadata = [json.loads(line) for line in (root / "evaluator.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = next((m for m in metadata if m["episode_id"] == args.episode), None)
    if meta is None: raise ValueError(f"Episode {args.episode} not found")
    cfg = meta["config"]
    cfg["physics_override"] = PhysicsParams(**meta["physics"])
    env = TaklaToonEnv(config=EnvConfig(**cfg), render_mode=None if args.headless else "human")
    checked = 0
    try:
        obs, _ = env.reset(seed=meta["seed"])
        if not args.headless:
            import pygame
            clock = pygame.time.Clock()
        with (root / "transitions.jsonl").open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if row["episode_id"] != args.episode: continue
                np.testing.assert_allclose(obs, row["observation"], atol=1e-5, rtol=0)
                obs, reward, terminated, truncated, _ = env.step(row["action"])
                np.testing.assert_allclose(obs, row["next_observation"], atol=1e-5, rtol=0)
                if abs(reward-row["reward"]) > 1e-7 or terminated != row["terminated"] or truncated != row["truncated"]:
                    raise AssertionError(f"Replay differs at step {row['step']}")
                checked += 1
                if not args.headless:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                            print(f"Replay stopped after {checked} verified transitions")
                            return
                    clock.tick(env.config.control_hz)
    finally:
        env.close()
    print(f"Replay verified: {checked} transitions in episode {args.episode}")


def snapshot(args):
    import pygame
    env = TaklaToonEnv(render_mode="rgb_array", config=config_from(args))
    try:
        obs, _ = env.reset(seed=args.seed)
        for _ in range(args.advance):
            obs, _, done, truncated, _ = env.step(scripted_action(obs, args.level))
            if done or truncated: break
        env.render()  # Initialize the renderer before setting operator-only presentation.
        env._renderer.seed = args.seed
        if args.advance:
            env._renderer.controller = "SCRIPTED REFERENCE / NOT AI"
        rgb = env.render()
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(pygame.surfarray.make_surface(np.transpose(rgb, (1,0,2))), str(output))
        print("Saved screenshot:", output.resolve())
    finally:
        env.close()


def check(args):
    from gymnasium.utils.env_checker import check_env
    env = TaklaToonEnv()
    try:
        check_env(env, skip_render_check=True)
        for level in ("motion", "jump", "gap"):
            obs, _ = env.reset(seed=42, options={"level": level})
            for _ in range(env.config.max_steps):
                obs, _, done, truncated, info = env.step(scripted_action(obs, level))
                if done or truncated: break
            print(f"{level}: {info['event']} in {info['step']} actions")
            if not info["success"]: raise RuntimeError("Reference controller failed")
    finally:
        env.close()
    print("Gymnasium API and reference-controller checks passed")


def main():
    parser = argparse.ArgumentParser(prog="taklatoon", description="TaklaToon physics sandbox. No trained AI included.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--level", choices=["motion", "jump", "gap"], default="gap")
        p.add_argument("--preset", choices=list(PRESETS), default="normal")
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--max-steps", type=int, default=600)
        p.add_argument("--randomize", action="store_true")

    p = sub.add_parser("play", help="Interactive manual sandbox (records by default)")
    common(p)
    p.add_argument("--no-record", action="store_true")
    p.add_argument("--output", type=Path)
    p.add_argument("--autoplay", action="store_true", help="Use a scripted reference, NOT trained AI")
    p.add_argument("--frames", type=int, default=0, help="Exit after N display frames; 0 means unlimited")
    p.set_defaults(func=play)
    p = sub.add_parser("collect", help="Headless transition collection")
    common(p)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--policy", choices=["random", "scripted"], default="random")
    p.add_argument("--output", type=Path)
    p.set_defaults(func=collect)
    p = sub.add_parser("inspect", help="Validate and summarize a closed recorded run")
    p.add_argument("directory", type=Path)
    p.set_defaults(func=inspect_run)
    p = sub.add_parser("replay", help="Replay AND verify recorded transitions")
    p.add_argument("directory", type=Path)
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--headless", action="store_true")
    p.set_defaults(func=replay)
    p = sub.add_parser("snapshot", help="Save a real rendered frame without opening a window")
    common(p)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--advance", type=int, default=0, help="Scripted actions before capture")
    p.set_defaults(func=snapshot)
    p = sub.add_parser("check", help="Check the API and all three reference tasks")
    p.set_defaults(func=check)
    args = parser.parse_args()
    if getattr(args, "episodes", 1) < 1: parser.error("--episodes must be positive")
    if getattr(args, "frames", 0) < 0 or getattr(args, "advance", 0) < 0:
        parser.error("--frames and --advance cannot be negative")
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("Stopped by user")

