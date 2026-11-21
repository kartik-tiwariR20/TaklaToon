import json
from dataclasses import replace
import numpy as np
import pytest
import gymnasium as gym
from gymnasium.utils.env_checker import check_env
from taklatoon import TaklaToonEnv, EnvConfig, Action, PhysicsParams
from taklatoon.config import PRESETS
from taklatoon.policies import scripted_action


def test_gymnasium_contract():
    env = TaklaToonEnv()
    try:
        check_env(env, skip_render_check=True)
        obs, info = env.reset(seed=42)
        assert obs.shape == (21,) and obs.dtype == np.float32
        assert env.observation_space.contains(obs)
    finally:
        env.close()


def test_registered_environment():
    env = gym.make("TaklaToon-v0", level="motion")
    try:
        obs, info = env.reset(seed=1)
        assert env.observation_space.contains(obs)
        assert len(env.step(0)) == 5
    finally:
        env.close()


def trajectory(seed, randomize=True):
    env = TaklaToonEnv(config=EnvConfig(randomize=randomize))
    try:
        observations = [env.reset(seed=seed)[0]]
        rewards = []
        for a in [2]*20 + [5] + [2]*25:
            obs, reward, done, truncated, _ = env.step(a)
            observations.append(obs)
            rewards.append(reward)
            if done or truncated: break
        return np.asarray(observations), rewards, env.evaluator_metadata()
    finally:
        env.close()


def test_seed_reproducibility_and_variation():
    a, ar, am = trajectory(42)
    b, br, bm = trajectory(42)
    np.testing.assert_array_equal(a, b)
    assert ar == br and am == bm
    _, _, other = trajectory(43)
    assert other["physics"] != am["physics"]


def test_hidden_parameters_are_not_agent_inputs():
    env = TaklaToonEnv()
    try:
        _, info = env.reset(seed=1)
        _, _, _, _, next_info = env.step(0)
        forbidden = ("gravity", "jump_speed", "ground_drag", "preset", "seed", "level")
        assert all(key not in json.dumps([info, next_info]) for key in forbidden)
        assert "gravity" in env.evaluator_metadata()["physics"]
    finally:
        env.close()


def test_gravity_changes_observed_trajectory():
    heights = []
    for preset in ("floaty", "heavy"):
        env = TaklaToonEnv(level="motion", preset=preset)
        try:
            env.reset()
            env.step(Action.JUMP)
            for _ in range(12): env.step(Action.IDLE)
            heights.append(env.body.position.y)
        finally:
            env.close()
    assert heights[0] > heights[1] + 30


def test_ground_contact_and_no_air_double_jump():
    env = TaklaToonEnv(level="motion")
    try:
        env.reset()
        env.step(Action.JUMP)
        env.step(Action.IDLE)
        before = env.body.velocity.y
        env.step(Action.JUMP)
        assert env.body.velocity.y < before  # Airborne jump cannot reset vertical speed.
        for _ in range(50): env.step(Action.IDLE)
        assert env._grounded
        assert abs(env.body.position.y - 172) < 1
        env.step(Action.JUMP)
        assert env.body.velocity.y > 450
    finally:
        env.close()


def test_held_jump_does_not_retrigger_after_landing():
    env = TaklaToonEnv(level="motion")
    try:
        env.reset()
        for _ in range(70): env.step(Action.JUMP)
        assert env._grounded
        assert abs(env.body.velocity.y) < 15
    finally:
        env.close()


@pytest.mark.parametrize("level", ["motion", "jump", "gap"])
@pytest.mark.parametrize("preset", list(PRESETS))
def test_scripted_reference_can_finish(level, preset):
    env = TaklaToonEnv(level=level, preset=preset)
    try:
        obs, _ = env.reset(seed=42)
        for _ in range(600):
            obs, reward, done, truncated, info = env.step(scripted_action(obs, level))
            if done or truncated: break
        assert info["success"], (level, preset, info, tuple(env.body.position))
        assert reward > 99
        with pytest.raises(RuntimeError): env.step(0)
    finally:
        env.close()


def test_time_limit_is_truncated_not_terminal():
    env = TaklaToonEnv(max_steps=2)
    try:
        env.reset()
        env.step(0)
        _, _, terminated, truncated, info = env.step(0)
        assert truncated and not terminated and info["event"] == "timeout"
        with pytest.raises(RuntimeError): env.step(0)
        obs, info = env.reset()
        assert info["step"] == 0 and env.return_so_far == 0
        assert env.body.velocity.length == 0
    finally:
        env.close()


def test_falling_is_terminal():
    env = TaklaToonEnv()
    try:
        env.reset()
        for _ in range(200):
            _, reward, done, truncated, info = env.step(Action.RIGHT)
            if done: break
        assert done and not truncated and info["event"] == "fall" and reward < -19
    finally:
        env.close()


def test_drag_reduces_stopping_distance():
    positions = []
    for preset in ("normal", "slippery"):
        env = TaklaToonEnv(level="motion", preset=preset)
        try:
            env.reset()
            for _ in range(20): env.step(2)
            start = env.body.position.x
            for _ in range(20): env.step(0)
            positions.append(env.body.position.x - start)
        finally:
            env.close()
    assert positions[1] > positions[0] + 50


def test_rendering_does_not_advance_physics():
    env = TaklaToonEnv(render_mode="rgb_array")
    try:
        obs, _ = env.reset(seed=42)
        first = env.render()
        for _ in range(5): env.render()
        np.testing.assert_array_equal(obs, env._observation())
        assert first.shape == (900, 1440, 3) and first.dtype == np.uint8
        assert env.steps == 0
    finally:
        env.close()


@pytest.mark.parametrize("action", [-1, 6, 0.5, "jump"])
def test_invalid_actions_rejected(action):
    env = TaklaToonEnv()
    try:
        env.reset()
        with pytest.raises(ValueError): env.step(action)
    finally:
        env.close()


def test_invalid_configuration_rejected():
    with pytest.raises(ValueError): EnvConfig(physics_hz=119)
    with pytest.raises(ValueError): EnvConfig(level="unknown")
    with pytest.raises(ValueError): PhysicsParams(gravity=float("nan"))
    with pytest.raises(ValueError): EnvConfig(max_steps=0)

