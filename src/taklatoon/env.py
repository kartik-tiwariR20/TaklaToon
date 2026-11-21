"""The real Pymunk simulator. No renderer or wall clock advances physics."""
from dataclasses import asdict, replace
from enum import IntEnum
import math
import gymnasium as gym
import numpy as np
import pymunk
from .config import EnvConfig, PRESETS, PhysicsParams
from .levels import (LEVELS, WORLD_WIDTH, WORLD_HEIGHT, PLAYER_WIDTH, PLAYER_HEIGHT,
                     MAX_SPEED, ACCELERATION, VELOCITY_SCALE)


class Action(IntEnum):
    IDLE = 0
    LEFT = 1
    RIGHT = 2
    JUMP = 3
    LEFT_JUMP = 4
    RIGHT_JUMP = 5


OBSERVATION_NAMES = ("x", "y", "vx", "vy", "grounded", "goal_dx", "goal_dy", "jump_held", "target_hold_fraction") + tuple(
    f"platform_{i}_{field}" for i in range(3) for field in ("left_dx", "right_dx", "top_dy", "mask")
)
PLAYER_COLLISION, PLATFORM_COLLISION, WALL_COLLISION = 1, 2, 3


class TaklaToonEnv(gym.Env):
    """21 normalized state features, six discrete actions, fixed-step physics.

    Hidden parameters are absent from observations AND info. evaluator_metadata()
    is a deliberate operator-only interface, never an input to the future agent.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, config=None, **config_kwargs):
        if render_mode not in (None, "human", "rgb_array"):
            raise ValueError("render_mode must be None, human, or rgb_array")
        if config is not None and config_kwargs:
            raise ValueError("Pass config OR keyword settings, not both")
        self.config = config or EnvConfig(**config_kwargs)
        if not isinstance(self.config, EnvConfig):
            raise TypeError("config must be EnvConfig")
        self.render_mode = render_mode
        self.metadata = dict(type(self).metadata, render_fps=self.config.control_hz)
        self.action_space = gym.spaces.Discrete(len(Action))
        self.observation_space = gym.spaces.Box(-10.0, 10.0, (len(OBSERVATION_NAMES),), np.float32)
        self._renderer = None
        self._ready = False
        self._done = False
        self._grounded = False
        self._jump_held = False
        self.trail = []
        self.last_action = Action.IDLE
        self.last_info = {}
        self.return_so_far = 0.0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if options:
            allowed = {"level", "preset", "randomize", "physics_override"}
            if set(options) - allowed:
                raise ValueError(f"Unknown reset options: {set(options) - allowed}")
            self.config = replace(self.config, **options)
        self.level = LEVELS[self.config.level]
        p = self.config.physics_override or PRESETS[self.config.preset]
        if self.config.randomize and self.config.physics_override is None:
            p = PhysicsParams(gravity=p.gravity * float(self.np_random.uniform(.85, 1.15)),
                              jump_speed=p.jump_speed * float(self.np_random.uniform(.95, 1.05)),
                              ground_drag=p.ground_drag * float(self.np_random.uniform(.8, 1.2)))
        self.physics = p
        self.space = pymunk.Space()
        self.space.gravity = (0, -p.gravity)
        self.space.iterations = 20
        self.space.collision_slop = .05
        for platform in self.level.platforms:
            shape = pymunk.Poly(self.space.static_body, [(platform.left, -200),
                (platform.right, -200), (platform.right, platform.top), (platform.left, platform.top)])
            shape.collision_type = PLATFORM_COLLISION
            shape.friction = 0.0  # Ground drag below is the ONLY variable stopping mechanism.
            self.space.add(shape)
        for x0, x1 in ((-50, 0), (WORLD_WIDTH, WORLD_WIDTH + 50)):
            wall = pymunk.Poly(self.space.static_body, [(x0, -250), (x1, -250), (x1, 850), (x0, 850)])
            wall.collision_type = WALL_COLLISION
            wall.friction = 0
            self.space.add(wall)
        self.body = pymunk.Body(1, math.inf)
        self.body.position = self.level.spawn
        self.shape = pymunk.Poly.create_box(self.body, (PLAYER_WIDTH, PLAYER_HEIGHT), radius=0)
        self.shape.collision_type = PLAYER_COLLISION
        self.shape.friction = 0.0
        self.shape.elasticity = 0.0
        self.space.add(self.body, self.shape)
        self.space.on_collision(PLAYER_COLLISION, PLATFORM_COLLISION, post_solve=self._contact)
        self.steps = 0
        self._grounded = True
        self._jump_held = False
        self._done = False
        self._ready = True
        self._target_hold = 0
        self.return_so_far = 0.0
        self.trail = [tuple(self.body.position)]
        self.last_action = Action.IDLE
        info = self._info("reset", False, {"step": 0., "progress": 0., "terminal": 0.})
        self.last_info = info
        if self.render_mode == "human":
            self.render()
        return self._observation(), info

    def _contact(self, arbiter, space, data):
        # Handler orders shapes player -> platform. A floor normal points down.
        if arbiter.normal.y < -.5:
            self._grounded = True

    def step(self, action):
        if not self._ready:
            raise RuntimeError("Call reset() before step()")
        if self._done:
            raise RuntimeError("Episode ended. Call reset() before another step()")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action!r}; expected an integer from 0 to 5")
        action = Action(int(action))
        self.last_action = action
        direction = -1 if action in (Action.LEFT, Action.LEFT_JUMP) else 1 if action in (Action.RIGHT, Action.RIGHT_JUMP) else 0
        jumping = action in (Action.JUMP, Action.LEFT_JUMP, Action.RIGHT_JUMP)
        if jumping and not self._jump_held and self._grounded:
            self.body.velocity = (self.body.velocity.x, self.physics.jump_speed)
            self._grounded = False
        self._jump_held = jumping
        before = abs(self.level.goal[0] - self.body.position.x)
        dt = 1 / self.config.physics_hz
        for _ in range(self.config.physics_hz // self.config.control_hz):
            grounded = self._grounded
            vx = self.body.velocity.x + direction * ACCELERATION * (1 if grounded else .55) * dt
            if grounded:
                vx *= math.exp(-self.physics.ground_drag * dt)
            self.body.velocity = (float(np.clip(vx, -MAX_SPEED, MAX_SPEED)), self.body.velocity.y)
            self._grounded = False
            self.space.step(dt)
        self.steps += 1
        self.trail.append(tuple(self.body.position))
        self.trail = self.trail[-180:]
        distance = abs(self.level.goal[0] - self.body.position.x)
        inside = distance <= 42 and self._grounded
        if self.config.level == "motion":
            inside = inside and abs(self.body.velocity.x) < 35
            self._target_hold = self._target_hold + 1 if inside else 0
            success = self._target_hold >= 6
        else:
            success = inside
        fallen = self.body.position.y < -80
        terminated = bool(success or fallen)
        truncated = bool(self.steps >= self.config.max_steps and not terminated)
        event = "success" if success else "fall" if fallen else "timeout" if truncated else "running"
        components = {"step": -.01, "progress": 8.0 * (before - distance) / WORLD_WIDTH,
                      "terminal": 100.0 if success else -20.0 if fallen else 0.0}
        reward = float(sum(components.values()))
        self.return_so_far += reward
        self._done = terminated or truncated
        info = self._info(event, success, components)
        self.last_info = info
        if self.render_mode == "human":
            self.render()
        return self._observation(), reward, terminated, truncated, info

    def _observation(self):
        x, y = self.body.position
        vx, vy = self.body.velocity
        values = [x / WORLD_WIDTH, y / WORLD_HEIGHT, vx / MAX_SPEED, vy / VELOCITY_SCALE,
                  float(self._grounded), (self.level.goal[0] - x) / WORLD_WIDTH,
                  (self.level.goal[1] - y) / WORLD_HEIGHT,
                  float(self._jump_held), min(self._target_hold / 6., 1.)]
        for i in range(3):
            if i < len(self.level.platforms):
                p = self.level.platforms[i]
                values += [(p.left - x) / WORLD_WIDTH, (p.right - x) / WORLD_WIDTH,
                           (p.top - y) / WORLD_HEIGHT, 1.0]
            else:
                values += [0., 0., 0., 0.]
        return np.asarray(values, dtype=np.float32)

    def _info(self, event, success, components):
        return {"event": event, "success": bool(success), "step": self.steps,
                "sim_time": self.steps / self.config.control_hz, "reward_components": components}

    def evaluator_metadata(self):
        if not self._ready:
            raise RuntimeError("Call reset() first")
        return {"level": self.config.level, "preset": self.config.preset,
                "randomize": self.config.randomize, "physics": asdict(self.physics),
                "config": {**asdict(self.config), "physics_override": asdict(self.physics)},
                "spawn": list(self.level.spawn), "goal": list(self.level.goal),
                "platforms": [asdict(p) for p in self.level.platforms]}

    def render(self):
        if self.render_mode is None:
            return None
        if not self._ready:
            raise RuntimeError("Call reset() before render()")
        if self._renderer is None:
            from .renderer import Renderer
            self._renderer = Renderer(human=self.render_mode == "human")
        return self._renderer.draw(self)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

