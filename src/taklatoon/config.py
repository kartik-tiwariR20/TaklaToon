"""Validated, serializable settings. Hidden physics is evaluator metadata."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PhysicsParams:
    gravity: float = 1000.0
    jump_speed: float = 520.0
    ground_drag: float = 5.0

    def __post_init__(self):
        ranges = {"gravity": (300, 2200), "jump_speed": (250, 800), "ground_drag": (0, 15)}
        for name, (lo, hi) in ranges.items():
            value = getattr(self, name)
            if not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be finite and between {lo} and {hi}")


PRESETS = {
    "normal": PhysicsParams(),
    "heavy": PhysicsParams(gravity=1400),
    "floaty": PhysicsParams(gravity=700),
    "slippery": PhysicsParams(ground_drag=0.65),
}


@dataclass(frozen=True)
class EnvConfig:
    level: str = "gap"
    preset: str = "normal"
    randomize: bool = False
    max_steps: int = 600
    control_hz: int = 30
    physics_hz: int = 120
    physics_override: PhysicsParams | None = None

    def __post_init__(self):
        if self.level not in {"motion", "jump", "gap"}:
            raise ValueError("level must be motion, jump, or gap")
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS)}")
        if type(self.randomize) is not bool:
            raise ValueError("randomize must be a boolean")
        for name in ("max_steps", "control_hz", "physics_hz"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.physics_hz < self.control_hz or self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        if self.control_hz > 240 or self.physics_hz > 1000:
            raise ValueError("Unsupported timestep settings")
        if self.physics_override is not None and not isinstance(self.physics_override, PhysicsParams):
            raise ValueError("physics_override must be PhysicsParams or None")

