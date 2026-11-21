from dataclasses import dataclass

WORLD_WIDTH = 1200.0
WORLD_HEIGHT = 600.0
PLAYER_WIDTH = 32.0
PLAYER_HEIGHT = 44.0
MAX_SPEED = 300.0
ACCELERATION = 1800.0
VELOCITY_SCALE = 1200.0


@dataclass(frozen=True)
class Platform:
    left: float
    right: float
    top: float


@dataclass(frozen=True)
class Level:
    name: str
    title: str
    instruction: str
    platforms: tuple[Platform, ...]
    spawn: tuple[float, float]
    goal: tuple[float, float]


LEVELS = {
    "motion": Level("motion", "MotionLab", "Stop inside the target zone.",
                    (Platform(0, 1200, 150),), (110, 172), (1000, 150)),
    "jump": Level("jump", "JumpLab", "Jump onto the raised platform and reach the flag.",
                  (Platform(0, 470, 150), Platform(600, 1200, 210)), (110, 172), (1090, 210)),
    "gap": Level("gap", "GapRunner", "Cross the gap. Reach the flag on the far side.",
                 (Platform(0, 440, 150), Platform(650, 1200, 150)), (110, 172), (1090, 150)),
}

