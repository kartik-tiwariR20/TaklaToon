"""TaklaToon: environment foundation, without a trained AI controller."""
from gymnasium.envs.registration import register, registry
from .config import EnvConfig, PhysicsParams
from .env import TaklaToonEnv, Action, OBSERVATION_NAMES

__version__ = "0.1.0"
if "TaklaToon-v0" not in registry:
    register(id="TaklaToon-v0", entry_point="taklatoon.env:TaklaToonEnv")

__all__ = ["TaklaToonEnv", "EnvConfig", "PhysicsParams", "Action", "OBSERVATION_NAMES"]

