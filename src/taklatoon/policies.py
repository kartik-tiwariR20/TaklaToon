"""Simple observable-state reference controller. This is NOT trained AI."""
from .env import Action
from .levels import WORLD_WIDTH, MAX_SPEED


def scripted_action(observation, level):
    x = float(observation[0]) * WORLD_WIDTH
    vx = float(observation[2]) * MAX_SPEED
    dx = float(observation[5]) * WORLD_WIDTH
    grounded = observation[4] > .5
    if level == "motion":
        # Basic PD braking using observable state; no access to hidden coefficients.
        demand = 3 * dx - 2.5 * vx
        if abs(dx) < 18 and abs(vx) < 18:
            return Action.IDLE
        return Action.RIGHT if demand > 35 else Action.LEFT if demand < -35 else Action.IDLE
    edge = float(observation[10]) * WORLD_WIDTH + x  # visible first-platform right bound
    if grounded and x < edge + 10 and x >= edge - 28:
        return Action.RIGHT_JUMP
    if dx < -20:
        return Action.LEFT
    return Action.RIGHT

