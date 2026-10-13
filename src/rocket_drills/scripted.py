"""Small scripted policies: baselines for the drills, and the controllers the tests use.

A policy here is a function ``policy(state, agent)`` that returns the eight RLGym controls for one
car: throttle, steer, pitch, yaw, roll, jump, boost, handbrake. All of them are stateless (they
decide from the state alone), so one function can drive any number of cars. They are deliberately
simple; use them as the floor that a trained policy should clear in
:func:`rocket_drills.evaluation.evaluate_drill`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable

import numpy as np
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BOOST, JUMP, PITCH, STEER, THROTTLE, YAW

from rocket_drills.geometry import CAR_REST_HEIGHT

#: A policy: the game state and the car to control in, eight controls out.
Policy = Callable[[GameState, Hashable], np.ndarray]


def _controls() -> np.ndarray:
    return np.zeros(8, dtype=np.float32)


def idle(state: GameState, agent: Hashable) -> np.ndarray:
    """Do nothing."""
    return _controls()


def drive_at_ball(state: GameState, agent: Hashable) -> np.ndarray:
    """Steer at the ball along the floor at full throttle, boosting while roughly facing it."""
    car = state.cars[agent]
    to_ball = state.ball.position - car.physics.position
    forward = car.physics.forward
    angle = math.atan2(
        forward[0] * to_ball[1] - forward[1] * to_ball[0],
        forward[0] * to_ball[0] + forward[1] * to_ball[1],
    )
    controls = _controls()
    controls[THROTTLE] = 1.0
    controls[STEER] = controls[YAW] = float(np.clip(3.0 * angle, -1.0, 1.0))
    controls[BOOST] = float(abs(angle) < 0.3)
    return controls


def boosted_jump(state: GameState, agent: Hashable) -> np.ndarray:
    """Jump, hold the jump for its full 0.2 s, lean back until the nose is well up, and boost.

    The plain way to go up for a ball ahead: no second jump and no steering.
    """
    car = state.cars[agent]
    controls = _controls()
    controls[THROTTLE] = 1.0
    controls[BOOST] = 1.0
    controls[JUMP] = float(car.on_ground or car.is_jumping)
    controls[PITCH] = float(car.physics.forward[2] < 0.85)
    return controls


def forward_dodge(state: GameState, agent: Hashable) -> np.ndarray:
    """Dodge forward as soon as the car has a dodge to use in the air, and do nothing else."""
    controls = _controls()
    if state.cars[agent].can_flip:
        controls[JUMP] = 1.0
        controls[PITCH] = -1.0
    return controls


def wavedash(state: GameState, agent: Hashable) -> np.ndarray:
    """Come down nose up and dodge forward just before the wheels touch.

    While the car falls it leans back to about 0.3 rad; when its wheels are within a tenth of a
    second of the floor it dodges forward. On the ground it holds the throttle.
    """
    car = state.cars[agent]
    controls = _controls()
    controls[THROTTLE] = 1.0
    if car.on_ground:
        return controls
    rest = CAR_REST_HEIGHT.get(car.hitbox_type, 17.03)
    sink = -float(car.physics.linear_velocity[2])
    clearance = float(car.physics.position[2]) - rest
    if car.can_flip and sink > 0.0 and clearance < 30.0 + 0.1 * sink:
        controls[JUMP] = 1.0
        controls[PITCH] = -1.0
    else:
        controls[PITCH] = float(np.clip(4.0 * (0.3 - car.physics.forward[2]), -1.0, 1.0))
    return controls


def carry(state: GameState, agent: Hashable) -> np.ndarray:
    """Keep a ball that rides on the roof there: match the ball's speed, steer under it."""
    car = state.cars[agent]
    physics = car.physics
    offset = state.ball.position - physics.position
    ahead = float(offset @ physics.forward)
    aside = float(offset @ physics.right)
    closing = float((state.ball.linear_velocity - physics.linear_velocity) @ physics.forward)
    controls = _controls()
    controls[THROTTLE] = float(np.clip(0.05 * (ahead - 10.0) + 0.02 * closing, -1.0, 1.0))
    controls[STEER] = float(np.clip(0.03 * aside, -1.0, 1.0))
    return controls
