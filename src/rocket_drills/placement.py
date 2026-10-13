"""Helpers that write the ball and cars into an RLGym ``GameState``.

The drills are built from these, and they are public so that you can write your own: create (or
receive) a state, call :func:`set_ball` and :func:`set_car`, and call :func:`mark_jumped` for
every car that should start in the air with a jump behind it.
"""

from __future__ import annotations

import numpy as np
from rlgym.rocket_league.api import Car, GameConfig, GameState, PhysicsObject
from rlgym.rocket_league.common_values import (
    BLUE_TEAM,
    BOOST_LOCATIONS,
    JUMP_MAX_TIME,
    OCTANE,
    ORANGE_TEAM,
)

from rocket_drills.geometry import UP, Vector, rotation_matrix


def _vector(values: Vector) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape != (3,):
        raise ValueError(f"expected three numbers, got {array.shape[0]}")
    return array.copy()


def empty_state(blue: int = 1, orange: int = 1, hitbox: int = OCTANE) -> GameState:
    """Create a state with ``blue`` and ``orange`` cars that a drill can be applied to.

    This is what ``RocketSimEngine.create_base_state()`` followed by RLGym's
    ``FixedTeamSizeMutator`` gives, without needing the simulator: handy for looking at a drill,
    or for testing your own. The cars are keyed ``"blue-0"``, ``"orange-0"`` and so on, sit at the
    origin and must be placed by a drill before the state is used.

    Args:
        blue: Number of blue cars.
        orange: Number of orange cars.
        hitbox: Hitbox of every car, one of the constants in RLGym's ``common_values``.
    """
    state = GameState()
    state.tick_count = 0
    state.goal_scored = False
    state.config = GameConfig()
    state.config.gravity = 1.0
    state.config.boost_consumption = 1.0
    state.config.dodge_deadzone = 0.5
    state.ball = PhysicsObject()
    state.boost_pad_timers = np.zeros(len(BOOST_LOCATIONS), dtype=np.float32)
    state.cars = {}
    for team, name, count in ((BLUE_TEAM, "blue", blue), (ORANGE_TEAM, "orange", orange)):
        for index in range(count):
            car = Car()
            car.team_num = team
            car.hitbox_type = hitbox
            car.physics = PhysicsObject()
            set_car(car, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0), boost=0.0)
            state.cars[f"{name}-{index}"] = car
    return state


def set_ball(
    state: GameState,
    position: Vector,
    velocity: Vector,
    angular_velocity: Vector = (0.0, 0.0, 0.0),
) -> None:
    """Place the ball.

    RocketSim leaves a ball whose velocity is exactly zero where it is, hanging in the air if that
    is where it was put, until something touches it. Give an airborne ball some velocity.
    """
    state.ball.position = _vector(position)
    state.ball.linear_velocity = _vector(velocity)
    state.ball.angular_velocity = _vector(angular_velocity)


def set_car(
    car: Car,
    position: Vector,
    velocity: Vector,
    forward: Vector,
    up: Vector = UP,
    *,
    boost: float,
    on_ground: bool = True,
    angular_velocity: Vector = (0.0, 0.0, 0.0),
) -> None:
    """Place a car and clear everything it remembers about jumps, flips and demolitions.

    Args:
        car: The car to write, as found in ``GameState.cars``.
        position: Where its origin goes.
        velocity: Its linear velocity.
        forward: Direction of its nose.
        up: Direction of its roof. It only has to be roughly right: see
            :func:`rocket_drills.geometry.rotation_matrix`.
        boost: Boost in the tank, 0 to 100.
        on_ground: Whether its wheels rest on a surface: the floor, a wall or the ceiling. Pass
            ``False`` for a car in the air, which then counts as never having jumped and keeps its
            dodge for as long as it stays up; call :func:`mark_jumped` to give it a jump.
        angular_velocity: Its angular velocity.
    """
    car.physics.position = _vector(position)
    car.physics.linear_velocity = _vector(velocity)
    car.physics.angular_velocity = _vector(angular_velocity)
    car.physics.rotation_mtx = rotation_matrix(forward, up)
    car.boost_amount = float(np.clip(boost, 0.0, 100.0))
    car.on_ground = bool(on_ground)

    car.ball_touches = 0
    car.bump_victim_id = None
    car.demo_respawn_timer = 0.0
    car.supersonic_time = 0.0
    car.boost_active_time = 0.0
    car.handbrake = 0.0

    car.is_jumping = False
    car.has_jumped = False
    car.is_holding_jump = False
    car.jump_time = 0.0

    car.has_flipped = False
    car.has_double_jumped = False
    car.air_time_since_jump = 0.0
    car.flip_time = 0.0
    car.flip_torque = np.zeros(3, dtype=np.float32)

    car.is_autoflipping = False
    car.autoflip_timer = 0.0
    car.autoflip_direction = 0.0


def mark_jumped(car: Car, air_time: float) -> None:
    """Give an airborne car a jump in its past.

    A car that a state mutator puts in the air has never jumped as far as the simulator knows, so
    its dodge never runs out: it can flip whenever it likes, and wheels on the ball give it
    nothing. That is not what a car that got up there by jumping looks like. This marks the car as
    having jumped and released the button, ``air_time`` seconds ago, which leaves it a dodge for
    another ``1.25 - air_time`` seconds and none after that, until a flip reset or a landing.

    Args:
        car: A car that was placed with ``on_ground=False``.
        air_time: Seconds in the air since the jump ended (RLGym's ``Car.air_time_since_jump``).

    Raises:
        ValueError: If ``air_time`` is negative or the car is on the ground.
    """
    if air_time < 0.0:
        raise ValueError("air_time must not be negative")
    if car.on_ground:
        raise ValueError("only a car placed with on_ground=False can have a jump behind it")
    car.has_jumped = True
    car.is_jumping = False
    car.is_holding_jump = False
    car.jump_time = JUMP_MAX_TIME
    car.air_time_since_jump = float(air_time)
    car.has_flipped = False
    car.has_double_jumped = False
    car.flip_time = 0.0
