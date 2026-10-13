"""Helpers shared by the tests: drill starts, hand-built state sequences, simulator loops."""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterator, Mapping

import numpy as np
from rlgym.rocket_league.api import Car, GameState
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z, SIDE_WALL_X

from rocket_drills.drills import Drill, DrillInfo
from rocket_drills.geometry import CORNER_SUM
from rocket_drills.placement import empty_state, set_ball, set_car
from rocket_drills.scripted import Policy, idle

# --- drills without the simulator ---------------------------------------------------------------


def starts(
    drill: Drill, count: int, sizes: tuple[int, int] = (1, 1), hitbox: int = 0
) -> Iterator[tuple[GameState, DrillInfo, Car]]:
    """Apply a drill ``count`` times; yield the state, the drill's info and the drill car."""
    for _ in range(count):
        state = empty_state(*sizes, hitbox=hitbox)
        shared_info: dict = {}
        drill.apply(state, shared_info)
        info = shared_info["drill"]
        yield state, info, state.cars[info.agent]


def surface_distance(position: np.ndarray) -> float:
    """Distance from a point to the nearest of the arena's bounding planes."""
    x, y, z = abs(position[0]), abs(position[1]), position[2]
    corner = (CORNER_SUM - x - y) / math.sqrt(2.0)
    return float(min(SIDE_WALL_X - x, BACK_WALL_Y - y, corner, z, CEILING_Z - z))


def to_ball(car: Car, state: GameState) -> tuple[np.ndarray, float]:
    """Unit direction and distance from a car to the ball."""
    offset = state.ball.position - car.physics.position
    distance = float(np.linalg.norm(offset))
    return offset / distance, distance


# --- hand-built state sequences for the detectors -----------------------------------------------


def quiet_state() -> GameState:
    """A 1v1 with the ball at rest in midfield and both cars parked far from it and each other."""
    state = empty_state(1, 1)
    set_ball(state, (0.0, 0.0, 93.15), (0.0, 0.0, 0.0))
    set_car(state.cars["blue-0"], (2000.0, -3000.0, 17.0), (0, 0, 0), (0.0, 1.0, 0.0), boost=50.0)
    set_car(
        state.cars["orange-0"], (-2000.0, 3000.0, 17.0), (0, 0, 0), (0.0, -1.0, 0.0), boost=50.0
    )
    return state


def after(state: GameState, ticks: int = 8) -> GameState:
    """A copy of ``state`` some ticks later, with the per-step touch counts cleared."""
    later = copy.deepcopy(state)
    later.tick_count = state.tick_count + ticks
    for car in later.cars.values():
        car.ball_touches = 0
    return later


def fly(car: Car, position, velocity=(0.0, 0.0, 0.0), forward=(0.0, 1.0, 0.0), up=(0.0, 0.0, 1.0)):
    """Put a car in the air, keeping what it remembers about jumps."""
    remembered = (car.has_jumped, car.has_flipped, car.has_double_jumped, car.flip_time)
    set_car(car, position, velocity, forward, up, boost=car.boost_amount, on_ground=False)
    car.has_jumped, car.has_flipped, car.has_double_jumped, car.flip_time = remembered


def skills_of(events, agent=None) -> list[str]:
    """Names of the events, of one car or of all."""
    return [event.skill for event in events if agent is None or event.agent == agent]


# --- simulator loops ----------------------------------------------------------------------------


def begin(engine, drill, sizes: tuple[int, int] = (1, 1), prepare: Callable | None = None):
    """Start an episode from a drill; return the engine's first state, the info and shared_info.

    ``prepare(state, info)`` may change the drill's state before the engine takes it.
    """
    state = engine.create_base_state()
    shared_info: dict = {}
    built = empty_state(*sizes)
    state.cars = built.cars
    drill.apply(state, shared_info)
    info = shared_info["drill"]
    if prepare is not None:
        prepare(state, info)
    return engine.set_state(state, shared_info), info, shared_info


def play(
    engine,
    state: GameState,
    shared_info: dict,
    policies: Mapping[object, Policy],
    seconds: float,
    tick_skip: int = 8,
) -> Iterator[GameState]:
    """Step the engine, each car under its policy (idle if it has none); yield every new state."""
    for _ in range(math.ceil(seconds * 120 / tick_skip - 1e-6)):
        actions = {
            agent: np.repeat(policies.get(agent, idle)(state, agent)[None], tick_skip, axis=0)
            for agent in state.cars
        }
        state = engine.step(actions, shared_info)
        yield state
