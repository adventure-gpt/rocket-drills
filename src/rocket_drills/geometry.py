"""Field constants, rotations and ball-flight helpers shared by the drills and the skill counters.

Everything uses RLGym's conventions: unreal units (uu), seconds and radians; ``x`` runs across the
pitch, ``y`` along it and ``z`` up. Blue defends the goal at ``-y`` and attacks ``+y``, orange the
other way round. Constants that RLGym defines are imported from
``rlgym.rocket_league.common_values``; the few it lacks are defined here and say how they were
obtained.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BALL_RADIUS,
    BLUE_TEAM,
    BREAKOUT,
    CEILING_Z,
    CORNER_CATHETUS_LENGTH,
    DOMINUS,
    GOAL_CENTER_TO_POST,
    GOAL_HEIGHT,
    GRAVITY,
    HYBRID,
    MERC,
    OCTANE,
    PLANK,
    SIDE_WALL_X,
)

Vector = Sequence[float] | np.ndarray

#: ``|x| + |y|`` on the diagonal corner walls.
CORNER_SUM: float = SIDE_WALL_X + BACK_WALL_Y - CORNER_CATHETUS_LENGTH

#: Linear drag on the ball in free flight, per second. RocketSim multiplies the ball's velocity by
#: ``0.97 ** dt`` every tick; a flight computed with this figure is within 6 uu of the simulated
#: one after two seconds (measured, RocketSim 2.2.1).
BALL_DRAG: float = -math.log(0.97)

#: Height of a car's origin when it rests on its wheels, by hitbox (measured, RocketSim 2.2.1).
CAR_REST_HEIGHT: dict[int, float] = {
    OCTANE: 17.03,
    DOMINUS: 17.07,
    PLANK: 18.67,
    BREAKOUT: 18.36,
    HYBRID: 17.03,
    MERC: 18.81,
}

#: Height of the ball's centre when it rests on the roof of a car that stands on the floor, by
#: hitbox (measured, RocketSim 2.2.1).
BALL_ON_ROOF_HEIGHT: dict[int, float] = {
    OCTANE: 150.3,
    DOMINUS: 142.0,
    PLANK: 139.0,
    BREAKOUT: 138.5,
    HYBRID: 148.5,
    MERC: 155.6,
}

#: Distance from a flat wall at which the ball's centre sits when the ball rolls on it (the ball's
#: radius plus the simulator's collision margin).
BALL_WALL_OFFSET: float = BALL_RADIUS + 3.5

#: Half-width and height of the goal mouth that the centre of the ball must pass through.
GOAL_MOUTH_HALF_WIDTH: float = GOAL_CENTER_TO_POST - BALL_RADIUS
GOAL_MOUTH_HEIGHT: float = GOAL_HEIGHT - BALL_RADIUS

#: How far ahead a shot or a save is judged: the ball must reach the goal line within this time.
ON_TARGET_HORIZON: float = 2.5

UP: np.ndarray = np.array([0.0, 0.0, 1.0])


def team_sign(team: int) -> float:
    """Return ``+1.0`` for blue, which attacks ``+y``, and ``-1.0`` for orange."""
    return 1.0 if team == BLUE_TEAM else -1.0


def attacked_goal_y(team: int) -> float:
    """Return the ``y`` of the goal line that ``team`` attacks."""
    return team_sign(team) * BACK_WALL_Y


def defended_goal_y(team: int) -> float:
    """Return the ``y`` of the goal line that ``team`` defends."""
    return -team_sign(team) * BACK_WALL_Y


def unit(vector: Vector) -> np.ndarray:
    """Return ``vector`` scaled to length one.

    Raises:
        ValueError: If the vector has no length.
    """
    array = np.asarray(vector, dtype=np.float64)
    length = float(np.linalg.norm(array))
    if length < 1e-9:
        raise ValueError("cannot normalise a zero vector")
    return array / length


def rotation_matrix(forward: Vector, up: Vector = UP) -> np.ndarray:
    """Build the rotation matrix RLGym expects from a car's forward and up directions.

    The columns of the result are the car's forward, right and up axes, which is the layout of
    ``PhysicsObject.rotation_mtx``. ``up`` only has to point roughly the right way: the part of it
    along ``forward`` is removed.

    Args:
        forward: Direction of the car's nose.
        up: Direction of the car's roof.

    Returns:
        A 3x3 ``float32`` matrix.

    Raises:
        ValueError: If ``up`` is parallel to ``forward``.
    """
    fwd = unit(forward)
    roof = np.asarray(up, dtype=np.float64)
    roof = roof - fwd * float(roof @ fwd)
    if float(np.linalg.norm(roof)) < 1e-6:
        raise ValueError("up must not be parallel to forward")
    roof = unit(roof)
    right = np.cross(roof, fwd)
    return np.stack([fwd, right, roof], axis=1).astype(np.float32)


def ball_flight(
    position: Vector, velocity: Vector, time: float, drag: float = BALL_DRAG
) -> tuple[np.ndarray, np.ndarray]:
    """Predict the ball's free flight: gravity and drag, no bounces.

    Args:
        position: Where the ball is now.
        velocity: How it is moving now.
        time: Seconds ahead.
        drag: Linear drag per second; ``0`` gives a plain parabola.

    Returns:
        The position and the velocity after ``time`` seconds.
    """
    start = np.asarray(position, dtype=np.float64)
    speed = np.asarray(velocity, dtype=np.float64)
    gravity = np.array([0.0, 0.0, -GRAVITY])
    if drag <= 0.0:
        return start + speed * time + 0.5 * gravity * time * time, speed + gravity * time
    decay = math.exp(-drag * time)
    terminal = gravity / drag
    end = start + (speed - terminal) * (1.0 - decay) / drag + terminal * time
    return end, (speed - terminal) * decay + terminal


def launch_velocity(
    start: Vector, target: Vector, time: float, drag: float = BALL_DRAG
) -> np.ndarray:
    """Return the velocity that takes the ball from ``start`` to ``target`` in ``time`` seconds.

    The inverse of :func:`ball_flight`: gravity and drag, no bounces.

    Raises:
        ValueError: If ``time`` is not positive.
    """
    if time <= 0.0:
        raise ValueError("time must be positive")
    origin = np.asarray(start, dtype=np.float64)
    goal = np.asarray(target, dtype=np.float64)
    gravity = np.array([0.0, 0.0, -GRAVITY])
    if drag <= 0.0:
        return (goal - origin - 0.5 * gravity * time * time) / time
    terminal = gravity / drag
    return (goal - origin - terminal * time) * drag / (1.0 - math.exp(-drag * time)) + terminal


def goal_crossing(
    position: Vector, velocity: Vector, goal_y: float, horizon: float = ON_TARGET_HORIZON
) -> tuple[float, float, float] | None:
    """Find where the ball's flight crosses the goal plane at ``goal_y``.

    The flight is a plain parabola under gravity: no drag, no bounces and no walls.

    Returns:
        ``(time, x, z)`` of the crossing, or ``None`` when the ball is not moving toward that
        plane or would take ``horizon`` seconds or longer to get there.
    """
    y_speed = float(velocity[1])
    if abs(y_speed) < 1.0:
        return None
    time = (goal_y - float(position[1])) / y_speed
    if not 0.0 < time < horizon:
        return None
    x = float(position[0]) + float(velocity[0]) * time
    z = float(position[2]) + float(velocity[2]) * time - 0.5 * GRAVITY * time * time
    return time, x, z


def on_target(
    position: Vector, velocity: Vector, goal_y: float, horizon: float = ON_TARGET_HORIZON
) -> bool:
    """Tell whether the ball is heading into the goal at ``goal_y``.

    True when the ball's flight under gravity alone (no drag, bounces or walls) crosses that goal
    line within ``horizon`` seconds, between the posts and below the crossbar. A ball that rolls
    along the floor counts; a ball that would bounce over the bar, or in off a wall, is not seen.
    """
    crossing = goal_crossing(position, velocity, goal_y, horizon)
    if crossing is None:
        return False
    _, x, z = crossing
    return abs(x) < GOAL_MOUTH_HALF_WIDTH and z < GOAL_MOUTH_HEIGHT


def xy_in_field(x: float, y: float, margin: float = 0.0) -> bool:
    """Tell whether a ground position is at least ``margin`` away from every wall.

    The walls are taken as planes: the side walls, the back walls and the four diagonal corner
    walls. The rounded edges where they meet the floor are ignored, and so are the goals.
    """
    x, y = abs(float(x)), abs(float(y))
    return (
        x <= SIDE_WALL_X - margin
        and y <= BACK_WALL_Y - margin
        and x + y <= CORNER_SUM - margin * math.sqrt(2.0)
    )


def in_field(position: Vector, margin: float = 0.0) -> bool:
    """Tell whether a point lies inside the arena, at least ``margin`` away from every surface.

    As :func:`xy_in_field`, with the floor and the ceiling added.
    """
    height = float(position[2])
    return xy_in_field(position[0], position[1], margin) and (
        margin <= height <= CEILING_Z - margin
    )


def clamp_to_field(x: float, y: float, margin: float = 0.0) -> tuple[float, float]:
    """Move a ground position inside the arena, at least ``margin`` away from every wall."""
    x = float(np.clip(x, -(SIDE_WALL_X - margin), SIDE_WALL_X - margin))
    y = float(np.clip(y, -(BACK_WALL_Y - margin), BACK_WALL_Y - margin))
    excess = abs(x) + abs(y) - (CORNER_SUM - margin * math.sqrt(2.0))
    if excess > 0.0:
        x -= math.copysign(0.5 * excess, x)
        y -= math.copysign(0.5 * excess, y)
    return x, y
