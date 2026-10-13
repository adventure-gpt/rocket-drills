"""Take-off: leaving the ground for a ball that a jump with boost reaches first."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from rocket_drills.drills.base import Drill, Range, Seed, Stages, check_range
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.geometry import ball_flight, clamp_to_field, in_field

#: The lowest and the highest ball a take-off start asks the car to reach.
MIN_HEIGHT = 300.0
MAX_HEIGHT = 1000.0

_BALL_TRIES = 16


def reachable_height(time: float) -> float:
    """Height a car has climbed ``time`` seconds after a jump held for 0.2 s with boost.

    A fit to RocketSim for a car that leans back until its nose is well up and boosts all the way,
    without a second jump, capped to the range the take-off drill uses.
    """
    return float(np.clip(150.0 * time + 300.0 * time * time, 340.0, MAX_HEIGHT))


def place_takeoff(
    scene: Scene,
    *,
    attacking: bool,
    arrival_time: Range,
    height: Range,
    upper_heights: bool,
    boost: Range,
) -> tuple[np.ndarray, float]:
    """Write one take-off start: the drill car on the floor and the ball on its way to a spot.

    Args:
        scene: The start being written.
        attacking: Start between the drill team's own third and the other team's box rather than
            deep in its own half.
        arrival_time: Range of the time, in seconds, after which the ball is at the spot.
        height: Range of the spot's height.
        upper_heights: Keep to the upper 30% of what the jump reaches by then.
        boost: Range of the drill car's boost.

    Returns:
        The spot, in the scene's frame, and the time at which the ball is there.
    """
    rng = scene.rng
    speed = rng.uniform(500.0, 1700.0)
    tau = rng.uniform(*arrival_time)
    heading = rng.uniform(-0.6, 0.6)  # off straight up the pitch
    direction = np.array([math.sin(heading), math.cos(heading)])
    car_x = rng.uniform(-2800.0, 2800.0)
    car_y = rng.uniform(-800.0, 3000.0) if attacking else rng.uniform(-4300.0, -1200.0)

    reach = float(np.clip(speed * tau, 550.0, 1900.0))
    spot_x = float(np.clip(car_x + direction[0] * reach + rng.normal(0.0, 120.0), -3500, 3500))
    spot_y = float(np.clip(car_y + direction[1] * reach + rng.normal(0.0, 120.0), -4600, 4500))
    spot_x, spot_y = clamp_to_field(spot_x, spot_y, 250.0)
    top = min(height[1], reachable_height(tau))
    bottom = min(height[0], top)
    if upper_heights:
        bottom = max(bottom, 0.7 * top)
    spot = np.array([spot_x, spot_y, rng.uniform(bottom, top)])

    for _ in range(_BALL_TRIES):  # (a ball that drifts in from a wall must start inside it)
        drift = direction * rng.uniform(-100.0, 450.0) + rng.normal(0.0, 90.0, 2)
        ball_velocity = np.array([drift[0], drift[1], rng.uniform(0.0, 350.0)])
        travelled, _ = ball_flight(np.zeros(3), ball_velocity, tau)
        if in_field(spot - travelled, 150.0):
            break
    scene.ball(spot - travelled, ball_velocity)
    scene.ground_car(
        scene.me, car_x, car_y, math.atan2(direction[1], direction[0]), speed, rng.uniform(*boost)
    )
    scene.keep_clear(spot_x, spot_y, 1500.0)
    return spot, tau


class Takeoff(Drill):
    """A car on the ground, and a ball in the air ahead of it that a jump with boost reaches.

    The drill car drives up the pitch at 500 to 1700 uu/s. The ball is in the air and will be at a
    spot ahead of the car, on its line or close to it, ``arrival_time`` seconds from now, at a
    height a boosted jump has reached by then. Going up is how the car gets to the ball first. The
    spot and the time are in ``DrillInfo.details`` as ``"target"`` and ``"target_time"``.

    Stages:
        ``"attacking"`` (default 70%): the car starts between its own third and the other team's
        box, so the touch goes on at their net.

        ``"defending"`` (default 30%): the car starts deep in its own half.

    By default the other team is in its own half in both, and none of its cars is within 1500 uu
    of the spot; team-mates are on the floor behind the drill car.

    Args:
        stages: See :class:`Drill`.
        arrival_time: Range of the seconds until the ball is at the spot.
        height: Range of the spot's height, within 300 to 1000 uu. The top of the range is
            lowered to what the jump has reached by the arrival time.
        upper_heights: Keep to the upper 30% of the reachable heights (the take-off of a play
            that needs the ball high, such as a flip reset).
        boost: Range of the drill car's boost.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    STAGES: ClassVar[Mapping[str, float]] = {"attacking": 0.7, "defending": 0.3}

    def __init__(
        self,
        *,
        stages: Stages = None,
        arrival_time: Range = (0.8, 1.5),
        height: Range = (MIN_HEIGHT, MAX_HEIGHT),
        upper_heights: bool = False,
        boost: Range = (45.0, 100.0),
        team: int | None = None,
        opponents: OpponentLayout = "defending",
        teammates: TeammateLayout = "behind",
        seed: Seed = None,
    ) -> None:
        super().__init__(
            stages=stages, team=team, opponents=opponents, teammates=teammates, seed=seed
        )
        self.arrival_time = check_range("arrival_time", arrival_time, lowest=0.3)
        self.height = check_range("height", height, lowest=MIN_HEIGHT)
        if self.height[1] > MAX_HEIGHT:
            raise ValueError(f"height must stay within {MIN_HEIGHT} to {MAX_HEIGHT}, got {height}")
        self.upper_heights = upper_heights
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        spot, tau = place_takeoff(
            scene,
            attacking=stage == "attacking",
            arrival_time=self.arrival_time,
            height=self.height,
            upper_heights=self.upper_heights,
            boost=self.boost,
        )
        return {"target": scene.to_world(spot), "target_time": tau}
