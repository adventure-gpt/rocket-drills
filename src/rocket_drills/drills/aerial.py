"""Air-play starts: a high ball above a grounded car, the ceiling, and the backboard."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z

from rocket_drills.drills.base import Drill, Range, Seed, Stages, check_range
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.geometry import BALL_WALL_OFFSET, UP, clamp_to_field, launch_velocity, unit

_MAX_FOLLOW_SPEED = 2200.0  # a car cannot go faster than 2300 uu/s


class Aerial(Drill):
    """A slow, high ball and a car on the ground near it, facing it, with boost.

    The ball is anywhere over the pitch, 450 to 1500 uu up, drifting at a few hundred uu/s. The
    drill car stands or rolls on the floor 400 to 1600 uu from the point below the ball, nose
    roughly at it. Nothing says which way the play should go: this is the plain aerial, for
    learning to get up to a ball at all. :class:`~rocket_drills.drills.Takeoff` is the timed
    version, with the ball on its way to a spot on the car's path.

    Args:
        height: Range of the ball's height.
        distance: Range of the distance along the floor between the car and the ball.
        boost: Range of the drill car's boost.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    def __init__(
        self,
        *,
        height: Range = (450.0, 1500.0),
        distance: Range = (400.0, 1600.0),
        boost: Range = (50.0, 100.0),
        team: int | None = None,
        opponents: OpponentLayout = "random",
        teammates: TeammateLayout = "random",
        seed: Seed = None,
    ) -> None:
        super().__init__(team=team, opponents=opponents, teammates=teammates, seed=seed)
        self.height = check_range("height", height, lowest=200.0)
        if self.height[1] > CEILING_Z - 200.0:
            raise ValueError(f"height must stay below {CEILING_Z - 200.0}, got {height}")
        self.distance = check_range("distance", distance, lowest=200.0)
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any] | None:
        rng = scene.rng
        ball = np.array(
            [rng.uniform(-3000.0, 3000.0), rng.uniform(-4000.0, 4000.0), rng.uniform(*self.height)]
        )
        drift = rng.normal(0.0, 250.0, 2)
        scene.ball(ball, (drift[0], drift[1], rng.uniform(-150.0, 150.0)))
        distance, bearing = rng.uniform(*self.distance), rng.uniform(-math.pi, math.pi)
        x, y = clamp_to_field(
            ball[0] + math.cos(bearing) * distance, ball[1] + math.sin(bearing) * distance, 250.0
        )
        yaw = math.atan2(ball[1] - y, ball[0] - x) + rng.normal(0.0, 0.3)
        scene.ground_car(scene.me, x, y, yaw, rng.uniform(0.0, 1000.0), rng.uniform(*self.boost))
        return None


class Ceiling(Drill):
    """The drill car drives on the ceiling toward the other team's half, the ball below and ahead.

    A car cannot stay on the ceiling: it drops off within a quarter of a second, whatever it does.
    Because it leaves without jumping it holds its dodge all the way down, and that is what a
    ceiling shot is: drop, line the ball up, and use the dodge on it. The ball starts 400 to 900 uu
    ahead of the car and 400 to 900 uu below it, drifting up the pitch.

    Args:
        boost: Range of the drill car's boost.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    def __init__(
        self,
        *,
        boost: Range = (30.0, 100.0),
        team: int | None = None,
        opponents: OpponentLayout = "defending",
        teammates: TeammateLayout = "behind",
        seed: Seed = None,
    ) -> None:
        super().__init__(team=team, opponents=opponents, teammates=teammates, seed=seed)
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any] | None:
        rng = scene.rng
        me = scene.me
        car = np.array(
            [
                rng.uniform(-3000.0, 3000.0),
                rng.uniform(-2000.0, 2500.0),
                CEILING_Z - scene.rest_height(me),
            ]
        )
        forward = unit((rng.normal(0.0, 0.25), 1.0, 0.0))
        speed, boost = rng.uniform(400.0, 1200.0), rng.uniform(*self.boost)
        ball = car + np.array(
            [rng.normal(0.0, 200.0), rng.uniform(400.0, 900.0), -rng.uniform(400.0, 900.0)]
        )
        ball[2] = max(ball[2], 300.0)
        scene.ball(
            ball, (rng.normal(0.0, 200.0), rng.uniform(0.0, 600.0), rng.uniform(-200.0, 100.0))
        )
        scene.surface_car(me, car, forward * speed, forward, -UP, boost)
        return None


class Backboard(Drill):
    """The ball flies at the other team's backboard and the drill car follows it in.

    The ball starts 1100 to 2100 uu from their back wall, 700 to 1300 uu up, and is on its way to
    a point on that wall above or beside the goal, which it reaches at 1200 to 2000 uu/s along the
    floor. Reading the rebound and hitting it before it comes down is a backboard read; when the
    drill car's own touch put the ball there it is a double tap. The point on the wall and the time
    until the ball gets there are in ``DrillInfo.details`` as ``"target"`` and ``"target_time"``.

    Stages:
        ``"airborne"`` (default 50%): the car is in the air 500 to 1100 uu behind the ball and a
        little below it, moving with it, with a jump 0.3 to 0.9 s behind it: its dodge runs out
        before or soon after the ball comes off the wall.

        ``"grounded"`` (default 50%): the car is on the floor 600 to 1200 uu behind the ball,
        driving at it.

    Args:
        stages: See :class:`Drill`.
        impact_height: Range of the height at which the ball meets the wall. The crossbar is at
            643 uu, so the default keeps the whole ball above the goal.
        boost: Range of the drill car's boost.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    STAGES: ClassVar[Mapping[str, float]] = {"airborne": 0.5, "grounded": 0.5}

    def __init__(
        self,
        *,
        stages: Stages = None,
        impact_height: Range = (750.0, 1500.0),
        boost: Range = (40.0, 100.0),
        team: int | None = None,
        opponents: OpponentLayout = "defending",
        teammates: TeammateLayout = "behind",
        seed: Seed = None,
    ) -> None:
        super().__init__(
            stages=stages, team=team, opponents=opponents, teammates=teammates, seed=seed
        )
        self.impact_height = check_range("impact_height", impact_height, lowest=200.0)
        if self.impact_height[1] > CEILING_Z - 300.0:
            raise ValueError(f"impact_height must stay below {CEILING_Z - 300.0}")
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        rng = scene.rng
        me = scene.me
        ball = np.array(
            [rng.uniform(-900.0, 900.0), rng.uniform(3000.0, 4000.0), rng.uniform(700.0, 1300.0)]
        )
        impact = np.array(
            [
                rng.uniform(-1000.0, 1000.0),
                BACK_WALL_Y - BALL_WALL_OFFSET,
                rng.uniform(*self.impact_height),
            ]
        )
        time = float(np.linalg.norm(impact[:2] - ball[:2])) / rng.uniform(1200.0, 2000.0)
        ball_velocity = launch_velocity(ball, impact, time)
        scene.ball(ball, ball_velocity)

        boost = rng.uniform(*self.boost)
        if stage == "airborne":
            car = ball - np.array(
                [rng.normal(0.0, 150.0), rng.uniform(500.0, 1100.0), rng.uniform(0.0, 300.0)]
            )
            car[2] = max(car[2], 300.0)
            toward = unit(ball - car)
            velocity = ball_velocity * rng.uniform(0.7, 0.9)
            velocity *= min(1.0, _MAX_FOLLOW_SPEED / float(np.linalg.norm(velocity)))
            scene.air_car(me, car, velocity, toward, UP, boost, air_time=rng.uniform(0.3, 0.9))
        else:
            x, y = ball[0] + rng.normal(0.0, 200.0), ball[1] - rng.uniform(600.0, 1200.0)
            yaw = math.atan2(ball[1] - y, ball[0] - x)
            scene.ground_car(me, x, y, yaw, rng.uniform(800.0, 1500.0), boost)
        return {"target": scene.to_world(impact), "target_time": time}
