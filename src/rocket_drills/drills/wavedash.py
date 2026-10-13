"""Landing practice: a race to the ball that only a dodge into the landing can win."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any

import numpy as np
from rlgym.rocket_league.common_values import BALL_RESTING_HEIGHT

from rocket_drills.drills.base import Drill, Range, Seed, check_range, check_share
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.geometry import UP, clamp_to_field, unit


class WavedashLanding(Drill):
    """Each team's first car comes down from a hop with an empty tank, racing to a ball ahead.

    The ball lies slow in the middle of the pitch. Each team's first car is on its own side of it,
    1500 to 4000 uu away (the two within about 10% of each other), heading at it at 300 to 1400
    uu/s, 60 to 350 uu off the floor and coming down, level but for a tilt of up to 0.45 rad, nose
    up in half of the starts. Both have 0 to 20 boost, so a dodge into the landing (a wavedash,
    worth about 500 uu/s at once) is the only speed to be had.

    In ``jumped_share`` of the cars the hop was a jump that ended 0.15 to 0.5 s ago, so the dodge
    is there but not for long; the others have not jumped and hold it until they land.

    Without an opponent the drill car makes the same start alone.

    Args:
        distance: Range of the distance from each car to the ball.
        height: Range of the height of the cars' wheels above the floor.
        jumped_share: Share of the two cars that have a jump behind them.
        team: See :class:`Drill`. The start is symmetric, so this only decides which car
            ``DrillInfo.agent`` names.
        opponents: See :class:`Drill`. Applies to the other team's cars other than its first.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    def __init__(
        self,
        *,
        distance: Range = (1500.0, 4000.0),
        height: Range = (60.0, 350.0),
        jumped_share: float = 0.5,
        team: int | None = None,
        opponents: OpponentLayout = "random",
        teammates: TeammateLayout = "random",
        seed: Seed = None,
    ) -> None:
        super().__init__(team=team, opponents=opponents, teammates=teammates, seed=seed)
        self.distance = check_range("distance", distance, lowest=600.0)
        self.height = check_range("height", height, lowest=20.0)
        self.jumped_share = check_share("jumped_share", jumped_share)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any] | None:
        rng = scene.rng
        ball_x, ball_y = rng.uniform(-2500.0, 2500.0), rng.uniform(-2200.0, 2200.0)
        roll = rng.normal(0.0, 150.0, 2)
        scene.ball((ball_x, ball_y, BALL_RESTING_HEIGHT), (roll[0], roll[1], 0.0))
        scene.keep_clear(ball_x, ball_y, 1200.0)
        reach = rng.uniform(*self.distance)
        racers: list[tuple[Hashable | None, float]] = [
            (scene.me, -1.0),
            (scene.opponents[0] if scene.opponents else None, 1.0),
        ]
        for agent, side in racers:  # the drill team comes from its own half, the other from its
            distance = reach * rng.uniform(0.95, 1.05)
            bearing = rng.uniform(-1.0, 1.0)  # off the straight line down the pitch
            x, y = clamp_to_field(
                ball_x + math.sin(bearing) * distance,
                ball_y + side * math.cos(bearing) * distance,
                300.0,
            )
            yaw = math.atan2(ball_y - y, ball_x - x) + rng.normal(0.0, 0.2)
            speed = rng.uniform(300.0, 1400.0)
            nose_up = rng.random() < 0.5
            pitch = rng.uniform(0.1, 0.45) if nose_up else rng.uniform(-0.2, 0.1)
            lean = rng.uniform(-0.3, 0.3)
            altitude = rng.uniform(*self.height)
            sink = rng.uniform(100.0, 600.0)
            boost = rng.uniform(0.0, 20.0)
            jumped = rng.random() < self.jumped_share
            air_time = rng.uniform(0.15, 0.5)
            if agent is None:
                continue
            forward = np.array(
                [
                    math.cos(pitch) * math.cos(yaw),
                    math.cos(pitch) * math.sin(yaw),
                    math.sin(pitch),
                ]
            )
            level_roof = unit(UP - forward * forward[2])
            roof = level_roof * math.cos(lean) + np.cross(level_roof, forward) * math.sin(lean)
            scene.air_car(
                agent,
                (x, y, scene.rest_height(agent) + altitude),
                (math.cos(yaw) * speed, math.sin(yaw) * speed, -sink),
                forward,
                roof,
                boost,
                air_time=air_time if jumped else None,
            )
            scene.keep_clear(x, y, 600.0)
        return None
