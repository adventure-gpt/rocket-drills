"""Ball-on-roof starts: carrying the ball, and defending against a car that carries it."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from typing import Any

from rlgym.rocket_league.common_values import BACK_WALL_Y

from rocket_drills.drills.base import Drill, Range, Seed, check_range
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.geometry import BALL_ON_ROOF_HEIGHT, clamp_to_field

_DEFAULT_ROOF_HEIGHT = 150.3  # the Octane's


def _carry(
    scene: Scene, agent: Hashable, x: float, y: float, yaw: float, speed: float, boost: float
) -> None:
    """Place a car on the floor with the ball resting on its roof, both moving together."""
    rng = scene.rng
    offset = rng.uniform(-20.0, 30.0)  # the ball a little behind or ahead of the roof's centre
    hitbox = scene.state.cars[agent].hitbox_type
    height = BALL_ON_ROOF_HEIGHT.get(hitbox, _DEFAULT_ROOF_HEIGHT)
    heading = (math.cos(yaw), math.sin(yaw))
    scene.ball(
        (x + heading[0] * offset, y + heading[1] * offset, height),
        (heading[0] * speed, heading[1] * speed, 0.0),
    )
    scene.ground_car(agent, x, y, yaw, speed, boost)
    scene.keep_clear(x, y, 800.0)


class Dribble(Drill):
    """The drill car carries the ball on its roof toward the other team's net.

    The car is on the floor anywhere from deep in its own half to the edge of the other team's
    box, heading roughly at their goal, with the ball resting on its roof and moving with it. From
    here a policy can keep the carry going, flick, or lose the ball; the other team gets practice
    at challenging a carry. No other car starts within 800 uu.

    The height of the ball is the one at which it rests on that hitbox's roof, so it settles
    without a bounce. A car that only coasts loses the ball off the front within a second.

    Args:
        speed: Range of the speed of the car and the ball.
        boost: Range of the drill car's boost.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    def __init__(
        self,
        *,
        speed: Range = (200.0, 1300.0),
        boost: Range = (0.0, 100.0),
        team: int | None = None,
        opponents: OpponentLayout = "random",
        teammates: TeammateLayout = "random",
        seed: Seed = None,
    ) -> None:
        super().__init__(team=team, opponents=opponents, teammates=teammates, seed=seed)
        self.speed = check_range("speed", speed, lowest=50.0)
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any] | None:
        rng = scene.rng
        x, y = rng.uniform(-3000.0, 3000.0), rng.uniform(-4000.0, 2500.0)
        yaw = math.atan2(BACK_WALL_Y - y, -x) + rng.normal(0.0, 0.5)
        _carry(scene, scene.me, x, y, yaw, rng.uniform(*self.speed), rng.uniform(*self.boost))
        return None


class FlickDefence(Drill):
    """The drill car shadows a car that carries the ball at its net.

    The other team's first car has the ball on its roof and drives at the drill team's goal at
    700 to 1500 uu/s, aimed inside the posts. The drill car is where a defender is when that
    happens: goal-side of the carrier and on its line, ``gap`` ahead of it, backing off at about
    its speed. In half of the starts it faces the ball and reverses, in the other half it drives
    toward its own net. What follows is usually a flick over the defender, and the answer is to go
    up and meet the ball rather than to keep backing off under it.

    The carrier is in ``DrillInfo.details`` as ``"carrier"``. No other car starts within 800 uu of
    the carrier.

    Args:
        gap: Range of the distance between the carrier and the drill car.
        boost: Range of the drill car's boost.
        team: The defending team. See :class:`Drill`.
        opponents: See :class:`Drill`. Applies to the other team's cars other than the carrier.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    def __init__(
        self,
        *,
        gap: Range = (900.0, 2400.0),
        boost: Range = (10.0, 100.0),
        team: int | None = None,
        opponents: OpponentLayout = "random",
        teammates: TeammateLayout = "random",
        seed: Seed = None,
    ) -> None:
        super().__init__(team=team, opponents=opponents, teammates=teammates, seed=seed)
        self.gap = check_range("gap", gap, lowest=400.0)
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        rng = scene.rng
        carrier = self._need_opponent(scene)
        own_goal_y = -BACK_WALL_Y
        x, y = rng.uniform(-2600.0, 2600.0), rng.uniform(-3200.0, 1500.0)
        yaw = math.atan2(own_goal_y - y, rng.uniform(-600.0, 600.0) - x) + rng.normal(0.0, 0.15)
        speed = rng.uniform(700.0, 1500.0)
        _carry(scene, carrier, x, y, yaw, speed, rng.uniform(20.0, 100.0))

        heading = (math.cos(yaw), math.sin(yaw))
        gap, side = rng.uniform(*self.gap), rng.uniform(-350.0, 350.0)
        mine_x, mine_y = clamp_to_field(
            x + heading[0] * gap - heading[1] * side,
            max(y + heading[1] * gap + heading[0] * side, -5000.0),
            120.0,
        )
        scene.keep_clear(mine_x, mine_y, 800.0)
        faces_ball = rng.random() < 0.5
        to_ball = math.atan2(y - mine_y, x - mine_x)
        to_goal = math.atan2(own_goal_y - mine_y, -mine_x)
        scene.ground_car(
            scene.me,
            mine_x,
            mine_y,
            yaw,
            speed * rng.uniform(0.5, 1.1),
            rng.uniform(*self.boost),
            forward_yaw=to_ball if faces_ball else to_goal,
        )
        return {"carrier": carrier}
