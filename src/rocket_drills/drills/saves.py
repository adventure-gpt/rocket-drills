"""Save practice: a shot that is already on target, and a defender who can still get to it."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, BALL_RESTING_HEIGHT

from rocket_drills.drills.base import Drill, Range, Seed, Stages, check_range, check_share
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.geometry import clamp_to_field, launch_velocity

_OWN_GOAL_Y = -BACK_WALL_Y  # in the drill's frame the drill team defends the goal at -y
_MAX_FLIGHT = 2.2  # seconds: every shot is on its way in within the 2.5 s on-target horizon
_BALL_GAP = 350.0  # closest the defender starts to the ball, along the floor
_CAR_GAP = 400.0  # closest the shooter starts to the defender
_TRIES = 16


def _yaw_to(x: float, y: float, target_x: float, target_y: float) -> float:
    return math.atan2(target_y - y, target_x - x)


class _ShotDrill(Drill):
    """What the two save drills share: the shot, the shooter and the defender's three starts."""

    _FAST: ClassVar[bool] = False

    def __init__(
        self,
        *,
        stages: Stages,
        speed: Range,
        shooter: bool,
        team: int | None,
        opponents: OpponentLayout,
        teammates: TeammateLayout,
        seed: Seed,
    ) -> None:
        super().__init__(
            stages=stages, team=team, opponents=opponents, teammates=teammates, seed=seed
        )
        self.speed = check_range("speed", speed, lowest=300.0)
        self.shooter = shooter

    def _shoot(
        self, scene: Scene, start: np.ndarray, target: np.ndarray, *, lifted: bool
    ) -> tuple[np.ndarray, float]:
        """Send the ball from ``start`` at ``target`` on the goal line.

        Returns:
            The direction of the shot along the floor, and its flight time.
        """
        rng = scene.rng
        along = target[:2] - start[:2]
        distance = float(np.linalg.norm(along))
        along = along / distance
        time = min(distance / rng.uniform(*self.speed), _MAX_FLIGHT)
        if lifted:
            velocity = launch_velocity(start, target, time)
        else:
            velocity = np.array([along[0], along[1], 0.0]) * (distance / time)
        scene.ball(start, velocity, rng.normal(0.0, 1.0, 3))
        return along, time

    def _defender(
        self,
        scene: Scene,
        stage: str,
        start: np.ndarray,
        target: np.ndarray,
        along: np.ndarray,
        boost: float,
    ) -> None:
        """Put the drill car in goal, between the ball and the net, or on its way back."""
        for _ in range(_TRIES):  # (not on top of the ball)
            x, y, yaw, speed, facing = self._defender_spot(scene.rng, stage, start, target, along)
            if math.hypot(x - start[0], y - start[1]) >= _BALL_GAP:
                break
        scene.ground_car(scene.me, x, y, yaw, speed, boost, forward_yaw=facing)

    def _defender_spot(
        self,
        rng: np.random.Generator,
        stage: str,
        start: np.ndarray,
        target: np.ndarray,
        along: np.ndarray,
    ) -> tuple[float, float, float, float, float]:
        """Draw the defender's position, direction of travel, speed and facing for a stage."""
        fast = self._FAST
        if stage == "in_goal":
            x = rng.uniform(-800.0, 800.0) if fast else rng.uniform(-700.0, 700.0)
            y = rng.uniform(-5050.0, -4300.0) if fast else rng.uniform(-5050.0, -4700.0)
            speed = rng.uniform(0.0, 800.0) if fast else rng.uniform(0.0, 300.0)
            yaw = _yaw_to(x, y, start[0], start[1]) + (rng.normal(0.0, 0.5) if fast else 0.0)
            return x, y, yaw, speed, yaw
        if stage == "shadowing":
            part = rng.uniform(0.3, 0.8) if fast else rng.uniform(0.35, 0.75)
            side = rng.uniform(-350.0, 350.0) if fast else rng.uniform(-300.0, 300.0)
            x = start[0] + (target[0] - start[0]) * part - side * along[1]
            y = start[1] + (_OWN_GOAL_Y - start[1]) * part + side * along[0]
            x, y = clamp_to_field(x, y, 120.0)
            faces_ball = rng.random() < 0.5
            to_ball, to_goal = _yaw_to(x, y, start[0], start[1]), _yaw_to(x, y, 0.0, _OWN_GOAL_Y)
            speed = rng.uniform(300.0, 1500.0) if fast else rng.uniform(300.0, 1400.0)
            # It moves back with the shot, whichever way it faces.
            return x, y, math.atan2(along[1], along[0]), speed, to_ball if faces_ball else to_goal
        # rotating back from the side
        wide = rng.uniform(1200.0, 3000.0) if fast else rng.uniform(1500.0, 3000.0)
        x = rng.choice([-1.0, 1.0]) * wide
        y = rng.uniform(-4900.0, -3200.0) if fast else rng.uniform(-4800.0, -3000.0)
        x, y = clamp_to_field(x, y, 200.0)
        speed = rng.uniform(800.0, 2000.0) if fast else rng.uniform(600.0, 1800.0)
        yaw = _yaw_to(x, y, 0.0, _OWN_GOAL_Y)
        return x, y, yaw, speed, yaw

    def _shooter(
        self, scene: Scene, start: np.ndarray, along: np.ndarray, follow_speed: Range
    ) -> None:
        """Put the other team's first car behind the ball, following the shot in."""
        rng = scene.rng
        back, speed = rng.uniform(400.0, 1400.0), rng.uniform(*follow_speed)
        boost = rng.uniform(0.0, 100.0)
        if not (self.shooter and scene.opponents):
            return
        defender = scene.position(scene.me)
        for further in (0.0, 400.0, 800.0, 1200.0):  # (not on top of a defender coming back)
            x, y = clamp_to_field(
                start[0] - along[0] * (back + further),
                start[1] - along[1] * (back + further),
                200.0,
            )
            if math.hypot(x - defender[0], y - defender[1]) >= _CAR_GAP:
                yaw = math.atan2(along[1], along[0])
                scene.ground_car(scene.opponents[0], x, y, yaw, speed, boost)
                return
        # No room behind the ball: the car is placed with the others, by the layout.


class Save(_ShotDrill):
    """A shot on target at the drill team's net, with its first car placed to make the save.

    The ball is in the drill team's half, rolling or in the air, and already on its way inside the
    posts at 1400 to 3200 uu/s. The other team's first car follows the shot in. The point where
    the ball crosses the goal line and the time it takes are in ``DrillInfo.details`` as
    ``"target"`` and ``"target_time"`` (for the rolling shots the time ignores the ball slowing
    down on the floor).

    Stages (where the drill car, the defender, starts):
        ``"in_goal"`` (default 45%): on or just off its goal line, facing the ball.

        ``"shadowing"`` (default 35%): part-way up the pitch between the ball and the net,
        moving back with the shot, facing either the ball or its own goal.

        ``"rotating"`` (default 20%): coming back from the side, heading for its goal.

    Args:
        stages: See :class:`Drill`.
        speed: Range of the shot's speed along the floor. A shot that would take more than 2.2 s
            to arrive is sped up until it does not.
        airborne_share: Share of the shots that are in the air rather than rolling.
        boost: Range of the defender's boost.
        shooter: Put the other team's first car behind the ball, following it in.
        team: The defending team. See :class:`Drill`.
        opponents: See :class:`Drill`. Applies to the other team's cars other than the shooter.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    STAGES: ClassVar[Mapping[str, float]] = {"in_goal": 0.45, "shadowing": 0.35, "rotating": 0.2}

    def __init__(
        self,
        *,
        stages: Stages = None,
        speed: Range = (1400.0, 3200.0),
        airborne_share: float = 0.4,
        boost: Range = (20.0, 100.0),
        shooter: bool = True,
        team: int | None = None,
        opponents: OpponentLayout = "random",
        teammates: TeammateLayout = "random",
        seed: Seed = None,
    ) -> None:
        super().__init__(
            stages=stages,
            speed=speed,
            shooter=shooter,
            team=team,
            opponents=opponents,
            teammates=teammates,
            seed=seed,
        )
        self.airborne_share = check_share("airborne_share", airborne_share)
        self.boost = check_range("boost", boost, lowest=0.0)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        rng = scene.rng
        start = np.array([rng.uniform(-3000.0, 3000.0), rng.uniform(-3800.0, -800.0), 0.0])
        airborne = rng.random() < self.airborne_share
        start[2] = rng.uniform(200.0, 900.0) if airborne else BALL_RESTING_HEIGHT
        target_height = rng.uniform(120.0, 480.0) if airborne else BALL_RESTING_HEIGHT
        target = np.array([rng.uniform(-700.0, 700.0), _OWN_GOAL_Y, target_height])
        along, time = self._shoot(scene, start, target, lifted=airborne)
        self._defender(scene, stage, start, target, along, rng.uniform(*self.boost))
        self._shooter(scene, start, along, follow_speed=(800.0, 1800.0))
        return {"target": scene.to_world(target), "target_time": time}


class FastSave(_ShotDrill):
    """A fast shot on target, most of them off the ground, at a defender who is often low on boost.

    As :class:`Save`, but the shot travels at 2500 to 3800 uu/s from 900 to 4100 uu out, which
    leaves 0.3 to 1.6 s. 40% of the shots are struck off the ground and rising, 35% are already in
    the air and 25% run along the floor. In ``low_boost_share`` of the starts the defender has 0
    to 15 boost; the rest are split evenly between 15 to 40 and 40 to 100.

    Stages (where the drill car, the defender, starts):
        ``"in_goal"`` (default 50%): goal-side near its net, on the line or up to 800 uu off it,
        roughly facing the ball.

        ``"shadowing"`` (default 35%): between the ball and the net, moving back with the shot.

        ``"rotating"`` (default 15%): coming back from the side, heading for its goal.

    Args:
        stages: See :class:`Drill`.
        speed: Range of the shot's speed along the floor.
        low_boost_share: Share of the starts in which the defender has 0 to 15 boost.
        shooter: Put the other team's first car behind the ball, following it in.
        team: The defending team. See :class:`Drill`.
        opponents: See :class:`Drill`. Applies to the other team's cars other than the shooter.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    STAGES: ClassVar[Mapping[str, float]] = {"in_goal": 0.5, "shadowing": 0.35, "rotating": 0.15}
    _FAST: ClassVar[bool] = True

    def __init__(
        self,
        *,
        stages: Stages = None,
        speed: Range = (2500.0, 3800.0),
        low_boost_share: float = 0.5,
        shooter: bool = True,
        team: int | None = None,
        opponents: OpponentLayout = "random",
        teammates: TeammateLayout = "random",
        seed: Seed = None,
    ) -> None:
        super().__init__(
            stages=stages,
            speed=speed,
            shooter=shooter,
            team=team,
            opponents=opponents,
            teammates=teammates,
            seed=seed,
        )
        self.low_boost_share = check_share("low_boost_share", low_boost_share)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        rng = scene.rng
        start = np.array([rng.uniform(-3000.0, 3000.0), rng.uniform(-4200.0, -1000.0), 0.0])
        kind = rng.random()
        if kind < 0.40:  # struck off the ground, rising
            start[2], target_height = BALL_RESTING_HEIGHT, rng.uniform(180.0, 500.0)
        elif kind < 0.75:  # already in the air
            start[2], target_height = rng.uniform(150.0, 700.0), rng.uniform(130.0, 500.0)
        else:  # along the floor
            start[2], target_height = BALL_RESTING_HEIGHT, BALL_RESTING_HEIGHT
        target = np.array([rng.uniform(-750.0, 750.0), _OWN_GOAL_Y, target_height])
        along, time = self._shoot(scene, start, target, lifted=kind < 0.75)
        tank = rng.random()
        low, medium = rng.uniform(0.0, 15.0), rng.uniform(15.0, 40.0)
        high = rng.uniform(40.0, 100.0)
        halfway = 0.5 * (1.0 + self.low_boost_share)
        boost = low if tank < self.low_boost_share else (medium if tank < halfway else high)
        self._defender(scene, stage, start, target, along, boost)
        self._shooter(scene, start, along, follow_speed=(1000.0, 2200.0))
        return {"target": scene.to_world(target), "target_time": time}
