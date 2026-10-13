"""The wall-to-air dribble, one stage per start."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BALL_RESTING_HEIGHT,
    GRAVITY,
    RAMP_HEIGHT,
    SIDE_WALL_X,
)

from rocket_drills.drills.base import (
    Drill,
    Seed,
    Stages,
    check_share,
    normalise_weights,
)
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.geometry import (
    BALL_WALL_OFFSET,
    UP,
    clamp_to_field,
    in_field,
    unit,
    xy_in_field,
)

_GOAL_CLEARANCE = 1150.0  # low on a back wall, stay this far from its middle: beside the goal
_CHALLENGER_GAP = 400.0  # closest a challenger starts to the drill car and to the ball
_TRIES = 200


@dataclass(frozen=True)
class _Wall:
    """A wall in the drill's frame: a point on it at floor level, its inward normal, its tangent."""

    kind: str
    origin: tuple[float, float]
    normal: tuple[float, float]
    tangent: tuple[float, float]
    reach: float  # usable distance along the wall, either side of the origin
    fillet: float  # radius of the curve between the wall and the floor

    def point(self, depth: float, along: float, height: float) -> np.ndarray:
        """Return the point ``depth`` off the wall, ``along`` it, ``height`` up."""
        return np.array(
            [
                self.origin[0] + depth * self.normal[0] + along * self.tangent[0],
                self.origin[1] + depth * self.normal[1] + along * self.tangent[1],
                height,
            ]
        )

    def rolling_ball(self, height: float) -> tuple[float, float, float]:
        """Describe a ball that rolls up the wall, with its centre ``height`` above the floor.

        Returns:
            Its distance from the wall, and the parts of its unit direction of travel that point
            off the wall and upward. Above the curve at the foot of the wall that direction is
            straight up; on the curve it leans in toward the wall.
        """
        drop = self.fillet - height
        hub = self.fillet - BALL_WALL_OFFSET  # from the centre of the curve to the ball's centre
        if drop <= 0.0 or hub <= 0.0:
            return BALL_WALL_OFFSET, 0.0, 1.0
        drop = min(drop, hub)
        inward = math.sqrt(hub * hub - drop * drop)
        return self.fillet - inward, -drop / hub, inward / hub

    def vector(self, off: float, along: float, up: float) -> np.ndarray:
        """Return the vector with those components off the wall, along it and upward."""
        return np.array(
            [
                off * self.normal[0] + along * self.tangent[0],
                off * self.normal[1] + along * self.tangent[1],
                up,
            ]
        )


# The curve at the foot of a side wall has the radius RLGym calls RAMP_HEIGHT; the one at the foot
# of a back wall is tighter (about 150 uu, measured with a rolling ball in RocketSim 2.2.1).
_BACK_FILLET = 150.0
_WALLS: dict[str, tuple[_Wall, ...]] = {
    "side": (
        _Wall("side", (SIDE_WALL_X, 0.0), (-1.0, 0.0), (0.0, 1.0), 3300.0, RAMP_HEIGHT),
        _Wall("side", (-SIDE_WALL_X, 0.0), (1.0, 0.0), (0.0, 1.0), 3300.0, RAMP_HEIGHT),
    ),
    "their_back": (
        _Wall("their_back", (0.0, BACK_WALL_Y), (0.0, -1.0), (1.0, 0.0), 2450.0, _BACK_FILLET),
    ),
    "own_back": (
        _Wall("own_back", (0.0, -BACK_WALL_Y), (0.0, 1.0), (1.0, 0.0), 2450.0, _BACK_FILLET),
    ),
}


class WallToAirChain(Drill):
    """The wall-to-air dribble, taught backwards: each stage starts where the next one ends.

    The full play: take the ball to a wall, carry it up the wall, leave the wall with it near the
    top, and carry it through the air to the other team's net. The drill uses either side wall
    anywhere along it, the other team's back wall and the drill team's own back wall (beside and
    above the goals).

    Stages:
        ``"approach"`` (default 10%): on the floor, the ball rolling at a wall (70%) or just
        starting up it (30%), the car behind the ball and heading for it.

        ``"setup"`` (default 20%): the ball rolling up the wall 450 to 1000 uu high, the car on
        the wall below it, wheels on the wall, driving up at it.

        ``"launch"`` (default 30%): the ball 1450 to 1800 uu high and coming off the wall, the
        car high on the wall below it: jump off and meet it in the air.

        ``"finish"`` (default 40%): the ball high and drifting away from the wall, the car in
        the air 260 to 500 uu from it on any side, nose at it, moving with it.

    Args:
        stages: See :class:`Drill`.
        walls: Weights of the walls: any of ``"side"``, ``"their_back"`` and ``"own_back"``.
        challenger_share: Share of the ``"launch"`` and ``"finish"`` starts in which a defender
            is already in the air on a course to meet the ball. With more than one opponent it is
            the last of them, so that the first can stay in net.
        low_boost_share: Share of the starts in which the drill car has 0 to 20 boost instead of
            enough for the play, so that a policy also meets the case where going up is wrong.
        jumped_share: Share of the ``"finish"`` starts in which the car left the wall with a jump
            0.2 to 1.0 s ago and so has a dodge only for the rest of the 1.25 s window. In the
            others it rolled off the wall without jumping and holds its dodge until it lands.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    STAGES: ClassVar[Mapping[str, float]] = {
        "approach": 0.1,
        "setup": 0.2,
        "launch": 0.3,
        "finish": 0.4,
    }
    WALLS: ClassVar[Mapping[str, float]] = {"side": 0.55, "their_back": 0.25, "own_back": 0.2}

    def __init__(
        self,
        *,
        stages: Stages = None,
        walls: Stages = None,
        challenger_share: float = 0.35,
        low_boost_share: float = 0.25,
        jumped_share: float = 0.5,
        team: int | None = None,
        opponents: OpponentLayout = "defending",
        teammates: TeammateLayout = "behind",
        seed: Seed = None,
    ) -> None:
        super().__init__(
            stages=stages, team=team, opponents=opponents, teammates=teammates, seed=seed
        )
        self.walls: dict[str, float] = normalise_weights(walls, self.WALLS, what="wall")
        self.challenger_share = check_share("challenger_share", challenger_share)
        self.low_boost_share = check_share("low_boost_share", low_boost_share)
        self.jumped_share = check_share("jumped_share", jumped_share)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        rng = scene.rng
        build = {
            "approach": self._approach,
            "setup": self._on_wall,
            "launch": self._on_wall,
            "finish": self._finish,
        }[stage]
        for _ in range(_TRIES):
            wall = self._choose_wall(rng)
            if build(scene, wall, stage):
                break
        else:  # pragma: no cover - the samplers accept well over half of their draws
            raise RuntimeError(f"{self.name} could not place the {stage} stage")

        challenged = rng.random() < self.challenger_share
        if stage in ("launch", "finish") and challenged and scene.opponents:
            self._challenger(scene, scene.opponents[-1])
        return {"wall": wall.kind}

    def _choose_wall(self, rng: np.random.Generator) -> _Wall:
        kinds = list(self.walls)
        kind = kinds[int(rng.choice(len(kinds), p=[self.walls[k] for k in kinds]))]
        options = _WALLS[kind]
        return options[int(rng.integers(len(options)))]

    def _boost(self, rng: np.random.Generator, enough: float) -> float:
        low = rng.random() < self.low_boost_share
        return float(rng.uniform(0.0, 20.0) if low else rng.uniform(enough, 100.0))

    @staticmethod
    def _along_speed(
        rng: np.random.Generator, wall: _Wall, along: float, low: float, high: float
    ) -> float:
        """Ball speed along the wall: up the pitch on a side wall, toward the goal on theirs."""
        if wall.kind == "side":
            return rng.uniform(low, high)
        if wall.kind == "their_back":
            return -np.sign(along) * rng.uniform(0.0, 0.7 * high)
        return rng.uniform(-0.6 * high, 0.6 * high)

    @staticmethod
    def _along(rng: np.random.Generator, wall: _Wall, low_on_wall: bool) -> float:
        if wall.kind != "side" and low_on_wall:
            return rng.choice([-1.0, 1.0]) * rng.uniform(_GOAL_CLEARANCE, wall.reach)
        return rng.uniform(-wall.reach, wall.reach)

    def _approach(self, scene: Scene, wall: _Wall, stage: str) -> bool:
        rng = scene.rng
        ball_along = self._along(rng, wall, low_on_wall=True)
        if rng.random() < 0.7:  # the ball rolling at the wall
            ball_depth = rng.uniform(250.0, 1300.0)
            off = -rng.uniform(400.0, 1300.0)
            along = self._along_speed(rng, wall, ball_along, -200.0, 700.0)
            ball = wall.point(ball_depth, ball_along, BALL_RESTING_HEIGHT)
            ball_velocity = wall.vector(off, along, 0.0)
            car_depth = ball_depth + rng.uniform(350.0, 1100.0)
            lag = (car_depth - ball_depth) * rng.uniform(0.5, 1.0)
            car_along = ball_along - along / abs(off) * lag + rng.normal(0.0, 150.0)
        else:  # the ball just starting up the wall
            ball_height = rng.uniform(150.0, 350.0)
            ball_depth, lean, rise = wall.rolling_ball(ball_height)
            ball = wall.point(ball_depth, ball_along, ball_height)
            along = self._along_speed(rng, wall, ball_along, -200.0, 600.0)
            climb = rng.uniform(500.0, 1100.0)
            ball_velocity = wall.vector(lean * climb, along, rise * climb)
            car_depth = rng.uniform(250.0, 800.0)
            car_along = ball_along + rng.normal(0.0, 200.0)
        car = wall.point(car_depth, car_along, 0.0)
        heading = ball + 0.4 * ball_velocity - car
        speed, boost = rng.uniform(700.0, 1600.0), self._boost(rng, 40.0)
        if not (xy_in_field(car[0], car[1], 150.0) and xy_in_field(ball[0], ball[1], 90.0)):
            return False
        scene.ball(ball, ball_velocity)
        scene.ground_car(scene.me, car[0], car[1], np.arctan2(heading[1], heading[0]), speed, boost)
        return True

    def _on_wall(self, scene: Scene, wall: _Wall, stage: str) -> bool:
        rng = scene.rng
        rest = scene.rest_height(scene.me)
        if stage == "setup":
            ball_along, ball_height = self._along(rng, wall, True), rng.uniform(450.0, 1000.0)
            along = self._along_speed(rng, wall, ball_along, -200.0, 600.0)
            ball = wall.point(BALL_WALL_OFFSET, ball_along, ball_height)
            ball_velocity = wall.vector(0.0, along, rng.uniform(600.0, 1400.0))
            car_along = ball_along - np.sign(along) * rng.uniform(0.0, 350.0)
            car_height = max(280.0, ball_height - rng.uniform(260.0, 450.0))
            speed, boost = rng.uniform(900.0, 1500.0), self._boost(rng, 60.0)
        else:
            ball_along, ball_height = self._along(rng, wall, False), rng.uniform(1450.0, 1800.0)
            along = self._along_speed(rng, wall, ball_along, 0.0, 800.0)
            ball = wall.point(BALL_WALL_OFFSET + rng.uniform(180.0, 450.0), ball_along, ball_height)
            ball_velocity = wall.vector(
                rng.uniform(150.0, 650.0), along, rng.uniform(-150.0, 350.0)
            )
            car_along = ball_along - np.sign(along) * rng.uniform(-150.0, 300.0)
            car_height = rng.uniform(1000.0, 1450.0)
            speed, boost = rng.uniform(600.0, 1200.0), self._boost(rng, 50.0)
        in_goal_mouth = (
            wall.kind != "side" and car_height < 800.0 and abs(car_along) < _GOAL_CLEARANCE - 100.0
        )
        if in_goal_mouth or abs(car_along) > wall.reach + 200.0:
            return False
        car = wall.point(rest, car_along, car_height)
        forward = unit(wall.vector(0.0, ball_along - car_along, ball_height - car_height))
        scene.ball(ball, ball_velocity)
        scene.surface_car(
            scene.me, car, forward * speed, forward, wall.vector(1.0, 0.0, 0.0), boost
        )
        return True

    def _finish(self, scene: Scene, wall: _Wall, stage: str) -> bool:
        rng = scene.rng
        if wall.kind == "side":
            depth, along = rng.uniform(800.0, 2600.0), rng.uniform(-2500.0, 3000.0)
            height = rng.uniform(1150.0, 1750.0)
            off, drift = rng.uniform(100.0, 700.0), rng.uniform(300.0, 1200.0)
        elif wall.kind == "their_back":
            depth, along = rng.uniform(300.0, 1500.0), rng.uniform(-2200.0, 2200.0)
            height = rng.uniform(900.0, 1700.0)
            off, drift = rng.uniform(100.0, 600.0), -np.sign(along) * rng.uniform(0.0, 500.0)
        else:
            depth, along = rng.uniform(500.0, 2200.0), rng.uniform(-2400.0, 2400.0)
            height = rng.uniform(1100.0, 1750.0)
            off, drift = rng.uniform(300.0, 1100.0), rng.uniform(-300.0, 300.0)
        ball = wall.point(depth, along, height)
        ball_velocity = wall.vector(off, drift, rng.uniform(-400.0, 100.0))
        offset = unit(rng.normal(size=3)) * rng.uniform(260.0, 500.0)  # the car, any side of it
        car = ball + offset
        car[2] = float(np.clip(car[2], 300.0, 1850.0))
        velocity = ball_velocity + rng.normal(0.0, 150.0, 3)
        boost = self._boost(rng, 40.0)
        jumped = rng.random() < self.jumped_share
        air_time = rng.uniform(0.2, 1.0)
        gap = float(np.linalg.norm(car - ball))
        if not (in_field(car, 150.0) and in_field(ball, 200.0) and gap >= 260.0):
            return False
        toward = (ball - car) / gap
        roof = UP if abs(toward[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        scene.ball(ball, ball_velocity)
        scene.air_car(
            scene.me, car, velocity, toward, roof, boost, air_time=air_time if jumped else None
        )
        return True

    @staticmethod
    def _challenger(scene: Scene, agent: Hashable) -> None:
        """Put a defender in the air between the ball's flight and its own net."""
        rng = scene.rng
        ahead = rng.uniform(0.5, 1.1)
        meeting = (
            scene.ball_position + scene.ball_velocity * ahead - UP * 0.5 * GRAVITY * ahead * ahead
        )
        meeting[2] = max(meeting[2], 250.0)
        for _ in range(12):
            position = meeting + np.array(
                [rng.normal(0.0, 300.0), rng.uniform(500.0, 1000.0), rng.uniform(-300.0, 200.0)]
            )
            position[0], position[1] = clamp_to_field(position[0], position[1], 300.0)
            position[2] = float(np.clip(position[2], 300.0, 1850.0))
            from_car = np.linalg.norm(position - scene.position(scene.me))
            from_ball = np.linalg.norm(position - scene.ball_position)
            if from_car >= _CHALLENGER_GAP and from_ball >= _CHALLENGER_GAP:
                break
        else:
            return  # no room for it next to the play: this start goes without
        toward = meeting - position
        gap = float(np.linalg.norm(toward))
        toward = toward / gap if gap > 1.0 else np.array([0.0, -1.0, 0.0])
        roof = UP if abs(toward[2]) < 0.95 else np.array([0.0, 1.0, 0.0])
        scene.air_car(
            agent,
            position,
            toward * rng.uniform(700.0, 1300.0),
            toward,
            roof,
            rng.uniform(30.0, 100.0),
            air_time=rng.uniform(0.3, 0.9),
        )
