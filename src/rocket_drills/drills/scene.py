"""The frame a drill is written in, and the placement of the cars that have no part in it."""

from __future__ import annotations

import math
from collections.abc import Hashable
from typing import Literal

import numpy as np
from rlgym.rocket_league.api import Car, GameState

from rocket_drills.geometry import (
    CAR_REST_HEIGHT,
    UP,
    Vector,
    clamp_to_field,
    team_sign,
    xy_in_field,
)
from rocket_drills.placement import mark_jumped, set_ball, set_car

OpponentLayout = Literal["defending", "random", "away"]
TeammateLayout = Literal["behind", "defending", "random", "away"]

OPPONENT_LAYOUTS: tuple[str, ...] = ("defending", "random", "away")
TEAMMATE_LAYOUTS: tuple[str, ...] = ("behind", "defending", "random", "away")

_DEFAULT_REST_HEIGHT = 17.03
_MIN_GAP = 300.0  # closest a car without a part in the drill starts to another car or the ball
_WALL_MARGIN = 150.0  # closest it starts to a wall
_TRIES = 24
# Where "away" cars are parked: in a row across the pitch in front of their own goal line,
# then in further rows in front of that one.
_AWAY_X = (-2400.0, 2400.0, -1600.0, 1600.0, -3200.0, 3200.0, -800.0, 800.0)
_AWAY_ROWS = (4600.0, 4200.0, 3800.0)
_AWAY_BOOST = 33.0


class Scene:
    """One start, written in the drill's own frame.

    In that frame the drill team attacks ``+y`` and defends the goal at ``-y``, whichever colour
    it plays. Every position, velocity and direction handed to a scene is turned half a turn about
    the vertical axis on its way into the state when the drill team is orange, so an orange start
    is the exact mirror image of the blue start made from the same random numbers.

    Attributes:
        state: The state being written.
        team: The drill team.
        rng: The drill's random generator.
        me: The drill car: the first car of the drill team.
        mates: The other cars of the drill team.
        opponents: The cars of the other team.
    """

    def __init__(self, state: GameState, team: int, rng: np.random.Generator) -> None:
        self.state = state
        self.team = team
        self.rng = rng
        own = [agent for agent, car in state.cars.items() if car.team_num == team]
        self.me: Hashable = own[0]
        self.mates: list[Hashable] = own[1:]
        self.opponents: list[Hashable] = [
            agent for agent, car in state.cars.items() if car.team_num != team
        ]
        self._sign = team_sign(team)
        self._placed: dict[Hashable, np.ndarray] = {}
        self._ball: np.ndarray | None = None
        self._ball_velocity: np.ndarray | None = None
        self._clear: list[tuple[float, float, float]] = []

    def to_world(self, vector: Vector) -> np.ndarray:
        """Turn a position, velocity or direction from the drill's frame into the arena's."""
        array = np.asarray(vector, dtype=np.float64)
        return np.array([self._sign * array[0], self._sign * array[1], array[2]], dtype=np.float32)

    def rest_height(self, agent: Hashable) -> float:
        """Return the height of a car's origin when it stands on its wheels."""
        return CAR_REST_HEIGHT.get(self.state.cars[agent].hitbox_type, _DEFAULT_REST_HEIGHT)

    def position(self, agent: Hashable) -> np.ndarray:
        """Return where a car was placed, in the drill's frame."""
        return self._placed[agent]

    @property
    def has_ball(self) -> bool:
        """Whether the ball has been placed."""
        return self._ball is not None

    def is_placed(self, agent: Hashable) -> bool:
        """Tell whether a car has been placed."""
        return agent in self._placed

    @property
    def ball_position(self) -> np.ndarray:
        """Where the ball was placed, in the drill's frame."""
        if self._ball is None:
            raise RuntimeError("the ball has not been placed yet")
        return self._ball

    @property
    def ball_velocity(self) -> np.ndarray:
        """The velocity the ball was given, in the drill's frame."""
        if self._ball_velocity is None:
            raise RuntimeError("the ball has not been placed yet")
        return self._ball_velocity

    def ball(
        self, position: Vector, velocity: Vector, angular_velocity: Vector = (0.0, 0.0, 0.0)
    ) -> None:
        """Place the ball."""
        self._ball = np.asarray(position, dtype=np.float64)
        self._ball_velocity = np.asarray(velocity, dtype=np.float64)
        set_ball(
            self.state,
            self.to_world(position),
            self.to_world(velocity),
            self.to_world(angular_velocity),
        )

    def ground_car(
        self,
        agent: Hashable,
        x: float,
        y: float,
        yaw: float,
        speed: float,
        boost: float,
        *,
        forward_yaw: float | None = None,
    ) -> None:
        """Place a car on the floor, moving at ``speed`` in the direction ``yaw``.

        ``forward_yaw`` turns its nose another way (a car that is reversing, say); by default it
        faces the way it moves.
        """
        facing = yaw if forward_yaw is None else forward_yaw
        self._write(
            agent,
            (x, y, self.rest_height(agent)),
            (math.cos(yaw) * speed, math.sin(yaw) * speed, 0.0),
            (math.cos(facing), math.sin(facing), 0.0),
            UP,
            boost,
            on_ground=True,
        )

    def surface_car(
        self,
        agent: Hashable,
        position: Vector,
        velocity: Vector,
        forward: Vector,
        up: Vector,
        boost: float,
    ) -> None:
        """Place a car with its wheels on a wall or on the ceiling."""
        self._write(agent, position, velocity, forward, up, boost, on_ground=True)

    def air_car(
        self,
        agent: Hashable,
        position: Vector,
        velocity: Vector,
        forward: Vector,
        up: Vector,
        boost: float,
        *,
        air_time: float | None = None,
    ) -> None:
        """Place a car in the air.

        ``air_time`` is how long ago its jump ended (see
        :func:`rocket_drills.placement.mark_jumped`); ``None`` leaves it as a car that never
        jumped and so holds its dodge until it lands.
        """
        self._write(agent, position, velocity, forward, up, boost, on_ground=False)
        if air_time is not None:
            mark_jumped(self.state.cars[agent], air_time)

    def keep_clear(self, x: float, y: float, radius: float) -> None:
        """Keep the cars placed by :meth:`fill` at least ``radius`` away from a ground position."""
        self._clear.append((float(x), float(y), float(radius)))

    def fill(self, opponents: OpponentLayout, teammates: TeammateLayout) -> None:
        """Place every car that has no part in the drill.

        Layouts:
            ``"defending"``: in the car's own half, guarding its net. The first one in or near
            the goal mouth, the others between the halfway line and the net, facing up the pitch.

            ``"behind"`` (team-mates only): on the floor behind the drill car, facing up the pitch.

            ``"random"``: anywhere on the floor, facing and rolling any way.

            ``"away"``: parked in a row 520 uu in front of its own goal line, out of the play.

        A car that would start within 300 uu of another car or of the ball, or inside an area the
        drill asked to keep clear, is drawn again, and parked as an ``"away"`` car if that keeps
        failing.
        """
        groups: tuple[tuple[list[Hashable], str, float], ...] = (
            (self.opponents, opponents, 1.0),
            (self.mates, teammates, -1.0),
        )
        for cars, layout, side in groups:
            waiting = [agent for agent in cars if agent not in self._placed]
            for index, agent in enumerate(waiting):
                self._fill_one(agent, layout, side, keeper=index == 0)

    def _fill_one(self, agent: Hashable, layout: str, side: float, *, keeper: bool) -> None:
        for attempt in range(_TRIES if layout != "away" else 0):
            x, y, yaw, speed, boost = self._sample(layout, side, keeper, again=attempt > 0)
            x, y = clamp_to_field(x, y, _WALL_MARGIN)
            if self.is_free(x, y):
                self.ground_car(agent, x, y, yaw, speed, boost)
                return
        slots = [(x, side * y) for y in _AWAY_ROWS for x in _AWAY_X]
        x, y = next((slot for slot in slots if self.is_free(*slot)), slots[-1])
        self.ground_car(agent, x, y, -side * math.pi / 2.0, 0.0, _AWAY_BOOST)

    def _sample(
        self, layout: str, side: float, keeper: bool, *, again: bool
    ) -> tuple[float, float, float, float, float]:
        rng = self.rng
        if layout == "defending":
            if again:  # the first place was taken or had to stay clear: anywhere in its half
                x, y = rng.uniform(-3200.0, 3200.0), side * rng.uniform(600.0, 4900.0)
            elif keeper:
                x, y = rng.uniform(-1200.0, 1200.0), side * rng.uniform(3600.0, 4900.0)
            else:
                x, y = rng.uniform(-3000.0, 3000.0), side * rng.uniform(500.0, 4200.0)
            yaw = -side * math.pi / 2.0 + rng.normal(0.0, 0.5)
            return x, y, yaw, rng.uniform(0.0, 700.0), rng.uniform(20.0, 100.0)
        if layout == "behind":
            x = rng.uniform(-3200.0, 3200.0)
            y = float(np.clip(self._placed[self.me][1] - rng.uniform(400.0, 2600.0), -4800, 4200))
            yaw = math.pi / 2.0 + rng.normal(0.0, 0.5)
            return x, y, yaw, rng.uniform(0.0, 900.0), rng.uniform(10.0, 100.0)
        x, y = rng.uniform(-3600.0, 3600.0), rng.uniform(-4600.0, 4600.0)
        yaw = rng.uniform(-math.pi, math.pi)
        return x, y, yaw, rng.uniform(0.0, 1400.0), rng.uniform(0.0, 100.0)

    def is_free(self, x: float, y: float) -> bool:
        """Tell whether a car may be put on the floor at ``(x, y)``.

        It may when the spot is inside the walls, at least 300 uu from the ball and from every
        car placed so far, and outside the areas the drill asked to keep clear.
        """
        if not xy_in_field(x, y, _WALL_MARGIN - 1.0):
            return False
        here = np.array([x, y, _DEFAULT_REST_HEIGHT])
        if self._ball is not None and np.linalg.norm(here - self._ball) < _MIN_GAP:
            return False
        if any(np.linalg.norm(here - other) < _MIN_GAP for other in self._placed.values()):
            return False
        return all(math.hypot(x - cx, y - cy) >= radius for cx, cy, radius in self._clear)

    def _write(
        self,
        agent: Hashable,
        position: Vector,
        velocity: Vector,
        forward: Vector,
        up: Vector,
        boost: float,
        *,
        on_ground: bool,
    ) -> None:
        car: Car = self.state.cars[agent]
        self._placed[agent] = np.asarray(position, dtype=np.float64)
        set_car(
            car,
            self.to_world(position),
            self.to_world(velocity),
            self.to_world(forward),
            self.to_world(up),
            boost=boost,
            on_ground=on_ground,
        )
