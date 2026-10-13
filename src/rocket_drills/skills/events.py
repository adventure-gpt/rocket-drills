"""The record of one detected mechanic, and the view of a step that the detectors work from."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field

from rlgym.rocket_league.api import Car, GameState


@dataclass(frozen=True)
class SkillEvent:
    """One mechanic, detected in one step.

    Attributes:
        skill: Name of the mechanic, such as ``"flip_reset"``.
        agent: The car that did it. ``None`` for the rare event that belongs to a team but to no
            car: a goal that nobody on the scoring team touched.
        team: The team of that car.
        tick: ``GameState.tick_count`` of the state in which it was detected.
    """

    skill: str
    agent: Hashable | None
    team: int
    tick: int


@dataclass
class Step:
    """Two consecutive game states and what has been detected between them so far.

    The tracker builds one per call and hands it to each detector in turn. A detector reads the
    two states, asks about touches and about the events of the detectors that ran before it, and
    calls :meth:`emit`.

    Attributes:
        prev: The state before the step.
        state: The state after it.
        dt: Seconds between the two.
        time: Seconds since the tracker was last reset, at ``state``.
        events: The events emitted in this step so far.
    """

    prev: GameState
    state: GameState
    dt: float
    time: float
    events: list[SkillEvent] = field(default_factory=list)
    _new_touches: frozenset[Hashable] = frozenset()
    _contact_times: dict[Hashable, float] = field(default_factory=dict)
    _event_times: dict[tuple[str, Hashable | None], float] = field(default_factory=dict)
    _last_toucher: Hashable | None = None

    def cars(self) -> list[tuple[Hashable, Car, Car]]:
        """Return ``(agent, car before, car after)`` for every car in the step."""
        return [(agent, self.prev.cars[agent], car) for agent, car in self.state.cars.items()]

    def touching(self, agent: Hashable) -> bool:
        """Tell whether the car was in contact with the ball at any tick of this step."""
        return self.state.cars[agent].ball_touches > 0

    def new_touch(self, agent: Hashable) -> bool:
        """Tell whether the car began a touch in this step.

        A touch begins when the car is in contact with the ball after at least the tracker's
        ``touch_gap`` seconds without contact, so one hit that lasts several ticks, or a ball that
        rests on the car, is one touch however often the tracker is stepped.
        """
        return agent in self._new_touches

    def last_contact(self, agent: Hashable) -> float | None:
        """Return the time of the car's latest contact with the ball, this step included."""
        return self._contact_times.get(agent)

    @property
    def last_toucher(self) -> Hashable | None:
        """The car that touched the ball most recently, this step included."""
        return self._last_toucher

    def emit(self, skill: str, agent: Hashable) -> None:
        """Record that ``agent`` did ``skill`` in this step."""
        car = self.state.cars[agent]
        self.events.append(SkillEvent(skill, agent, car.team_num, self.state.tick_count))
        self._event_times[(skill, agent)] = self.time

    def emit_for_team(self, skill: str, team: int) -> None:
        """Record an event that belongs to a team but to none of its cars."""
        self.events.append(SkillEvent(skill, None, team, self.state.tick_count))

    def happened(self, skill: str, agent: Hashable) -> bool:
        """Tell whether ``agent`` did ``skill`` in this step, as far as detection has got."""
        return any(event.skill == skill and event.agent == agent for event in self.events)

    def last_time(self, skill: str, agent: Hashable) -> float | None:
        """Return when ``agent`` last did ``skill`` since the reset, this step included."""
        return self._event_times.get((skill, agent))
