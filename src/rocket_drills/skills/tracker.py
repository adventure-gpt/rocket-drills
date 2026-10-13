"""The skill tracker: feed it game states, get events and rates per car-minute."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM, TICKS_PER_SECOND

from rocket_drills.skills.detectors import Detector, default_detectors
from rocket_drills.skills.events import SkillEvent, Step

_TEAM_NAMES = {BLUE_TEAM: "blue", ORANGE_TEAM: "orange"}


@dataclass(frozen=True)
class SkillSummary:
    """Totals of a :class:`SkillTracker`, and the rates they amount to.

    Attributes:
        skills: Names of the events that were being counted.
        counts: Number of events of each skill, per car.
        team_counts: Number of events of each skill, per team. This includes the events that have
            no car (see :class:`~rocket_drills.skills.GoalDetector`).
        car_minutes: Minutes of play that were tracked, per car.
        teams: The team of each car.
    """

    skills: tuple[str, ...]
    counts: Mapping[Hashable, Mapping[str, int]]
    team_counts: Mapping[int, Mapping[str, int]]
    car_minutes: Mapping[Hashable, float]
    teams: Mapping[Hashable, int]

    def per_car(self) -> dict[Hashable, dict[str, float]]:
        """Return the events per car-minute of each skill, for each car."""
        return {
            agent: {
                skill: _rate(counts.get(skill, 0), self.car_minutes[agent]) for skill in self.skills
            }
            for agent, counts in self.counts.items()
        }

    def per_team(self) -> dict[int, dict[str, float]]:
        """Return the events per car-minute of each skill, for each team.

        A team's events are divided by the minutes of all its cars together, so the figure is the
        rate of an average car of that team and can be compared between team sizes.
        """
        rates = {}
        for team, counts in self.team_counts.items():
            minutes = sum(m for agent, m in self.car_minutes.items() if self.teams[agent] == team)
            rates[team] = {skill: _rate(counts.get(skill, 0), minutes) for skill in self.skills}
        return rates

    def format(self, per_car: bool = False) -> str:
        """Lay the rates out as a text table: one row per skill, one column per team or car.

        Args:
            per_car: Give every car a column instead of every team.
        """
        if per_car:
            columns = {str(agent): rates for agent, rates in self.per_car().items()}
            minutes = {str(agent): value for agent, value in self.car_minutes.items()}
        else:
            columns = {
                _TEAM_NAMES.get(team, f"team {team}"): rates
                for team, rates in sorted(self.per_team().items())
            }
            minutes = {
                _TEAM_NAMES.get(team, f"team {team}"): sum(
                    m for agent, m in self.car_minutes.items() if self.teams[agent] == team
                )
                for team in sorted(self.team_counts)
            }
        label_width = max([len("events per car-minute"), *(len(skill) for skill in self.skills)])
        width = max([10, *(len(name) + 2 for name in columns)])
        lines = [
            "events per car-minute".ljust(label_width)
            + "".join(name.rjust(width) for name in columns)
        ]
        for skill in self.skills:
            cells = "".join(f"{rates[skill]:{width}.3f}" for rates in columns.values())
            lines.append(skill.ljust(label_width) + cells)
        cells = "".join(f"{minutes[name]:{width}.2f}" for name in columns)
        lines.append("car-minutes".ljust(label_width) + cells)
        return "\n".join(lines)


def _rate(count: int, minutes: float) -> float:
    return count / minutes if minutes > 0.0 else 0.0


class SkillTracker:
    """Counts mechanics in a stream of game states.

    Call :meth:`reset` with the first state of every episode and :meth:`step` with each state
    after it. ``step`` returns the events of that step; the totals build up over episodes until
    :meth:`clear`, and :meth:`summary` turns them into events per car-minute::

        tracker = SkillTracker()
        env.reset()
        tracker.reset(env.state)
        ...
        env.step(actions)
        events = tracker.step(env.state)
        ...
        print(tracker.summary().format())

    The tracker reads nothing but ``GameState`` objects, so it works on states from any source: an
    RLGym environment, the transition engine directly, or replays converted with ``rlgym-tools``.
    It may be stepped every tick or every few ticks; the definitions are in seconds.

    Args:
        detectors: The detectors to run, in order. A detector sees the events of the ones before
            it, so one that builds on another (:class:`~rocket_drills.skills.ShotAfter`) goes
            after it. Defaults to :func:`~rocket_drills.skills.default_detectors`.
        touch_gap: Seconds a car must be out of contact with the ball before its next contact
            counts as a new touch.

    Raises:
        ValueError: If two detectors emit an event of the same name.
    """

    def __init__(
        self, detectors: Sequence[Detector] | None = None, *, touch_gap: float = 0.05
    ) -> None:
        self.detectors: list[Detector] = (
            default_detectors() if detectors is None else list(detectors)
        )
        names = [skill for detector in self.detectors for skill in detector.skills]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"more than one detector emits {duplicates}")
        self.skills: tuple[str, ...] = tuple(names)
        self.touch_gap = touch_gap
        self._counts: dict[Hashable, dict[str, int]] = {}
        self._team_counts: dict[int, dict[str, int]] = {}
        self._seconds: dict[Hashable, float] = {}
        self._teams: dict[Hashable, int] = {}
        self._prev: GameState | None = None
        self._time = 0.0
        self._contact_times: dict[Hashable, float] = {}
        self._event_times: dict[tuple[str, Hashable | None], float] = {}
        self._last_toucher: Hashable | None = None

    def reset(self, state: GameState) -> None:
        """Start a new episode from ``state``. The totals are kept."""
        self._prev = state
        self._time = 0.0
        self._contact_times = {}
        self._event_times = {}
        self._last_toucher = None
        for detector in self.detectors:
            detector.reset(state)

    def step(self, state: GameState, dt: float | None = None) -> list[SkillEvent]:
        """Advance to ``state`` and return the events that happened on the way.

        Args:
            state: The state that follows the one given to the previous call.
            dt: Seconds since that state. By default it is taken from the difference of the two
                ``tick_count`` values at 120 ticks per second.

        Raises:
            RuntimeError: If :meth:`reset` has not been called.
            ValueError: If no time has passed, or the cars are not the ones of the previous state.
        """
        prev = self._prev
        if prev is None:
            raise RuntimeError("call reset() with the first state of the episode before step()")
        if dt is None:
            dt = (state.tick_count - prev.tick_count) / TICKS_PER_SECOND
        if dt <= 0.0:
            raise ValueError(
                "no time has passed since the previous state: pass dt, or call reset() at the "
                "start of each episode"
            )
        if state.cars.keys() != prev.cars.keys():
            raise ValueError("the cars changed: call reset() at the start of each episode")

        self._time += dt
        new_touches = set()
        for agent, car in state.cars.items():
            if car.ball_touches > 0:
                last = self._contact_times.get(agent)
                if last is None or self._time - dt - last >= self.touch_gap - 1e-9:
                    new_touches.add(agent)
                self._contact_times[agent] = self._time
                self._last_toucher = agent

        step = Step(
            prev=prev,
            state=state,
            dt=dt,
            time=self._time,
            _new_touches=frozenset(new_touches),
            _contact_times=self._contact_times,
            _event_times=self._event_times,
            _last_toucher=self._last_toucher,
        )
        for detector in self.detectors:
            detector.update(step)

        for agent, car in state.cars.items():
            self._seconds[agent] = self._seconds.get(agent, 0.0) + dt
            self._teams[agent] = car.team_num
            self._counts.setdefault(agent, {})
            self._team_counts.setdefault(car.team_num, {})
        for event in step.events:
            team = self._team_counts.setdefault(event.team, {})
            team[event.skill] = team.get(event.skill, 0) + 1
            if event.agent is not None:
                own = self._counts.setdefault(event.agent, {})
                own[event.skill] = own.get(event.skill, 0) + 1
        self._prev = state
        return step.events

    def clear(self) -> None:
        """Set the totals back to zero."""
        self._counts = {}
        self._team_counts = {}
        self._seconds = {}
        self._teams = {}

    def count(self, skill: str, agent: Hashable | None = None) -> int:
        """Return how many ``skill`` events have been counted, for one car or for all of them.

        Raises:
            KeyError: If no detector of this tracker emits ``skill``.
        """
        if skill not in self.skills:
            raise KeyError(f"no detector emits {skill!r}; this tracker counts {self.skills}")
        if agent is not None:
            return self._counts.get(agent, {}).get(skill, 0)
        return sum(counts.get(skill, 0) for counts in self._team_counts.values())

    def summary(self) -> SkillSummary:
        """Return the totals so far and the rates per car-minute."""
        return SkillSummary(
            skills=self.skills,
            counts={agent: dict(counts) for agent, counts in self._counts.items()},
            team_counts={team: dict(counts) for team, counts in self._team_counts.items()},
            car_minutes={agent: seconds / 60.0 for agent, seconds in self._seconds.items()},
            teams=dict(self._teams),
        )
