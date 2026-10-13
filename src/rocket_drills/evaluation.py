"""Run a policy on a drill and report what it makes of each stage."""

from __future__ import annotations

import math
from collections.abc import Hashable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from rlgym.api import StateMutator, TransitionEngine
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, TICKS_PER_SECOND
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator

from rocket_drills.drills.base import DRILL_INFO_KEY, DrillInfo
from rocket_drills.scripted import Policy, idle
from rocket_drills.sim import make_engine
from rocket_drills.skills.tracker import SkillTracker

#: The outcomes :func:`evaluate_drill` reports, in the order of the table.
OUTCOMES: tuple[str, ...] = (
    "airborne",
    "first_touch",
    "touch",
    "air_touch",
    "flip_reset",
    "dodge",
    "shot",
    "save",
    "goal",
    "conceded",
)

_AIRBORNE_WINDOW = 2.0  # seconds from the start in which leaving the ground counts as "airborne"
_AIRBORNE_HEIGHT = 60.0


@dataclass(frozen=True)
class StageResult:
    """Outcome rates of the starts of one stage.

    Attributes:
        drill: Name of the drill (or of the mutator, if it was not a drill).
        stage: Name of the stage.
        starts: Number of episodes that started from it.
        rates: Share of those episodes in which each of :data:`OUTCOMES` happened.
    """

    drill: str
    stage: str | None
    starts: int
    rates: Mapping[str, float]


@dataclass(frozen=True)
class DrillReport:
    """What :func:`evaluate_drill` found.

    Attributes:
        seconds: Length of each episode.
        results: One entry per drill and stage that came up, in order of first appearance.
    """

    seconds: float
    results: tuple[StageResult, ...]

    def rate(self, outcome: str, stage: str | None = None) -> float:
        """Return the share of starts with ``outcome``, over one stage or over all of them.

        Raises:
            KeyError: If ``outcome`` is not one of :data:`OUTCOMES`.
            ValueError: If no episode started from ``stage``.
        """
        if outcome not in OUTCOMES:
            raise KeyError(f"unknown outcome {outcome!r}; choose from {OUTCOMES}")
        chosen = [r for r in self.results if stage is None or r.stage == stage]
        starts = sum(result.starts for result in chosen)
        if starts == 0:
            raise ValueError(f"no starts from stage {stage!r}")
        return sum(result.rates[outcome] * result.starts for result in chosen) / starts

    def format(self) -> str:
        """Lay the report out as a text table: one row per stage, one column per outcome."""
        labels = [
            result.drill
            if result.stage in (None, "default")
            else f"{result.drill} / {result.stage}"
            for result in self.results
        ]
        width = max([len("stage"), *(len(label) for label in labels)])
        header = "stage".ljust(width) + "  starts" + "".join(f"  {o:>11s}" for o in OUTCOMES)
        lines = [f"share of starts with each outcome within {self.seconds:g} s", header]
        for label, result in zip(labels, self.results, strict=True):
            cells = "".join(f"  {result.rates[outcome]:11.3f}" for outcome in OUTCOMES)
            lines.append(label.ljust(width) + f"  {result.starts:6d}" + cells)
        return "\n".join(lines)


def evaluate_drill(
    drill: StateMutator[GameState],
    policy: Policy,
    *,
    opponent_policy: Policy = idle,
    episodes: int = 200,
    seconds: float = 5.0,
    team_size: int | tuple[int, int] = 1,
    tick_skip: int = 8,
    engine: TransitionEngine[Hashable, GameState, np.ndarray] | None = None,
) -> DrillReport:
    """Play ``episodes`` starts of a drill and count what the drill car achieves from each stage.

    Every episode starts from the drill, runs for ``seconds`` or until a goal, and is scored on
    whether each of these happened to the drill car (``DrillInfo.agent``) in that time:

    * ``airborne``: it was off the ground, more than 60 uu up, at some point of the first 2 s
      (only tells you something for the stages that start on the ground);
    * ``first_touch``: the first touch of the ball by any car was its touch;
    * ``touch``: it touched the ball;
    * ``air_touch``: it touched the ball in the air (an ``aerial_touch`` or
      ``air_dribble_touch`` of the skill tracker);
    * ``flip_reset``: it got a flip reset;
    * ``dodge``: it started a dodge in the air;
    * ``shot`` / ``save``: it put the ball on target, or took it off target for its own goal;
    * ``goal`` / ``conceded``: its team scored, or the other team did.

    The definitions are those of :mod:`rocket_drills.skills`. With a
    :class:`~rocket_drills.drills.DrillMix` the report has a row per drill and stage; starts from
    a mutator that is not a drill are scored for the first blue car.

    Args:
        drill: The drill (or any state mutator) to start from.
        policy: Controls the drill car and its team: a function ``policy(state, agent)`` that
            returns the eight controls for one car. To evaluate a trained network, wrap its
            observation builder, forward pass and action parser in such a function.
        opponent_policy: Controls the other team.
        episodes: Number of starts.
        seconds: Length of an episode.
        team_size: Cars per team, or ``(blue, orange)``.
        tick_skip: Ticks for which each decision is held (8 is the usual 15 decisions a second).
        engine: The transition engine to run on. Defaults to
            :func:`rocket_drills.sim.make_engine`.

    Returns:
        The share of starts with each outcome, per stage.

    Raises:
        ValueError: If ``episodes`` or ``tick_skip`` is less than one, or ``seconds`` is not
            positive.
    """
    if episodes < 1 or tick_skip < 1 or seconds <= 0.0:
        raise ValueError("episodes and tick_skip must be at least 1 and seconds positive")
    if engine is None:
        engine = make_engine()
    blue, orange = (team_size, team_size) if isinstance(team_size, int) else team_size
    sizes = FixedTeamSizeMutator(blue, orange)
    tracker = SkillTracker()
    steps = math.ceil(seconds * TICKS_PER_SECOND / tick_skip)
    tallies: dict[tuple[str, str | None], dict[str, int]] = {}

    for _ in range(episodes):
        shared_info: dict[str, Any] = {}
        state = engine.create_base_state()
        sizes.apply(state, shared_info)
        drill.apply(state, shared_info)
        state = engine.set_state(state, shared_info)
        info = shared_info.get(DRILL_INFO_KEY)
        if not isinstance(info, DrillInfo):
            info = DrillInfo(name=type(drill).__name__)
        me, team = info.agent, info.team
        if me is None:
            me = next(a for a, car in state.cars.items() if car.team_num == BLUE_TEAM)
            team = BLUE_TEAM

        tracker.reset(state)
        seen = dict.fromkeys(OUTCOMES, False)
        ball_touched = False
        for step in range(steps):
            actions = {}
            for agent, car in state.cars.items():
                chosen = policy if car.team_num == team else opponent_policy
                controls = np.asarray(chosen(state, agent), dtype=np.float32).reshape(1, 8)
                actions[agent] = np.repeat(controls, tick_skip, axis=0)
            before = state.cars[me]
            state = engine.step(actions, shared_info)
            car = state.cars[me]
            skills = {event.skill for event in tracker.step(state) if event.agent == me}

            up = not car.on_ground and car.physics.position[2] > _AIRBORNE_HEIGHT
            if up and (step + 1) * tick_skip <= _AIRBORNE_WINDOW * TICKS_PER_SECOND:
                seen["airborne"] = True
            if car.ball_touches > 0:
                seen["touch"] = True
                seen["first_touch"] |= not ball_touched
            ball_touched |= any(other.ball_touches > 0 for other in state.cars.values())
            seen["air_touch"] |= bool(skills & {"aerial_touch", "air_dribble_touch"})
            seen["flip_reset"] |= "flip_reset" in skills
            seen["dodge"] |= car.has_flipped and not before.has_flipped and not car.on_ground
            seen["shot"] |= "shot" in skills
            seen["save"] |= "save" in skills
            if state.goal_scored:
                seen["goal" if state.scoring_team == team else "conceded"] = True
                break

        tally = tallies.setdefault((info.name, info.stage), dict.fromkeys(("starts", *OUTCOMES), 0))
        tally["starts"] += 1
        for outcome, happened in seen.items():
            tally[outcome] += int(happened)

    results = tuple(
        StageResult(
            drill=name,
            stage=stage,
            starts=tally["starts"],
            rates={outcome: tally[outcome] / tally["starts"] for outcome in OUTCOMES},
        )
        for (name, stage), tally in tallies.items()
    )
    return DrillReport(seconds=seconds, results=results)
