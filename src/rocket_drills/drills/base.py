"""The base class of every drill and the record a drill leaves about the start it made."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

from rocket_drills.drills.scene import (
    OPPONENT_LAYOUTS,
    TEAMMATE_LAYOUTS,
    OpponentLayout,
    Scene,
    TeammateLayout,
)

#: Key under which a drill stores its :class:`DrillInfo` in RLGym's ``shared_info`` dictionary.
DRILL_INFO_KEY = "drill"

Seed = int | np.random.Generator | np.random.SeedSequence | None
Stages = Mapping[str, float] | Iterable[str] | str | None
Range = tuple[float, float]


@dataclass(frozen=True)
class DrillInfo:
    """What a drill did on the last reset. Found at ``shared_info["drill"]``.

    Attributes:
        name: Name of the drill (its class name), or of the mutator a :class:`DrillMix` picked.
        stage: Which of the drill's stages this start is; ``None`` for a mutator that is not a
            drill.
        agent: The drill car: the car the start is built around.
        team: The team of the drill car.
        details: What else the drill promises about the start, such as ``"target"`` (where the
            ball will be, in arena coordinates) and ``"target_time"`` (seconds until then).
    """

    name: str
    stage: str | None = None
    agent: Hashable | None = None
    team: int | None = None
    details: Mapping[str, Any] = field(default_factory=dict)


def normalise_weights(
    choice: Stages, defaults: Mapping[str, float], what: str = "stage"
) -> dict[str, float]:
    """Turn a stage selection into probabilities over the names in ``defaults``.

    Args:
        choice: ``None`` for the default weights, one name, several names (equal weights) or a
            mapping from name to weight.
        defaults: The names that exist, with their default weights.
        what: What the names are called in error messages.

    Returns:
        A probability for every name in ``defaults``, in that order, zero for the ones left out.

    Raises:
        ValueError: On an unknown name, a negative weight, or weights that sum to zero.
    """
    if choice is None:
        weights = dict(defaults)
    elif isinstance(choice, str):
        weights = {choice: 1.0}
    elif isinstance(choice, Mapping):
        weights = {str(name): float(weight) for name, weight in choice.items()}
    else:
        weights = {str(name): 1.0 for name in choice}
    unknown = sorted(set(weights) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {what} {unknown}; choose from {list(defaults)}")
    if any(weight < 0.0 for weight in weights.values()):
        raise ValueError(f"{what} weights must not be negative")
    total = sum(weights.values())
    if total <= 0.0:
        raise ValueError(f"at least one {what} needs a positive weight")
    return {name: weights.get(name, 0.0) / total for name in defaults}


def check_range(name: str, value: Range, lowest: float | None = None) -> Range:
    """Validate a ``(low, high)`` parameter and return it as floats.

    Raises:
        ValueError: If ``low`` is above ``high`` or below ``lowest``.
    """
    low, high = float(value[0]), float(value[1])
    if low > high:
        raise ValueError(f"{name} must be (low, high) with low <= high, got {value}")
    if lowest is not None and low < lowest:
        raise ValueError(f"{name} must not go below {lowest}, got {value}")
    return low, high


def check_share(name: str, value: float) -> float:
    """Validate a parameter that is a probability.

    Raises:
        ValueError: If the value is outside 0 to 1.
    """
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return float(value)


class Drill(StateMutator[GameState]):
    """A practice scenario: an RLGym state mutator that starts an episode in one situation.

    A drill writes the ball and every car. Use it wherever RLGym takes a ``StateMutator``, after
    a mutator that creates the cars::

        MutatorSequence(FixedTeamSizeMutator(2, 2), Takeoff())

    It works for any team sizes as long as the teams it needs are present. The start is built
    around one car, the *drill car*, which is the first car of the drill team; the rest are placed
    according to ``opponents`` and ``teammates``. After every reset
    ``shared_info["drill"]`` holds a :class:`DrillInfo` naming the drill car and the stage.

    Blue and orange starts are mirror images of each other: a drill seeded the same way gives, for
    ``team=ORANGE_TEAM``, the blue start turned half a turn about the centre of the pitch.

    Args:
        stages: Which of the drill's stages to start from: ``None`` for the drill's default
            weights, one name, several names (equal weights) or a mapping from name to weight.
        team: The team of the drill car, ``BLUE_TEAM`` or ``ORANGE_TEAM``; ``None`` picks one at
            random on every reset.
        opponents: Where the other team's cars start unless they have a part in the drill:
            ``"defending"`` (in their own half, the first of them in or near its net),
            ``"random"`` (anywhere on the floor) or ``"away"`` (parked in front of their own
            goal, out of the play).
        teammates: Where the drill car's team-mates start: ``"behind"`` (on the floor behind
            it), ``"defending"``, ``"random"`` or ``"away"``.
        seed: Seed or generator for the drill's random numbers.

    Attributes:
        STAGES: The drill's stages and their default weights.
        stages: The probability of each stage, as chosen at construction.
    """

    STAGES: ClassVar[Mapping[str, float]] = {"default": 1.0}

    def __init__(
        self,
        *,
        stages: Stages = None,
        team: int | None = None,
        opponents: OpponentLayout = "defending",
        teammates: TeammateLayout = "behind",
        seed: Seed = None,
    ) -> None:
        if team not in (None, BLUE_TEAM, ORANGE_TEAM):
            raise ValueError(f"team must be BLUE_TEAM (0), ORANGE_TEAM (1) or None, got {team!r}")
        if opponents not in OPPONENT_LAYOUTS:
            raise ValueError(f"opponents must be one of {OPPONENT_LAYOUTS}, got {opponents!r}")
        if teammates not in TEAMMATE_LAYOUTS:
            raise ValueError(f"teammates must be one of {TEAMMATE_LAYOUTS}, got {teammates!r}")
        self.stages: dict[str, float] = normalise_weights(stages, self.STAGES)
        self.team = team
        self.opponents: OpponentLayout = opponents
        self.teammates: TeammateLayout = teammates
        self._rng = np.random.default_rng(seed)

    @property
    def name(self) -> str:
        """The drill's name, as it appears in :class:`DrillInfo` and in reports."""
        return type(self).__name__

    def reseed(self, seed: Seed) -> None:
        """Restart the drill's random numbers from ``seed``."""
        self._rng = np.random.default_rng(seed)

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        """Write one start into ``state`` and a :class:`DrillInfo` into ``shared_info``.

        Raises:
            ValueError: If the state has no cars, or lacks a car the drill needs.
            RuntimeError: If the drill's ``_build`` left the ball or the drill car unplaced.
        """
        if not state.cars:
            raise ValueError(
                f"{self.name} found no cars: apply a mutator that creates them first, for example "
                "MutatorSequence(FixedTeamSizeMutator(1, 1), drill)"
            )
        rng = self._rng
        names = list(self.stages)
        stage = names[int(rng.choice(len(names), p=[self.stages[name] for name in names]))]
        team = self._choose_team(state)
        scene = Scene(state, team, rng)
        details = self._build(scene, stage)
        if not scene.has_ball or not scene.is_placed(scene.me):
            raise RuntimeError(f"{self.name}._build must place the ball and the drill car")
        scene.fill(self.opponents, self.teammates)
        shared_info[DRILL_INFO_KEY] = DrillInfo(
            name=self.name, stage=stage, agent=scene.me, team=team, details=dict(details or {})
        )

    def _choose_team(self, state: GameState) -> int:
        present = {car.team_num for car in state.cars.values()}
        if self.team is not None:
            if self.team not in present:
                raise ValueError(f"{self.name} was given team={self.team}, which has no cars")
            return self.team
        coin = self._rng.random() < 0.5
        if len(present) == 1:
            return next(iter(present))
        return ORANGE_TEAM if coin else BLUE_TEAM

    def _need_opponent(self, scene: Scene) -> Hashable:
        if not scene.opponents:
            raise ValueError(f"{self.name} needs at least one car on each team")
        return scene.opponents[0]

    @abstractmethod
    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any] | None:
        """Place the ball and every car that has a part in the drill.

        Args:
            scene: The start being written, in the frame where the drill team attacks ``+y``.
            stage: The stage that was drawn.

        Returns:
            The ``details`` of the :class:`DrillInfo`, if the drill has any to give.
        """
        raise NotImplementedError
