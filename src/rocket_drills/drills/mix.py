"""A weighted mixture of drills and other state mutators."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import numpy as np
from rlgym.api import StateMutator
from rlgym.rocket_league.api import GameState

from rocket_drills.drills.base import DRILL_INFO_KEY, Drill, DrillInfo, Seed


class DrillMix(StateMutator[GameState]):
    """Start every episode from one of several mutators, picked at random by weight.

    The entries can be drills, RLGym's own mutators, or anything else with an
    ``apply(state, shared_info)`` method::

        DrillMix({KickoffMutator(): 0.6, Takeoff(): 0.2, FlipResetChain(): 0.2})

    The weights are relative; they do not have to sum to one. Whatever is picked, the choice is
    recorded: after a reset ``shared_info["drill"]`` is the :class:`DrillInfo` the drill wrote, or
    one that names the class of the mutator when the pick was not a drill.

    Args:
        drills: A mapping from mutator to weight, or an iterable of ``(mutator, weight)`` pairs.
            Use the pairs to give the same mutator object more than one entry.
        seed: Seed or generator for the choice. When given, every drill in the mix is also
            reseeded from it, so that one seed fixes the whole sequence of starts. (Mutators that
            are not drills keep their own randomness: RLGym's ``KickoffMutator`` draws from
            Python's ``random`` module.)

    Attributes:
        mutators: The entries, in the order given.
        probabilities: The probability of each entry.
        counts: How many times each entry has been picked so far.

    Raises:
        ValueError: If there are no entries, a weight is negative, or the weights sum to zero.
    """

    def __init__(
        self,
        drills: Mapping[StateMutator[GameState], float]
        | Iterable[tuple[StateMutator[GameState], float]],
        *,
        seed: Seed = None,
    ) -> None:
        pairs = list(drills.items()) if isinstance(drills, Mapping) else list(drills)
        if not pairs:
            raise ValueError("DrillMix needs at least one mutator")
        self.mutators: tuple[StateMutator[GameState], ...] = tuple(m for m, _ in pairs)
        weights = np.array([float(weight) for _, weight in pairs])
        if (weights < 0.0).any():
            raise ValueError("weights must not be negative")
        if weights.sum() <= 0.0:
            raise ValueError("at least one weight must be positive")
        self.probabilities: np.ndarray = weights / weights.sum()
        self.counts: list[int] = [0] * len(pairs)
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            self.reseed(seed)

    def reseed(self, seed: Seed) -> None:
        """Restart the choice, and every drill in the mix, from ``seed``."""
        if isinstance(seed, np.random.Generator):
            sequence = np.random.SeedSequence(int(seed.integers(2**63)))
        elif isinstance(seed, np.random.SeedSequence):
            sequence = seed
        else:
            sequence = np.random.SeedSequence(seed)
        own, *children = sequence.spawn(len(self.mutators) + 1)
        self._rng = np.random.default_rng(own)
        for mutator, child in zip(self.mutators, children, strict=True):
            if isinstance(mutator, (Drill, DrillMix)):
                mutator.reseed(child)

    def apply(self, state: GameState, shared_info: dict[str, Any]) -> None:
        """Pick one mutator by weight and apply it."""
        index = int(self._rng.choice(len(self.mutators), p=self.probabilities))
        mutator = self.mutators[index]
        self.counts[index] += 1
        shared_info.pop(DRILL_INFO_KEY, None)
        mutator.apply(state, shared_info)
        if DRILL_INFO_KEY not in shared_info:
            shared_info[DRILL_INFO_KEY] = DrillInfo(name=type(mutator).__name__)
