"""DrillMix: weights honoured, RLGym's own mutators welcome, the pick recorded."""

from __future__ import annotations

import numpy as np
import pytest
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from rocket_drills import DrillMix
from rocket_drills.drills import Aerial, DrillInfo, FlipResetChain, Save, Takeoff
from rocket_drills.placement import empty_state


def picks(mix: DrillMix, count: int, sizes: tuple[int, int] = (1, 1)) -> list[DrillInfo]:
    found = []
    for _ in range(count):
        shared_info: dict = {}
        mix.apply(empty_state(*sizes), shared_info)
        found.append(shared_info["drill"])
    return found


def test_weights_are_honoured():
    mix = DrillMix(
        {KickoffMutator(): 0.6, Takeoff(): 0.1, FlipResetChain(): 0.1, Save(): 0.2}, seed=1
    )
    assert mix.probabilities.tolist() == pytest.approx([0.6, 0.1, 0.1, 0.2])
    names = [info.name for info in picks(mix, 2000)]
    shares = {name: names.count(name) / len(names) for name in set(names)}
    assert shares == pytest.approx(
        {"KickoffMutator": 0.6, "Takeoff": 0.1, "FlipResetChain": 0.1, "Save": 0.2}, abs=0.035
    )
    assert mix.counts == [
        names.count(n) for n in ("KickoffMutator", "Takeoff", "FlipResetChain", "Save")
    ]


def test_weights_are_relative_and_a_zero_weight_is_never_picked():
    mix = DrillMix({Takeoff(): 3.0, Aerial(): 1.0, Save(): 0.0}, seed=2)
    assert mix.probabilities.tolist() == pytest.approx([0.75, 0.25, 0.0])
    names = [info.name for info in picks(mix, 400)]
    assert "Save" not in names
    assert names.count("Takeoff") / 400 == pytest.approx(0.75, abs=0.06)


def test_a_mutator_that_is_not_a_drill_is_named_in_the_info():
    mix = DrillMix({KickoffMutator(): 1.0}, seed=3)
    shared_info: dict = {"drill": "stale"}
    state = empty_state(2, 2)
    mix.apply(state, shared_info)
    info = shared_info["drill"]
    assert info == DrillInfo(name="KickoffMutator")
    assert info.stage is None
    assert info.agent is None
    assert state.ball.position.tolist() == [0.0, 0.0, pytest.approx(93.15)]  # a kickoff it is


def test_a_drill_leaves_its_own_info():
    info = picks(DrillMix({FlipResetChain(stages="reset"): 1.0}, seed=4), 1)[0]
    assert (info.name, info.stage, info.agent) == ("FlipResetChain", "reset", info.agent)
    assert info.agent in ("blue-0", "orange-0")


def test_pairs_allow_the_same_object_twice_and_mixes_nest():
    drill = Takeoff()
    mix = DrillMix([(drill, 1.0), (drill, 1.0), (DrillMix({Aerial(): 1.0}), 2.0)], seed=5)
    assert mix.probabilities.tolist() == pytest.approx([0.25, 0.25, 0.5])
    names = [info.name for info in picks(mix, 300)]
    assert set(names) == {"Takeoff", "Aerial"}
    assert names.count("Aerial") / 300 == pytest.approx(0.5, abs=0.09)


def test_a_seed_fixes_the_picks_and_the_starts_of_the_drills():
    def run(seed):
        mix = DrillMix({Takeoff(): 1.0, Aerial(): 1.0, FlipResetChain(): 1.0}, seed=seed)
        out = []
        for _ in range(30):
            state, shared_info = empty_state(1, 1), {}
            mix.apply(state, shared_info)
            out.append((shared_info["drill"].name, *state.ball.position.tolist()))
        return out

    assert run(6) == run(6)
    assert run(6) != run(7)
    mix = DrillMix({Takeoff(): 1.0, Aerial(): 1.0}, seed=np.random.default_rng(8))
    first = [info.name for info in picks(mix, 20)]
    mix.reseed(np.random.default_rng(8))
    assert [info.name for info in picks(mix, 20)] == first


def test_a_mix_composes_with_rlgyms_mutators():
    sequence = MutatorSequence(
        FixedTeamSizeMutator(2, 2), DrillMix({KickoffMutator(): 0.5, Takeoff(): 0.5}, seed=9)
    )
    seen = set()
    for _ in range(40):
        state = empty_state(0, 0)
        shared_info: dict = {}
        sequence.apply(state, shared_info)
        assert len(state.cars) == 4
        assert all(np.isfinite(car.physics.position).all() for car in state.cars.values())
        seen.add(shared_info["drill"].name)
    assert seen == {"KickoffMutator", "Takeoff"}


def test_bad_mixes_are_refused():
    with pytest.raises(ValueError, match="at least one mutator"):
        DrillMix({})
    with pytest.raises(ValueError, match="negative"):
        DrillMix({Takeoff(): -1.0, Aerial(): 2.0})
    with pytest.raises(ValueError, match="positive"):
        DrillMix({Takeoff(): 0.0})
