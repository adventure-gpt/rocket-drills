"""Shared fixtures. The tests marked ``sim`` step RocketSim and skip when it cannot be started."""

from __future__ import annotations

import pytest

from rocket_drills.sim import SimulatorUnavailableError, make_engine


@pytest.fixture(scope="session")
def engine():
    """One RocketSim engine for the whole run, without RLBot's one-tick action delay.

    The collision meshes come from ``ROCKET_DRILLS_MESHES`` if it is set, else from the copy
    inside RLGym's package (see ``rocket_drills.sim.make_engine``). If the simulator cannot be
    started, every test that asks for it is skipped with the reason.
    """
    try:
        return make_engine(rlbot_delay=False)
    except SimulatorUnavailableError as error:
        pytest.skip(f"simulator unavailable: {error}")
