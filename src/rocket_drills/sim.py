"""Starting the simulator, and telling clearly when it cannot be started.

RocketSim steps the physics and needs the arena's collision meshes to do it. This package ships
none. With ``rlgym-rocket-league`` 2.0.1 nothing has to be set up: that package carries the soccar
meshes and RLGym's ``RocketSimEngine`` loads them from there. :func:`make_engine` is for the other
cases: an install without them, or meshes you want to choose yourself.
"""

from __future__ import annotations

import contextlib
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlgym.rocket_league.sim import RocketSimEngine

#: Environment variable that names a folder of collision meshes for :func:`make_engine` to use.
MESHES_ENV = "ROCKET_DRILLS_MESHES"


class SimulatorUnavailableError(RuntimeError):
    """RocketSim is not installed, or it found no collision meshes."""


def make_engine(
    meshes: str | os.PathLike[str] | None = None, *, rlbot_delay: bool = True
) -> RocketSimEngine:
    """Create RLGym's ``RocketSimEngine``, pointing RocketSim at collision meshes first if asked.

    The meshes come from the first of these that is given:

    1. the ``meshes`` argument;
    2. the folder named by the ``ROCKET_DRILLS_MESHES`` environment variable;
    3. the copy inside the installed ``rlgym-rocket-league`` package, which is what
       ``RocketSimEngine`` initialises RocketSim with when nothing else has.

    A meshes folder contains one sub-folder per game mode (``soccar``) with the ``.cmf`` files.

    RocketSim takes its meshes once per process. The first folder it is given stays in use, even
    if it turns out to hold no meshes, and a later call with a different one has no effect. So a
    folder of your own has to be given before the first engine of the process is created, here or
    with ``RocketSim.init(folder)``. (On its own RocketSim would fall back to the
    ``RS_COLLISION_MESHES`` environment variable and to ``./collision_meshes``; RLGym's engine
    gives it a folder before it gets that far.)

    Args:
        meshes: Folder of collision meshes, or ``None`` to use the order above.
        rlbot_delay: Passed to ``RocketSimEngine``: delay every action by one tick, as RLBot does.

    Raises:
        SimulatorUnavailableError: If RocketSim is not installed or cannot load the arena.
    """
    try:
        import RocketSim
        from rlgym.rocket_league.sim import RocketSimEngine
    except ImportError as error:
        raise SimulatorUnavailableError(
            "RocketSim is not installed; install it with: pip install 'rlgym-rocket-league[sim]'"
        ) from error

    folder = meshes if meshes is not None else os.environ.get(MESHES_ENV)
    if folder:
        # RocketSim refuses a second initialisation: the meshes loaded first stay in use.
        with contextlib.suppress(RuntimeError):
            RocketSim.init(os.fspath(folder))
    try:
        return RocketSimEngine(rlbot_delay=rlbot_delay)
    except RuntimeError as error:
        raise SimulatorUnavailableError(
            "RocketSim could not load the arena's collision meshes"
            + (f" from {os.fspath(folder)!r}" if folder else "")
            + f". Give make_engine(meshes=...) or the {MESHES_ENV} environment variable a folder "
            "that contains soccar/*.cmf, before any other engine is created in this process. "
            f"RocketSim said: {error}"
        ) from error
