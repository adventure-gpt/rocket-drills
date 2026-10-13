"""Practice scenarios: RLGym state mutators that start an episode in one situation.

Every drill is a :class:`Drill`, which is an RLGym ``StateMutator``. Apply it after a mutator that
creates the cars, alone or in a :class:`DrillMix`::

    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )
    from rocket_drills.drills import DrillMix, FlipResetChain, Takeoff

    mutator = MutatorSequence(
        FixedTeamSizeMutator(1, 1),
        DrillMix({KickoffMutator(): 0.6, Takeoff(): 0.2, FlipResetChain(): 0.2}),
    )
"""

from rocket_drills.drills.aerial import Aerial, Backboard, Ceiling
from rocket_drills.drills.base import DRILL_INFO_KEY, Drill, DrillInfo
from rocket_drills.drills.dribble import Dribble, FlickDefence
from rocket_drills.drills.flip_reset import FlipResetChain
from rocket_drills.drills.mix import DrillMix
from rocket_drills.drills.saves import FastSave, Save
from rocket_drills.drills.scene import Scene
from rocket_drills.drills.takeoff import Takeoff
from rocket_drills.drills.wall_air import WallToAirChain
from rocket_drills.drills.wavedash import WavedashLanding

#: Every drill class, in the order of the catalogue.
ALL_DRILLS: tuple[type[Drill], ...] = (
    Takeoff,
    Aerial,
    FlipResetChain,
    WallToAirChain,
    Ceiling,
    Backboard,
    Dribble,
    FlickDefence,
    Save,
    FastSave,
    WavedashLanding,
)

__all__ = [
    "ALL_DRILLS",
    "DRILL_INFO_KEY",
    "Aerial",
    "Backboard",
    "Ceiling",
    "Dribble",
    "Drill",
    "DrillInfo",
    "DrillMix",
    "FastSave",
    "FlickDefence",
    "FlipResetChain",
    "Save",
    "Scene",
    "Takeoff",
    "WallToAirChain",
    "WavedashLanding",
]
