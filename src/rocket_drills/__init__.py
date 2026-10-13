"""Practice scenarios and skill counters for Rocket League bots trained with RLGym v2.

Two parts, usable together or apart:

* :mod:`rocket_drills.drills`: state mutators that start an episode in one situation (a ball in
  the air ahead of a grounded car, a car about to get a flip reset, a shot coming at a
  defender, ...), with :class:`DrillMix` to mix them with each other and with kickoffs.
* :mod:`rocket_drills.skills`: a :class:`SkillTracker` that watches consecutive game states and
  counts mechanics per car-minute, built from one small detector per mechanic.

:mod:`rocket_drills.evaluation` runs a policy on a drill and reports per-stage outcome rates;
:mod:`rocket_drills.scripted` has scripted baseline policies; :mod:`rocket_drills.placement` has
the helpers for writing your own drills.
"""

from rocket_drills.drills import DRILL_INFO_KEY, Drill, DrillInfo, DrillMix
from rocket_drills.evaluation import DrillReport, evaluate_drill
from rocket_drills.placement import empty_state, mark_jumped, set_ball, set_car
from rocket_drills.skills import SkillEvent, SkillReward, SkillSummary, SkillTracker

__version__ = "0.1.0"

__all__ = [
    "DRILL_INFO_KEY",
    "Drill",
    "DrillInfo",
    "DrillMix",
    "DrillReport",
    "SkillEvent",
    "SkillReward",
    "SkillSummary",
    "SkillTracker",
    "__version__",
    "empty_state",
    "evaluate_drill",
    "mark_jumped",
    "set_ball",
    "set_car",
]
