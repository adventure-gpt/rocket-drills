"""Skill counters: detectors that watch consecutive game states and count mechanics.

:class:`SkillTracker` is the entry point. It runs a list of small detectors, one per mechanic,
over each pair of consecutive states and keeps totals per car and per team::

    from rocket_drills.skills import SkillTracker

    tracker = SkillTracker()
    tracker.reset(first_state)
    for state in later_states:
        for event in tracker.step(state):
            print(event.skill, event.agent)
    print(tracker.summary().format())
"""

from rocket_drills.skills.detectors import (
    AerialTouchDetector,
    BackboardBounceDetector,
    CeilingDropDetector,
    Detector,
    FlickDetector,
    FlipResetDetector,
    GoalDetector,
    SaveDetector,
    ShotAfter,
    ShotDetector,
    TouchDetector,
    WavedashDetector,
    default_detectors,
    in_the_air,
    wheels_on_ball,
)
from rocket_drills.skills.events import SkillEvent, Step
from rocket_drills.skills.reward import SKILL_EVENTS_KEY, SkillReward
from rocket_drills.skills.tracker import SkillSummary, SkillTracker

__all__ = [
    "SKILL_EVENTS_KEY",
    "AerialTouchDetector",
    "BackboardBounceDetector",
    "CeilingDropDetector",
    "Detector",
    "FlickDetector",
    "FlipResetDetector",
    "GoalDetector",
    "SaveDetector",
    "ShotAfter",
    "ShotDetector",
    "SkillEvent",
    "SkillReward",
    "SkillSummary",
    "SkillTracker",
    "Step",
    "TouchDetector",
    "WavedashDetector",
    "default_detectors",
    "in_the_air",
    "wheels_on_ball",
]
