"""An RLGym reward function that pays for detected mechanics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState

from rocket_drills.skills.tracker import SkillTracker

#: Key under which :class:`SkillReward` stores the events of the last step in ``shared_info``.
SKILL_EVENTS_KEY = "skill_events"


class SkillReward(RewardFunction[AgentID, GameState, float]):
    """Pays each car a fixed amount for every mechanic it is detected doing.

    A thin adapter from :class:`SkillTracker` to RLGym's ``RewardFunction``::

        reward_fn = CombinedReward(
            (GoalReward(), 10.0),
            SkillReward({"flip_reset_shot": 2.0, "double_tap": 2.0, "wavedash": 0.2}),
        )

    The events of the last step are also put in ``shared_info["skill_events"]``, and the tracker
    is available as :attr:`tracker` for its totals.

    Paying for a mechanic is not the same as teaching it: a policy will collect whatever the
    detector counts in the cheapest way it finds. The events that end in a shot
    (``flip_reset_shot``, ``ceiling_shot``, ``double_tap``, ``flick_shot``) are harder to farm
    than the ones that do not.

    Args:
        weights: Reward per event, by skill name. Skills that are not listed pay nothing.
        tracker: The tracker to use; a default one is created if omitted. Do not also step this
            tracker yourself: the reward function resets and steps it.

    Raises:
        ValueError: If a weight names a skill that the tracker does not count.
    """

    def __init__(self, weights: Mapping[str, float], tracker: SkillTracker | None = None) -> None:
        self.tracker = tracker if tracker is not None else SkillTracker()
        unknown = sorted(set(weights) - set(self.tracker.skills))
        if unknown:
            raise ValueError(
                f"unknown skill {unknown}; the tracker counts {list(self.tracker.skills)}"
            )
        self.weights: dict[str, float] = {name: float(value) for name, value in weights.items()}

    def reset(
        self, agents: list[AgentID], initial_state: GameState, shared_info: dict[str, Any]
    ) -> None:
        """Start a new episode."""
        self.tracker.reset(initial_state)
        shared_info[SKILL_EVENTS_KEY] = []

    def get_rewards(
        self,
        agents: list[AgentID],
        state: GameState,
        is_terminated: dict[AgentID, bool],
        is_truncated: dict[AgentID, bool],
        shared_info: dict[str, Any],
    ) -> dict[AgentID, float]:
        """Step the tracker and pay for the events of this step."""
        events = self.tracker.step(state)
        shared_info[SKILL_EVENTS_KEY] = events
        rewards = dict.fromkeys(agents, 0.0)
        for event in events:
            if event.agent in rewards:
                rewards[event.agent] += self.weights.get(event.skill, 0.0)
        return rewards
