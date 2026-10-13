"""An RLGym environment that starts from a mix of drills, with the skill counters running.

Run it with:  python examples/quickstart.py

The cars take random actions, so the counts are near zero: what a policy that has learned nothing
looks like. Replace the random actions with your own policy's to measure it.
"""

from collections import Counter

import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from rocket_drills import DrillMix, SkillTracker
from rocket_drills.drills import FlipResetChain, Save, Takeoff, WavedashLanding


def main() -> None:
    mix = DrillMix(
        {
            KickoffMutator(): 0.4,
            Takeoff(): 0.2,
            FlipResetChain(): 0.2,
            Save(): 0.1,
            WavedashLanding(): 0.1,
        }
    )
    env = RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(blue_size=2, orange_size=2), mix),
        obs_builder=DefaultObs(zero_padding=2),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=GoalReward(),
        transition_engine=RocketSimEngine(),
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(timeout_seconds=6.0),
    )

    tracker = SkillTracker()
    starts = Counter()
    rng = np.random.default_rng(0)
    for _ in range(100):
        env.reset()
        tracker.reset(env.state)
        info = env.shared_info["drill"]  # which start this episode got, and which car it is for
        starts[info.name if info.stage in (None, "default") else f"{info.name} / {info.stage}"] += 1
        done = False
        while not done:
            actions = {agent: np.array([rng.integers(90)]) for agent in env.agents}
            _, _, terminated, truncated = env.step(actions)
            for event in tracker.step(env.state):
                if event.skill in ("flip_reset", "wavedash"):
                    print(f"tick {event.tick}: {event.agent} did a {event.skill}")
            done = any(terminated.values()) or any(truncated.values())

    print("\nstarts:")
    for name, count in starts.most_common():
        print(f"  {count:3d}  {name}")
    print()
    print(tracker.summary().format())


if __name__ == "__main__":
    main()
