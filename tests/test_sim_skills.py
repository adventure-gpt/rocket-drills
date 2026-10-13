"""The skill counters on real physics, and the whole toolkit inside an RLGym environment.

These step RocketSim and are skipped when it cannot be started (see ``conftest.py``).
"""

from __future__ import annotations

import numpy as np
import pytest
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward
from rlgym.rocket_league.state_mutators import (
    FixedTeamSizeMutator,
    KickoffMutator,
    MutatorSequence,
)

from helpers import begin, play, skills_of
from rocket_drills import DrillMix, SkillReward, SkillTracker, scripted
from rocket_drills.drills import ALL_DRILLS, FlipResetChain, Save
from rocket_drills.placement import empty_state, mark_jumped, set_ball, set_car
from rocket_drills.skills import FlickDetector

pytestmark = pytest.mark.sim

BLUE = "blue-0"


def lone_car(
    engine, ball, ball_velocity, car, car_velocity, forward, *, in_air=False, air_time=None
):
    """Hand a one-car state to the engine and return its first state."""
    state = engine.create_base_state()
    state.cars = empty_state(1, 0).cars
    set_ball(state, ball, ball_velocity)
    set_car(state.cars[BLUE], car, car_velocity, forward, boost=50.0, on_ground=not in_air)
    if air_time is not None:
        mark_jumped(state.cars[BLUE], air_time)
    return engine.set_state(state, {})


def controls(**pressed) -> np.ndarray:
    names = ("throttle", "steer", "pitch", "yaw", "roll", "jump", "boost", "handbrake")
    return np.array([[pressed.get(name, 0.0) for name in names]], dtype=np.float32)


def test_a_shot_and_its_goal_go_to_the_car_that_dodged_into_the_ball(engine):
    drill = FlipResetChain(stages="finish", keeper_out_share=1.0, seed=5)
    scored = credited = shots = 0
    for _ in range(30):
        start, info, shared = begin(engine, drill)
        tracker = SkillTracker()
        tracker.reset(start)
        names = []
        for frame in play(engine, start, shared, {info.agent: scripted.forward_dodge}, 3.0):
            names += skills_of(tracker.step(frame), info.agent)
            if frame.goal_scored:
                scored += 1
                break
        shots += "shot" in names
        credited += "goal" in names
        assert "flip_reset" not in names  # it never had its wheels on the ball
    assert scored >= 18
    assert credited == scored
    assert shots >= 24


def test_a_shot_nobody_stops_is_a_goal_for_the_team_and_for_no_car(engine):
    def park(state, info):
        car = state.cars[info.agent]
        set_car(car, (3500.0, 0.0, 17.0), (0, 0, 0), (0.0, 1.0, 0.0), boost=0.0)

    drill = Save(seed=7, shooter=False, opponents="away")
    for _ in range(10):
        start, info, shared = begin(engine, drill, prepare=park)
        tracker = SkillTracker()
        tracker.reset(start)
        events = []
        for frame in play(engine, start, shared, {}, 5.0):
            events += tracker.step(frame)
            if frame.goal_scored:
                break
        assert [(e.skill, e.agent, e.team) for e in events] == [("goal", None, 1 - info.team)]


def flick_attempt(engine, speed, offset, hold, gap, pitch, tick_skip):
    """Carry the ball, jump for ``hold`` ticks, wait ``gap`` ticks, then dodge.

    ``pitch`` is the stick: 1 for a backflip, which brings the nose up into the ball, -1 for a
    front flip, which leaves the ball behind.

    Returns how many flicks the tracker counted when stepped every ``tick_skip`` ticks, and
    whether the dodge really touched the ball (always judged tick by tick).
    """
    schedule = np.zeros((144, 8), dtype=np.float32)
    schedule[:, 0] = 1.0  # throttle throughout; the first 24 ticks settle the ball on the roof
    schedule[24 : 24 + hold, 5] = 1.0  # jump
    schedule[24 + hold + gap :, 5] = 1.0  # jump again...
    schedule[24 + hold + gap :, 2] = pitch  # ...with the stick back or forward: a dodge

    def start():
        return lone_car(
            engine,
            ball=(0.0, -2000.0 + offset, 150.3),
            ball_velocity=(0.0, speed, 0.0),
            car=(0.0, -2000.0, 17.03),
            car_velocity=(0.0, speed, 0.0),
            forward=(0.0, 1.0, 0.0),
        )

    state = start()
    assert FlickDetector.on_roof(state.cars[BLUE], state.ball)
    really_hit = False
    for action in schedule:
        car = engine.step({BLUE: action[None]}, {}).cars[BLUE]
        really_hit |= bool(car.is_flipping and car.ball_touches > 0)

    state = start()  # the same physics again, read every tick_skip ticks
    tracker = SkillTracker()
    tracker.reset(state)
    counted = 0
    for first in range(0, len(schedule), tick_skip):
        state = engine.step({BLUE: schedule[first : first + tick_skip]}, {})
        counted += skills_of(tracker.step(state)).count("flick")
    return counted, really_hit


@pytest.mark.parametrize("tick_skip", [1, 8])
def test_a_dodge_touch_after_a_carry_is_a_flick(engine, tick_skip):
    flicks = misses = 0
    for speed in (500.0, 900.0):
        for offset in (-30.0, -10.0, 10.0, 30.0):
            for hold, gap, pitch in ((8, 1, 1.0), (16, 8, 1.0), (24, 8, 1.0), (16, 1, -1.0)):
                counted, really_hit = flick_attempt(
                    engine, speed, offset, hold, gap, pitch, tick_skip
                )
                assert counted == int(really_hit), (speed, offset, hold, gap, pitch)
                flicks += counted
                misses += 1 - counted
    assert flicks >= 5  # the grid has flicks...
    assert misses >= 5  # ...and dodges that did not connect, which must not count


def test_a_ball_knocked_onto_their_backboard_is_a_backboard_bounce(engine):
    state = lone_car(
        engine,
        ball=(0.0, 3900.0, 900.0),
        ball_velocity=(0.0, 600.0, 250.0),
        car=(0.0, 3650.0, 870.0),
        car_velocity=(0.0, 2000.0, 250.0),
        forward=(0.0, 1.0, 0.1),
        in_air=True,
        air_time=0.5,
    )
    tracker = SkillTracker()
    tracker.reset(state)
    names = []
    for _ in range(12):
        state = engine.step({BLUE: np.zeros((8, 8), dtype=np.float32)}, {})
        names += skills_of(tracker.step(state), BLUE)
    # The car flies on after its hit and may meet the rebound again; the first three are fixed.
    assert names[:3] == ["touch", "aerial_touch", "backboard_bounce"]


def test_the_quick_start_runs_in_an_rlgym_environment(engine):
    mix = DrillMix({KickoffMutator(): 1.0, **{cls(): 1.0 for cls in ALL_DRILLS}}, seed=1)
    skill_reward = SkillReward({"touch": 0.1, "shot": 1.0})
    env = RLGym(
        state_mutator=MutatorSequence(FixedTeamSizeMutator(2, 2), mix),
        obs_builder=DefaultObs(zero_padding=2),
        action_parser=RepeatAction(LookupTableAction(), repeats=8),
        reward_fn=CombinedReward((GoalReward(), 10.0), skill_reward),
        transition_engine=engine,
        termination_cond=GoalCondition(),
        truncation_cond=TimeoutCondition(3.0),
    )
    tracker = SkillTracker()
    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(40):
        observations = env.reset()
        seen.add(env.shared_info["drill"].name)
        tracker.reset(env.state)
        done = False
        while not done:
            actions = {agent: np.array([rng.integers(90)]) for agent in env.agents}
            observations, rewards, terminated, truncated = env.step(actions)
            tracker.step(env.state)
            assert set(rewards) == set(env.agents)
            assert all(np.isfinite(obs).all() for obs in observations.values())
            done = any(terminated.values()) or any(truncated.values())
    assert len(seen) >= 9  # the kickoff and most of the eleven drills came up
    summary = tracker.summary()
    assert set(summary.car_minutes) == {"blue-0", "blue-1", "orange-0", "orange-1"}
    assert sum(summary.car_minutes.values()) > 1.0
    assert tracker.count("touch") == skill_reward.tracker.count("touch") > 0
    assert "car-minutes" in summary.format()
