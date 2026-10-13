"""The tracker's bookkeeping: episodes, time, totals, rates, and the reward adapter."""

from __future__ import annotations

import pytest
from rlgym.rocket_league.reward_functions import CombinedReward, GoalReward

from helpers import after, quiet_state, skills_of
from rocket_drills.placement import empty_state, set_ball
from rocket_drills.skills import (
    SKILL_EVENTS_KEY,
    Detector,
    ShotDetector,
    SkillEvent,
    SkillReward,
    SkillTracker,
    TouchDetector,
)

BLUE, ORANGE = "blue-0", "orange-0"


def touch_episode(tracker: SkillTracker, touches: dict[str, int], steps: int = 15, ticks: int = 8):
    """One episode of ``steps`` steps in which each car makes some separate touches."""
    state = quiet_state()
    tracker.reset(state)
    events = []
    for step in range(steps):
        state = after(state, ticks)
        for agent, count in touches.items():
            state.cars[agent].ball_touches = int(step % 3 == 0 and step // 3 < count)
        events += tracker.step(state)
    return events


def test_step_needs_a_reset_first_and_time_to_have_passed():
    tracker = SkillTracker()
    with pytest.raises(RuntimeError, match="reset"):
        tracker.step(quiet_state())
    start = quiet_state()
    tracker.reset(start)
    with pytest.raises(ValueError, match="no time has passed"):
        tracker.step(after(start, ticks=0))
    assert tracker.step(after(start, ticks=0), dt=1 / 120) == []  # unless the caller says how much
    with pytest.raises(ValueError, match="cars changed"):
        tracker.step(after(empty_state(2, 2), ticks=8))


def test_time_comes_from_the_tick_count_or_from_dt():
    tracker = SkillTracker([TouchDetector()])
    start = quiet_state()
    tracker.reset(start)
    tracker.step(after(start, ticks=120))
    assert tracker.summary().car_minutes == pytest.approx({BLUE: 1 / 60, ORANGE: 1 / 60})
    tracker.step(after(start, ticks=240), dt=59.0)
    assert tracker.summary().car_minutes[BLUE] == pytest.approx(1.0)


def test_totals_build_up_over_episodes_until_cleared():
    tracker = SkillTracker([TouchDetector()])
    events = touch_episode(tracker, {BLUE: 2, ORANGE: 1})
    assert skills_of(events, BLUE) == ["touch", "touch"]
    assert all(isinstance(event, SkillEvent) for event in events)
    assert (events[0].team, events[0].tick) == (0, 8)
    touch_episode(tracker, {BLUE: 1})
    assert tracker.count("touch") == 4
    assert tracker.count("touch", BLUE) == 3
    assert tracker.count("touch", ORANGE) == 1
    with pytest.raises(KeyError, match="no detector emits"):
        tracker.count("shot")
    tracker.clear()
    assert tracker.count("touch") == 0
    assert tracker.summary().car_minutes == {}


def test_summary_gives_events_per_car_minute_for_cars_and_teams():
    tracker = SkillTracker([TouchDetector(), ShotDetector()])
    state = empty_state(2, 1)
    set_ball(state, (0.0, 0.0, 93.15), (0.0, 0.0, 0.0))
    tracker.reset(state)
    for step in range(450):  # 450 steps of 8 ticks: 30 seconds
        state = after(state)
        state.cars["blue-0"].ball_touches = int(step % 50 == 0)  # 9 touches
        state.cars["blue-1"].ball_touches = int(step % 150 == 0)  # 3 touches
        tracker.step(state)
    summary = tracker.summary()
    assert summary.skills == ("touch", "shot")
    assert summary.car_minutes == pytest.approx({"blue-0": 0.5, "blue-1": 0.5, "orange-0": 0.5})
    assert summary.counts["blue-0"] == {"touch": 9}
    assert summary.teams == {"blue-0": 0, "blue-1": 0, "orange-0": 1}
    per_car = summary.per_car()
    assert per_car["blue-0"] == pytest.approx({"touch": 18.0, "shot": 0.0})
    assert per_car["blue-1"]["touch"] == pytest.approx(6.0)
    assert per_car["orange-0"]["touch"] == 0.0
    per_team = summary.per_team()
    assert per_team[0]["touch"] == pytest.approx(12.0)  # 12 touches in one car-minute of blue
    assert per_team[1] == {"touch": 0.0, "shot": 0.0}


def test_the_summary_formats_as_a_table():
    tracker = SkillTracker([TouchDetector()])
    touch_episode(tracker, {BLUE: 2}, steps=450)
    lines = tracker.summary().format().splitlines()
    assert lines[0].split() == ["events", "per", "car-minute", "blue", "orange"]
    assert lines[1].split() == ["touch", "4.000", "0.000"]
    assert lines[2].split() == ["car-minutes", "0.50", "0.50"]
    by_car = tracker.summary().format(per_car=True).splitlines()
    assert by_car[0].split()[-2:] == [BLUE, ORANGE]
    assert SkillTracker().summary().format().startswith("events per car-minute")  # nothing yet


def test_detectors_are_run_in_order_and_must_not_share_event_names():
    class Second(Detector):
        skills = ("second",)

        def update(self, step):
            for agent in step.state.cars:
                if step.happened("touch", agent):
                    assert step.last_time("touch", agent) == step.time
                    step.emit("second", agent)

    tracker = SkillTracker([TouchDetector(), Second()])
    assert tracker.skills == ("touch", "second")
    assert skills_of(touch_episode(tracker, {BLUE: 1}), BLUE) == ["touch", "second"]
    with pytest.raises(ValueError, match="more than one detector"):
        SkillTracker([TouchDetector(), TouchDetector()])


def test_the_touch_gap_is_a_setting_of_the_tracker():
    def touches(tracker):
        return len(touch_episode(tracker, {BLUE: 3}, steps=9, ticks=8))

    assert touches(SkillTracker([TouchDetector()])) == 3  # two clean steps between contacts
    assert touches(SkillTracker([TouchDetector()], touch_gap=0.5)) == 1


def test_skill_reward_pays_per_event_and_shares_the_events():
    reward = SkillReward({"touch": 0.5, "shot": 2.0})
    shared_info: dict = {}
    start = quiet_state()
    set_ball(start, (0.0, 3000.0, 93.15), (1500.0, 300.0, 0.0))
    agents = list(start.cars)
    reward.reset(agents, start, shared_info)
    assert shared_info[SKILL_EVENTS_KEY] == []
    hit = after(start)
    set_ball(hit, (0.0, 3100.0, 93.15), (0.0, 2500.0, 0.0))
    hit.cars[BLUE].ball_touches = 1
    hit.cars[ORANGE].ball_touches = 1
    done = dict.fromkeys(agents, False)
    rewards = reward.get_rewards(agents, hit, done, done, shared_info)
    assert rewards == {BLUE: 2.5, ORANGE: 0.5}  # both touched; only blue attacks that goal
    assert skills_of(shared_info[SKILL_EVENTS_KEY], BLUE) == ["touch", "shot"]
    quiet = reward.get_rewards(agents, after(hit), done, done, shared_info)
    assert quiet == {BLUE: 0.0, ORANGE: 0.0}
    assert reward.tracker.count("shot") == 1


def test_skill_reward_checks_its_weights_and_combines_with_rlgyms_rewards():
    with pytest.raises(ValueError, match="unknown skill"):
        SkillReward({"flip_rest": 1.0})
    tracker = SkillTracker([TouchDetector()])
    assert SkillReward({"touch": 1.0}, tracker).tracker is tracker
    with pytest.raises(ValueError, match="unknown skill"):
        SkillReward({"shot": 1.0}, tracker)

    combined = CombinedReward((GoalReward(), 10.0), (SkillReward({"touch": 1.0}), 0.5))
    start = quiet_state()
    agents = list(start.cars)
    combined.reset(agents, start, {})
    hit = after(start)
    hit.cars[ORANGE].ball_touches = 1
    done = dict.fromkeys(agents, False)
    assert combined.get_rewards(agents, hit, done, done, {}) == {BLUE: 0.0, ORANGE: 0.5}
