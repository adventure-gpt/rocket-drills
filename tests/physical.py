"""Physical checks of the drills in the simulator: what really happens from each start.

Every function plays ``count`` starts of one drill in RocketSim and returns the share of them in
which something held. ``test_sim_drills.py`` asserts lower bounds on these shares, and
``tools/measure_drills.py`` prints them all: its output is where the measured facts in
``docs/drills.md`` come from.
"""

from __future__ import annotations

import numpy as np

from helpers import begin, play
from rocket_drills import SkillTracker, scripted
from rocket_drills.drills import (
    Backboard,
    Ceiling,
    Dribble,
    FastSave,
    FlipResetChain,
    Save,
    Takeoff,
    WallToAirChain,
    WavedashLanding,
)
from rocket_drills.geometry import defended_goal_y, on_target, team_sign
from rocket_drills.placement import set_car
from rocket_drills.skills import FlickDetector

Shares = dict[str, float]


def _share(flags) -> float:
    return float(np.mean(flags)) if len(flags) else float("nan")


def park_the_drill_car(state, info) -> None:
    """Move the drill car to a far corner before the start, so that it touches nothing."""
    car = state.cars[info.agent]
    sign = team_sign(car.team_num)
    set_car(car, (3500.0 * sign, -4300.0 * sign, 17.0), (0, 0, 0), (0, 1, 0), boost=0.0)


def _skills(tracker, state, agent) -> set[str]:
    return {event.skill for event in tracker.step(state) if event.agent == agent}


def takeoff(engine, stage: str, count: int = 100, seed: int = 0) -> Shares:
    """The ball is where the drill says, when it says; a boosted jump is up there by then."""
    arrives, error = [], []
    drill = Takeoff(stages=stage, seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        target, time = info.details["target"], info.details["target_time"]
        ticks = round(time * 120)
        (then,) = play(engine, start, shared, {}, ticks / 120, tick_skip=ticks)
        error.append(float(np.linalg.norm(then.ball.position - target)))
        arrives.append(error[-1] < 40.0)

    high_enough, at_height, touched = [], [], []
    drill = Takeoff(stages=stage, seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        target, time = info.details["target"], info.details["target_time"]
        top, touch = 0.0, False
        for frame in play(engine, start, shared, {info.agent: scripted.boosted_jump}, time + 0.3):
            car = frame.cars[info.agent]
            if frame.tick_count <= (time + 0.07) * 120:
                top = max(top, float(car.physics.position[2]))
            touch |= car.ball_touches > 0
        high_enough.append(top >= target[2] - 130.0)
        at_height.append(top >= target[2])
        touched.append(touch)
    return {
        "ball_within_40uu_of_target_at_target_time": _share(arrives),
        "largest_distance_from_target": float(np.max(error)),
        "jump_within_130uu_of_target_height_in_time": _share(high_enough),
        "jump_at_target_height_in_time": _share(at_height),
        "jump_touches_ball": _share(touched),
    }


def flip_reset_finish(engine, count: int = 100, seed: int = 0, tick_skip: int = 8) -> Shares:
    """A forward dodge from the finish stage sends the ball at the goal."""
    holds, sent, shot, goal = [], [], [], []
    drill = FlipResetChain(stages="finish", seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        me, sign = info.agent, team_sign(info.team)
        car = start.cars[me]
        holds.append(car.can_flip and not car.has_jumped)
        tracker = SkillTracker()
        tracker.reset(start)
        before = sign * start.ball.linear_velocity[1]
        gain, shots, scored = 0.0, False, False
        for frame in play(engine, start, shared, {me: scripted.forward_dodge}, 3.0, tick_skip):
            shots |= "shot" in _skills(tracker, frame, me)
            gain = max(gain, sign * frame.ball.linear_velocity[1] - before)
            if frame.goal_scored:
                scored = frame.scoring_team == info.team
                break
        sent.append(gain > 300.0)
        shot.append(shots)
        goal.append(scored)
    return {
        "starts_holding_a_dodge": _share(holds),
        "dodge_adds_300_toward_goal": _share(sent),
        "dodge_is_counted_as_a_shot": _share(shot),
        "goal_within_3s": _share(goal),
    }


def flip_reset_contact(engine, count: int = 100, seed: int = 0, tick_skip: int = 8) -> Shares:
    """Coasting from the reset stage puts the wheels on the ball: a counted flip reset."""
    jumped, spent, counted, once, dodge_after = [], [], [], [], []
    drill = FlipResetChain(stages="reset", seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        me = info.agent
        car = start.cars[me]
        jumped.append(car.has_jumped and not car.on_ground)
        spent.append(not car.has_flip)
        tracker = SkillTracker()
        tracker.reset(start)
        resets, regained = 0, False
        for frame in play(engine, start, shared, {}, 1.0, tick_skip):
            resets += "flip_reset" in _skills(tracker, frame, me)
            car = frame.cars[me]
            regained |= resets > 0 and car.can_flip and not car.has_jumped
        counted.append(resets > 0)
        once.append(resets == 1)
        if resets:
            dodge_after.append(regained)
    return {
        "starts_in_the_air_having_jumped": _share(jumped),
        "starts_with_the_dodge_spent": _share(spent),
        "flip_reset_counted": _share(counted),
        "flip_reset_counted_exactly_once": _share(once),
        "holds_a_dodge_after_the_reset": _share(dodge_after),
    }


def wall_to_air(engine, stage: str, count: int = 100, seed: int = 0) -> Shares:
    """The drill car is where the stage says after the first eight ticks of physics."""
    on_wheels, on_floor, near_wall, airborne = [], [], [], []
    drill = WallToAirChain(stages=stage, seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        (then,) = play(engine, start, shared, {}, 8 / 120)
        car = then.cars[info.agent]
        position = car.physics.position
        from_wall = min(4096 - abs(position[0]), 5120 - abs(position[1]))
        on_wheels.append(car.on_ground)
        on_floor.append(car.on_ground and position[2] < 30.0)
        near_wall.append(car.on_ground and from_wall < 60.0)
        airborne.append(not car.on_ground)
    return {
        "wheels_down_after_8_ticks": _share(on_wheels),
        "on_the_floor_after_8_ticks": _share(on_floor),
        "on_a_wall_after_8_ticks": _share(near_wall),
        "in_the_air_after_8_ticks": _share(airborne),
    }


def save(engine, fast: bool, count: int = 100, seed: int = 0) -> Shares:
    """Left alone, the shot goes in; the lifted ones cross the line where and when stated."""
    cls = FastSave if fast else Save
    drill = cls(seed=seed, shooter=False, opponents="away")
    targeted, goes_in, place, late = [], [], [], []
    for _ in range(count):
        start, info, shared = begin(engine, drill, prepare=park_the_drill_car)
        ball = start.ball
        targeted.append(on_target(ball.position, ball.linear_velocity, defended_goal_y(info.team)))
        lifted = ball.position[2] > 150.0 or ball.linear_velocity[2] > 100.0
        target, time = info.details["target"], info.details["target_time"]
        scored, crossed = False, None
        for frame in play(engine, start, shared, {}, 5.0, tick_skip=1):
            if crossed is None and abs(frame.ball.position[1]) >= 5120.0:
                crossed = frame
            if frame.goal_scored:
                scored = True
                break
        goes_in.append(scored)
        if lifted and crossed is not None:
            place.append(float(np.linalg.norm((crossed.ball.position - target)[[0, 2]])))
            late.append(abs(crossed.tick_count / 120 - time))
    return {
        "on_target_at_the_start": _share(targeted),
        "goes_in_untouched_within_5s": _share(goes_in),
        "lifted_shots_cross_within_30uu_of_target": _share(np.array(place) < 30.0),
        "lifted_shots_cross_within_0.03s_of_target_time": _share(np.array(late) < 0.03),
    }


def dribble(engine, count: int = 100, seed: int = 0) -> Shares:
    """The ball stays on the roof of a car that keeps under it, and leaves one that coasts."""
    out: Shares = {}
    for name, policy in (("coasting", scripted.idle), ("carry", scripted.carry)):
        kept: dict[float, list[bool]] = {0.5: [], 1.0: [], 2.0: []}
        drill = Dribble(seed=seed, opponents="away")
        for _ in range(count):
            start, info, shared = begin(engine, drill)
            me = info.agent
            for frame in play(engine, start, shared, {me: policy}, 2.0):
                for seconds, flags in kept.items():
                    if frame.tick_count == round(seconds * 15) * 8:
                        flags.append(FlickDetector.on_roof(frame.cars[me], frame.ball))
        for seconds, flags in kept.items():
            out[f"{name}_ball_on_roof_after_{seconds}s"] = _share(flags)
    return out


def ceiling(engine, count: int = 100, seed: int = 0) -> Shares:
    """The car starts with its wheels on the ceiling, falls off by itself and keeps its dodge."""
    stuck, dropped, counted, holding = [], [], [], []
    drill = Ceiling(seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        me = info.agent
        tracker = SkillTracker()
        tracker.reset(start)
        first, drop, car = None, False, start.cars[me]
        for frame in play(engine, start, shared, {}, 0.6):
            car = frame.cars[me]
            if first is None:
                first = car.on_ground and car.physics.position[2] > 1950.0
            drop |= "ceiling_drop" in _skills(tracker, frame, me)
        stuck.append(first)
        dropped.append(not car.on_ground)
        counted.append(drop)
        holding.append(car.can_flip and not car.has_jumped)
    return {
        "wheels_on_the_ceiling_after_8_ticks": _share(stuck),
        "off_the_ceiling_within_0.6s": _share(dropped),
        "ceiling_drop_counted": _share(counted),
        "holds_a_dodge_in_the_air_after_0.6s": _share(holding),
    }


def backboard(engine, stage: str, count: int = 100, seed: int = 0) -> Shares:
    """Left alone, the ball comes off their backboard above the goal, where and when stated."""
    bounced, near, on_time = [], [], []
    drill = Backboard(stages=stage, seed=seed, opponents="away")
    for _ in range(count):
        start, info, shared = begin(engine, drill, prepare=park_the_drill_car)
        sign = team_sign(info.team)
        target, time = info.details["target"], info.details["target_time"]
        closest, when = np.inf, None
        toward = sign * start.ball.linear_velocity[1]
        for frame in play(engine, start, shared, {}, time + 0.5, tick_skip=1):
            ball = frame.ball
            away = sign * ball.linear_velocity[1]
            if when is None:
                closest = min(closest, float(np.linalg.norm(ball.position - target)))
                if toward > 300.0 and away < -100.0 and ball.position[2] > 650.0:
                    when = frame.tick_count / 120
            toward = away
        bounced.append(when is not None)
        near.append(closest < 30.0)
        on_time.append(when is not None and abs(when - time) < 0.05)
    return {
        "bounces_off_their_backboard_above_the_goal": _share(bounced),
        "passes_within_30uu_of_target": _share(near),
        "bounces_within_0.05s_of_target_time": _share(on_time),
    }


def wavedash(engine, count: int = 100, seed: int = 0, tick_skip: int = 8) -> Shares:
    """A scripted dodge into the landing gains the speed and is counted; idling is not."""
    gained, counted, idle_counted = [], [], []
    drill = WavedashLanding(seed=seed)
    for _ in range(count):
        start, info, shared = begin(engine, drill)
        me = info.agent
        other = next(agent for agent in start.cars if agent != me)
        tracker = SkillTracker()
        tracker.reset(start)
        before = float(np.linalg.norm(start.cars[me].physics.linear_velocity[:2]))
        peak, mine, theirs, landed = before, 0, 0, None
        for frame in play(engine, start, shared, {me: scripted.wavedash}, 1.0, tick_skip):
            events = tracker.step(frame)
            mine += sum(e.skill == "wavedash" and e.agent == me for e in events)
            theirs += sum(e.skill == "wavedash" and e.agent == other for e in events)
            car = frame.cars[me]
            if landed is None and car.on_ground:
                landed = frame.tick_count
            if landed is None or frame.tick_count <= landed + 16:
                peak = max(peak, float(np.linalg.norm(car.physics.linear_velocity[:2])))
        gained.append(peak - before >= 350.0)
        counted.append(mine >= 1)
        idle_counted.append(theirs >= 1)
    return {
        "script_gains_350_by_the_landing": _share(gained),
        "wavedash_counted": _share(counted),
        "wavedash_counted_for_the_idle_car": _share(idle_counted),
    }
