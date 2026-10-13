"""Each detector on hand-built sequences of game states: what counts, and what must not."""

from __future__ import annotations

import pytest
from rlgym.rocket_league.common_values import BACK_WALL_Y, CEILING_Z

from helpers import after, fly, quiet_state, skills_of
from rocket_drills.placement import set_ball, set_car
from rocket_drills.skills import (
    AerialTouchDetector,
    BackboardBounceDetector,
    CeilingDropDetector,
    FlickDetector,
    FlipResetDetector,
    GoalDetector,
    SaveDetector,
    ShotAfter,
    ShotDetector,
    SkillTracker,
    TouchDetector,
    WavedashDetector,
    default_detectors,
    in_the_air,
    wheels_on_ball,
)

BLUE, ORANGE = "blue-0", "orange-0"


def run(detectors, states, agent=None):
    """Feed states to a tracker; return the list of event names of each step."""
    tracker = SkillTracker(detectors)
    tracker.reset(states[0])
    return [skills_of(tracker.step(state), agent) for state in states[1:]]


def run_all(detectors, states, agent=None):
    """As ``run``, with the event names of all the steps in one list."""
    return [name for step in run(detectors, states, agent) for name in step]


def test_the_default_set_names_every_skill_once():
    names = [skill for detector in default_detectors() for skill in detector.skills]
    assert names == [
        "touch",
        "shot",
        "save",
        "goal",
        "aerial_touch",
        "air_dribble_touch",
        "flip_reset",
        "ceiling_drop",
        "backboard_bounce",
        "flick",
        "wavedash",
        "flip_reset_shot",
        "ceiling_shot",
        "double_tap",
        "flick_shot",
    ]
    assert SkillTracker().skills == tuple(names)


# --- touch --------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("ticks", "contact", "expected"),
    [
        # 8 ticks a step: contact in consecutive steps is one touch, a clean step makes a new one
        (8, [1, 1, 1, 0, 1, 0, 0, 1], [1, 0, 0, 0, 1, 0, 0, 1]),
        # every tick: a hit lasting three ticks, a one-tick gap, then a real second touch
        (1, [1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]),
    ],
)
def test_a_touch_begins_after_a_gap_without_contact(ticks, contact, expected):
    states = [quiet_state()]
    for touching in contact:
        state = after(states[-1], ticks)
        state.cars[BLUE].ball_touches = touching
        states.append(state)
    counted = [len(names) for names in run([TouchDetector()], states, BLUE)]
    assert counted == expected
    assert run([TouchDetector()], states, ORANGE) == [[] for _ in contact]


# --- shot and save ------------------------------------------------------------------------------


def shot_sequence(toucher: str, on_target_before: bool = False):
    """The ball in blue's attacking half; someone touches it and it heads into the +y goal."""
    start = quiet_state()
    before = (0.0, 1500.0, 0.0) if on_target_before else (1500.0, 300.0, 0.0)
    set_ball(start, (0.0, 3000.0, 93.15), before)
    hit = after(start)
    set_ball(hit, (0.0, 3100.0, 93.15), (0.0, 2500.0, 0.0))
    hit.cars[toucher].ball_touches = 1
    return [start, hit, after(hit)]


def test_a_shot_is_a_touch_that_puts_the_ball_on_target_for_the_goal_attacked():
    detectors = [ShotDetector(), SaveDetector()]
    assert run(detectors, shot_sequence(BLUE)) == [["shot"], []]
    assert run(detectors, shot_sequence(ORANGE)) == [[], []]  # orange attacks the other goal
    assert run(detectors, shot_sequence(BLUE, on_target_before=True)) == [
        [],
        [],
    ]  # already going in
    untouched = shot_sequence(BLUE)
    untouched[1].cars[BLUE].ball_touches = 0
    assert run(detectors, untouched) == [[], []]


def test_a_shot_needs_the_goal_within_the_horizon():
    assert run([ShotDetector(horizon=0.5)], shot_sequence(BLUE)) == [[], []]  # 0.8 s away


def test_a_save_is_a_touch_that_takes_the_ball_off_target_for_the_goal_defended():
    start = quiet_state()
    set_ball(start, (0.0, -3000.0, 300.0), (0.0, -2500.0, 200.0))  # going in at blue's goal
    cleared = after(start)
    set_ball(cleared, (0.0, -3100.0, 300.0), (800.0, 1500.0, 400.0))
    cleared.cars[BLUE].ball_touches = 1
    detectors = [ShotDetector(), SaveDetector()]
    assert run(detectors, [start, cleared, after(cleared)]) == [["save"], []]

    by_orange = after(start)
    set_ball(by_orange, (0.0, -3100.0, 300.0), (800.0, 1500.0, 400.0))
    by_orange.cars[ORANGE].ball_touches = 1
    assert run(detectors, [start, by_orange]) == [[]]  # their own shot, taken off target

    still_in = after(start)
    set_ball(still_in, (0.0, -3100.0, 300.0), (100.0, -2000.0, 100.0))
    still_in.cars[BLUE].ball_touches = 1
    assert run(detectors, [start, still_in]) == [[]]  # touched, and still going in


# --- goal ---------------------------------------------------------------------------------------


def test_a_goal_goes_to_the_last_toucher_of_the_scoring_team_once():
    start = quiet_state()
    touch = after(start)
    touch.cars[BLUE].ball_touches = 1
    deflect = after(touch)
    deflect.cars[ORANGE].ball_touches = 1
    goal = after(deflect)
    set_ball(goal, (0.0, 5300.0, 200.0), (0.0, 1000.0, 0.0))
    goal.goal_scored = True
    tracker = SkillTracker([GoalDetector()])
    tracker.reset(start)
    events = [tracker.step(state) for state in (touch, deflect, goal, after(goal))]
    assert [skills_of(e) for e in events] == [[], [], ["goal"], []]
    assert (events[2][0].agent, events[2][0].team) == (BLUE, 0)


def test_a_goal_nobody_of_the_scoring_team_touched_counts_for_the_team_only():
    start = quiet_state()
    own_goal = after(start)
    own_goal.cars[BLUE].ball_touches = 1
    set_ball(own_goal, (0.0, -5300.0, 200.0), (0.0, -1000.0, 0.0))
    own_goal.goal_scored = True
    tracker = SkillTracker([GoalDetector()])
    tracker.reset(start)
    (event,) = tracker.step(own_goal)
    assert (event.skill, event.agent, event.team) == ("goal", None, 1)
    summary = tracker.summary()
    assert summary.team_counts[1]["goal"] == 1
    assert summary.counts[ORANGE] == {}
    assert tracker.count("goal") == 1
    tracker.reset(start)  # the next episode starts clean: the last toucher is forgotten
    late = after(start)
    set_ball(late, (0.0, 5300.0, 200.0), (0.0, 1000.0, 0.0))
    late.goal_scored = True
    assert tracker.step(late)[0].agent is None


# --- touches in the air -------------------------------------------------------------------------


def air_touch_states(ball_height=600.0, car_in_air=True):
    start = quiet_state()
    set_ball(start, (0.0, 0.0, ball_height), (0.0, 100.0, 10.0))
    car = start.cars[BLUE]
    if car_in_air:
        fly(car, (0.0, -150.0, ball_height - 50.0))
    states = [start]
    for touching in (1, 0, 0, 1, 0, 0, 1):
        state = after(states[-1])
        state.cars[BLUE].ball_touches = touching
        states.append(state)
    return states


def test_the_first_air_touch_is_an_aerial_touch_and_the_later_ones_air_dribble_touches():
    states = air_touch_states()
    assert run([AerialTouchDetector()], states, BLUE) == [
        ["aerial_touch"],
        [],
        [],
        ["air_dribble_touch"],
        [],
        [],
        ["air_dribble_touch"],
    ]


def test_landing_ends_the_stretch_and_wheels_on_the_ball_do_not():
    states = air_touch_states()
    landed = states[3].cars[BLUE]
    set_car(landed, (0.0, -150.0, 17.0), (0, 0, 0), (0.0, 1.0, 0.0), boost=50.0)
    fly(states[4].cars[BLUE], (0.0, -150.0, 550.0))
    states[4].cars[BLUE].ball_touches = 1
    names = run([AerialTouchDetector()], states, BLUE)
    assert names[0] == ["aerial_touch"]
    assert names[3] == ["aerial_touch"]  # a new stretch after the landing
    assert names[6] == ["air_dribble_touch"]

    states = air_touch_states()
    standing = states[3].cars[BLUE]
    set_car(standing, (0.0, 0.0, 700.0), (0, 0, 0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), boost=50.0)
    assert wheels_on_ball(standing, states[3].ball)  # on_ground, but only on the ball
    assert in_the_air(standing, states[3].ball)
    assert run([AerialTouchDetector()], states, BLUE)[3] == ["air_dribble_touch"]
    standing.physics.position[1] = 600.0  # the same car standing on something else
    assert not in_the_air(standing, states[3].ball)


def test_low_balls_and_grounded_cars_make_no_air_touch():
    assert run([AerialTouchDetector()], air_touch_states(ball_height=200.0), BLUE) == [[]] * 7
    assert run([AerialTouchDetector()], air_touch_states(car_in_air=False), BLUE) == [[]] * 7
    high = run([AerialTouchDetector(min_ball_height=100.0)], air_touch_states(200.0), BLUE)
    assert high[0] == ["aerial_touch"]


def test_in_the_air_means_no_wheel_on_anything_but_the_ball():
    state = quiet_state()
    car = state.cars[BLUE]
    set_ball(state, (0.0, 0.0, 800.0), (0.0, 100.0, 10.0))

    fly(car, (0.0, -150.0, 800.0))
    assert in_the_air(car, state.ball)  # no wheel touches anything

    car.wheels_with_contact = (True, True, False, False)
    assert not car.on_ground  # RLGym wants three wheels for that...
    assert not in_the_air(car, state.ball)  # ...but two wheels on something is not flying

    # Riding the side wall, the ball rolling on the same wall beside the car: the ball's centre
    # is further from the wall than the car's, so it is on the roof's side, not under the wheels.
    set_car(car, (4079.0, 0.0, 800.0), (0, 0, 800), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), boost=50.0)
    set_ball(state, (4001.0, 0.0, 950.0), (0.0, 0.0, 800.0))
    assert not in_the_air(car, state.ball)

    # Two front wheels on the ball, which sits ahead of and below the car: off every surface.
    fly(car, (0.0, 0.0, 900.0))
    car.wheels_with_contact = (True, True, False, False)
    set_ball(state, (0.0, 120.0, 800.0), (0.0, 100.0, 10.0))
    assert not wheels_on_ball(car, state.ball)  # not standing on it
    assert in_the_air(car, state.ball)


# --- flip reset ---------------------------------------------------------------------------------


def reset_states(*, jumped=True, flipped=False, height=800.0, ball_gap=100.0, still_on_ball=True):
    """A car in the air above the ball, wheels down; then its jump flags are cleared."""
    start = quiet_state()
    set_ball(start, (0.0, 0.0, height), (0.0, 50.0, 10.0))
    car = start.cars[BLUE]
    fly(car, (0.0, 0.0, height + 150.0), velocity=(0.0, 0.0, -400.0))
    car.has_jumped, car.has_flipped, car.air_time_since_jump = jumped, flipped, 1.6
    contact = after(start)
    car = contact.cars[BLUE]
    fly(car, (0.0, 0.0, height + ball_gap))
    car.has_jumped = car.has_flipped = False
    car.air_time_since_jump = 0.0
    car.on_ground = still_on_ball
    return [start, contact, after(contact)]


def test_a_flip_reset_is_counted_with_the_wheels_still_on_the_ball_or_already_off_it():
    detectors = [FlipResetDetector()]
    assert run(detectors, reset_states(still_on_ball=True), BLUE) == [["flip_reset"], []]
    assert run(detectors, reset_states(still_on_ball=False), BLUE) == [["flip_reset"], []]
    assert run(detectors, reset_states(jumped=False, flipped=True), BLUE) == [["flip_reset"], []]


def test_what_is_not_a_flip_reset():
    detectors = [FlipResetDetector()]
    never_jumped = reset_states(jumped=False)
    assert run(detectors, never_jumped, BLUE) == [[], []]  # it held its dodge all along
    assert run(detectors, reset_states(height=150.0), BLUE) == [[], []]  # too low: the floor
    assert run(detectors, reset_states(ball_gap=400.0), BLUE) == [[], []]  # nowhere near the ball

    flags_kept = reset_states()
    flags_kept[1].cars[BLUE].has_jumped = True
    assert run(detectors, flags_kept, BLUE) == [[], []]  # nothing was reset

    on_a_wall = reset_states()
    car = on_a_wall[1].cars[BLUE]
    set_car(car, (0.0, 150.0, 800.0), (0, 0, 0), (0.0, 0.0, 1.0), (0.0, -1.0, 0.0), boost=50.0)
    assert car.on_ground  # wheels on something beside the ball, pointing away from it
    assert not wheels_on_ball(car, on_a_wall[1].ball)
    assert run(detectors, on_a_wall, BLUE) == [[], []]


def test_flip_reset_thresholds_are_parameters():
    assert run([FlipResetDetector(min_height=100.0)], reset_states(height=150.0), BLUE)[0] == [
        "flip_reset"
    ]
    assert run([FlipResetDetector(reach=90.0)], reset_states(), BLUE) == [[], []]


# --- ceiling ------------------------------------------------------------------------------------


def test_a_ceiling_drop_is_leaving_the_ceiling():
    start = quiet_state()
    on_ceiling = (0.0, 0.0, CEILING_Z - 17.0)
    set_car(
        start.cars[BLUE], on_ceiling, (0, 500, 0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), boost=50.0
    )
    stays = after(start)
    drops = after(stays)
    fly(drops.cars[BLUE], (0.0, 60.0, CEILING_Z - 40.0), up=(0.0, 0.0, -1.0))
    assert run([CeilingDropDetector()], [start, stays, drops, after(drops)], BLUE) == [
        [],
        ["ceiling_drop"],
        [],
    ]

    start = quiet_state()
    high_on_a_wall = (4079.0, 0.0, 1900.0)
    set_car(
        start.cars[BLUE], high_on_a_wall, (0, 0, 0), (0.0, 0.0, 1.0), (-1.0, 0.0, 0.0), boost=50.0
    )
    off = after(start)
    fly(off.cars[BLUE], (4000.0, 0.0, 1900.0))
    assert run([CeilingDropDetector()], [start, off], BLUE) == [[]]  # a wall is not the ceiling


# --- backboard ----------------------------------------------------------------------------------


def backboard_states(toucher=BLUE, wait_steps=4, bounce_height=900.0, wall=1.0):
    """A touch, a wait, then the ball coming off the back wall at ``wall * 5120``."""
    start = quiet_state()
    set_ball(start, (0.0, wall * 3000.0, bounce_height), (0.0, wall * 2000.0, 100.0))
    touch = after(start)
    touch.cars[toucher].ball_touches = 1
    states = [start, touch]
    for _ in range(wait_steps):
        states.append(after(states[-1]))
    bounce = after(states[-1])
    set_ball(
        bounce, (0.0, wall * (BACK_WALL_Y - 150.0), bounce_height), (0.0, -wall * 1200.0, 50.0)
    )
    states.append(bounce)
    return states


def test_a_backboard_bounce_is_the_last_touchers_ball_coming_off_the_wall_it_attacks():
    detectors = [BackboardBounceDetector()]
    assert run(detectors, backboard_states(), BLUE)[-1] == ["backboard_bounce"]
    assert run_all(detectors, backboard_states(), ORANGE) == []
    assert run_all(detectors, backboard_states(toucher=ORANGE)) == []  # its own backboard
    assert run(detectors, backboard_states(toucher=ORANGE, wall=-1.0), ORANGE)[-1] == [
        "backboard_bounce"
    ]
    assert run_all(detectors, backboard_states(bounce_height=400.0)) == []  # below the bar
    assert run_all(detectors, backboard_states(wait_steps=50)) == []  # 3.4 s after the touch
    assert run([BackboardBounceDetector(touch_window=5.0)], backboard_states(wait_steps=50))[
        -1
    ] == ["backboard_bounce"]


# --- flick --------------------------------------------------------------------------------------


def flick_states(carry=True, dodge=True, wait_steps=0):
    """A car with the ball on its roof jumps and dodges into it."""
    start = quiet_state()
    car = start.cars[BLUE]
    set_car(car, (0.0, 0.0, 17.0), (0.0, 800.0, 0.0), (0.0, 1.0, 0.0), boost=50.0)
    set_ball(start, (0.0, 10.0 if carry else 600.0, 150.0), (0.0, 800.0, 0.0))
    carrying = after(start)
    jumped = after(carrying)
    fly(jumped.cars[BLUE], (0.0, 100.0, 60.0), velocity=(0.0, 800.0, 300.0))
    jumped.cars[BLUE].has_jumped = True
    set_ball(jumped, (0.0, 110.0, 200.0), (0.0, 800.0, 350.0))
    states = [start, carrying, jumped]
    for _ in range(wait_steps):
        states.append(after(states[-1]))
    hit = after(states[-1])
    car = hit.cars[BLUE]
    car.has_flipped, car.flip_time = dodge, 0.05
    car.ball_touches = 1
    again = after(hit)
    again.cars[BLUE].ball_touches = 1
    again.cars[BLUE].flip_time = 0.12
    return [*states, hit, again]


def test_a_flick_is_a_dodge_touch_soon_after_carrying_the_ball_and_counts_once():
    detectors = [FlickDetector()]
    assert run(detectors, flick_states(), BLUE) == [[], [], ["flick"], []]
    assert run_all(detectors, flick_states(carry=False), BLUE) == []
    assert run_all(detectors, flick_states(dodge=False), BLUE) == []
    assert run_all(detectors, flick_states(wait_steps=12), BLUE) == []  # 0.9 s after the carry
    patient = [FlickDetector(carry_window=1.5)]
    assert run_all(patient, flick_states(wait_steps=12), BLUE) == ["flick"]


def test_the_roof_is_a_box_above_a_car_that_stands_on_its_wheels():
    state = flick_states()[0]
    car = state.cars[BLUE]
    assert FlickDetector.on_roof(car, state.ball)
    set_ball(state, (0.0, 200.0, 150.0), (0, 0, 1))
    assert not FlickDetector.on_roof(car, state.ball)  # off the front
    set_ball(state, (0.0, 0.0, 300.0), (0, 0, 1))
    assert not FlickDetector.on_roof(car, state.ball)  # too high above it
    set_ball(state, (0.0, 0.0, 150.0), (0, 0, 1))
    car.on_ground = False
    assert not FlickDetector.on_roof(car, state.ball)  # the car is in the air


# --- wavedash -----------------------------------------------------------------------------------


def landing_states(*, dodge_steps_before=1, speed_before=1000.0, speed_after=1500.0, steps=8):
    """A car comes down; it may dodge some steps before its wheels touch."""
    start = quiet_state()
    fly(start.cars[BLUE], (2000.0, -3000.0, 120.0), velocity=(0.0, speed_before, -300.0))
    states = [start, after(start, steps)]
    if dodge_steps_before:
        dodging = after(states[-1], steps)
        car = dodging.cars[BLUE]
        car.has_flipped, car.flip_time = True, 0.01
        car.physics.linear_velocity[1] = speed_after
        states.append(dodging)
        for _ in range(dodge_steps_before - 1):
            states.append(after(states[-1], steps))
    landed = after(states[-1], steps)
    car = landed.cars[BLUE]
    set_car(car, (2000.0, -2800.0, 17.0), (0.0, speed_after, 0.0), (0.0, 1.0, 0.0), boost=50.0)
    return [*states, landed, after(landed, steps)]


def test_a_wavedash_is_a_landing_that_keeps_the_speed_of_a_dodge():
    detectors = [WavedashDetector()]
    assert run_all(detectors, landing_states(), BLUE) == ["wavedash"]
    # The dodge and the landing inside one step: only the speed shows it.
    assert run_all(detectors, landing_states(dodge_steps_before=0), BLUE) == ["wavedash"]
    # Stepped every tick, with the wheels down 20 ticks after the dodge.
    every_tick = landing_states(dodge_steps_before=20, steps=1)
    assert run_all(detectors, every_tick, BLUE) == ["wavedash"]


def test_what_is_not_a_wavedash():
    detectors = [WavedashDetector()]
    plain = landing_states(dodge_steps_before=0, speed_after=900.0)
    assert run_all(detectors, plain, BLUE) == []  # a landing loses speed
    small = landing_states(speed_after=1300.0)
    assert run_all(detectors, small, BLUE) == []  # under 350 uu/s gained
    assert run_all([WavedashDetector(min_gain=250.0)], small, BLUE) == ["wavedash"]
    long_ago = landing_states(dodge_steps_before=6)  # the dodge 0.4 s before the wheels touch
    assert run_all(detectors, long_ago, BLUE) == []
    assert run_all([WavedashDetector(dodge_window=0.5)], long_ago, BLUE) == ["wavedash"]

    onto_the_ball = landing_states()
    landed = onto_the_ball[-2]
    car = landed.cars[BLUE]
    set_car(car, (0.0, 0.0, 800.0), (0.0, 1500.0, 0.0), (0.0, 1.0, 0.0), boost=50.0)
    set_ball(landed, (0.0, 0.0, 700.0), (0.0, 50.0, 10.0))
    assert wheels_on_ball(car, landed.ball)
    assert run_all(detectors, onto_the_ball, BLUE) == []  # that is a flip reset's contact


def test_a_speed_gain_with_no_dodge_seen_counts_only_for_a_clean_landing():
    """Coming down a wall's curve, a bump or the ball can all add speed inside one step."""
    detectors = [WavedashDetector()]

    def landing(change):
        states = landing_states(dodge_steps_before=0)
        change(states[-2])
        return run_all(detectors, states, BLUE)

    def near_a_wall(state):
        state.cars[BLUE].physics.position[0] = 3900.0

    def on_a_slope(state):
        car = state.cars[BLUE]
        set_car(car, (2000.0, -2800.0, 17.0), (0, 1500, 0), (0, 1, 0), (0.6, 0.0, 0.8), boost=50.0)

    def touching_the_ball(state):
        state.cars[BLUE].ball_touches = 1

    def bumping(state):
        state.cars[BLUE].bump_victim_id = ORANGE

    def bumped(state):
        state.cars[ORANGE].bump_victim_id = BLUE

    assert landing(lambda state: None) == ["wavedash"]
    for spoiled in (near_a_wall, on_a_slope, touching_the_ball, bumping, bumped):
        assert landing(spoiled) == [], spoiled.__name__

    # With the dodge seen in the air beforehand, none of that matters.
    seen = landing_states(dodge_steps_before=1)
    near_a_wall(seen[-2])
    touching_the_ball(seen[-2])
    assert run_all(detectors, seen, BLUE) == ["wavedash"]


def test_one_dodge_is_one_wavedash_however_the_car_bounces():
    states = landing_states()
    bounce = after(states[-1])
    fly(bounce.cars[BLUE], (2000.0, -2700.0, 25.0), velocity=(0.0, 1500.0, 50.0))
    down = after(bounce)
    set_car(down.cars[BLUE], (2000.0, -2600.0, 17.0), (0.0, 1500.0, 0.0), (0, 1, 0), boost=50.0)
    assert run_all([WavedashDetector()], [*states, bounce, down], BLUE) == ["wavedash"]


# --- shots that follow something ----------------------------------------------------------------


def then_shot(states, steps_later, airborne=None):
    """Extend a sequence with quiet steps and then a shot by blue."""
    states = list(states)
    for _ in range(steps_later - 1):
        states.append(after(states[-1]))
    before = after(states[-1])
    set_ball(before, (0.0, 3000.0, 400.0), (1500.0, 300.0, 0.0))
    shot = after(before)
    set_ball(shot, (0.0, 3100.0, 400.0), (0.0, 2500.0, 100.0))
    shot.cars[BLUE].ball_touches = 1
    if airborne is not None:
        shot.cars[BLUE].on_ground = not airborne
    return [*states, before, shot]


def test_a_flip_reset_shot_is_a_shot_within_two_seconds_of_the_reset():
    soon = then_shot(reset_states()[:2], steps_later=10)  # 0.7 s later
    assert run(None, soon, BLUE)[-1] == ["touch", "shot", "flip_reset_shot"]
    late = then_shot(reset_states()[:2], steps_later=40)  # 2.7 s later
    assert "flip_reset_shot" not in run_all(None, late, BLUE)
    assert "shot" in run(None, late, BLUE)[-1]


def test_a_double_tap_is_an_air_shot_within_two_seconds_of_a_backboard_bounce():
    in_the_air_shot = then_shot(backboard_states(), steps_later=6, airborne=True)
    assert "double_tap" in run(None, in_the_air_shot, BLUE)[-1]
    from_the_ground = then_shot(backboard_states(), steps_later=6, airborne=False)
    names = run(None, from_the_ground, BLUE)[-1]
    assert "shot" in names
    assert "double_tap" not in names
    assert "double_tap" not in run_all(None, then_shot(backboard_states(), 40, True), BLUE)


def test_a_ceiling_shot_is_a_shot_within_two_seconds_of_leaving_the_ceiling():
    start = quiet_state()
    on_ceiling = (0.0, 0.0, CEILING_Z - 17.0)
    set_car(
        start.cars[BLUE], on_ceiling, (0, 500, 0), (0.0, 1.0, 0.0), (0.0, 0.0, -1.0), boost=50.0
    )
    drops = after(start)
    fly(drops.cars[BLUE], (0.0, 60.0, CEILING_Z - 40.0), up=(0.0, 0.0, -1.0))
    assert "ceiling_shot" in run(None, then_shot([start, drops], steps_later=15), BLUE)[-1]
    assert "ceiling_shot" not in run_all(None, then_shot([start, drops], 40), BLUE)


def test_a_flick_shot_is_a_flick_that_is_a_shot_in_the_same_step():
    states = flick_states()[:4]  # up to and including the dodge touch
    set_ball(states[2], (0.0, 3000.0, 200.0), (1500.0, 300.0, 350.0))  # not on target before...
    set_ball(states[3], (0.0, 3100.0, 260.0), (0.0, 2500.0, 300.0))  # ...on target after
    # (The ball is above 250 uu and the car in the air, so the touch is an aerial touch as well.)
    assert run(None, states, BLUE)[-1] == ["touch", "shot", "aerial_touch", "flick", "flick_shot"]
    not_a_shot = flick_states()[:4]
    assert "flick_shot" not in run_all(None, not_a_shot, BLUE)


def test_shot_after_can_be_built_for_any_pair_of_events():
    detectors = [ShotDetector(), AerialTouchDetector(), ShotAfter("air_shot", "aerial_touch", 0.0)]
    states = air_touch_states()[:1]
    assert run(detectors, then_shot(states, steps_later=1, airborne=True), BLUE)[-1] == [
        "shot",
        "aerial_touch",
        "air_shot",
    ]
