"""The drills in the simulator: each start is physically what it claims to be.

These step RocketSim and are skipped when it cannot be started (see ``conftest.py``). The shares
are measured by ``physical.py`` over a modest number of starts; the thresholds leave room below
the figures quoted in ``docs/drills.md``, which come from larger runs.
"""

from __future__ import annotations

import numpy as np
import pytest
from rlgym.rocket_league.common_values import GRAVITY

import physical
from helpers import begin, play
from rocket_drills import SkillTracker
from rocket_drills.drills import ALL_DRILLS, Dribble, FlickDefence, FlipResetChain, WallToAirChain
from rocket_drills.placement import empty_state, mark_jumped, set_ball, set_car

pytestmark = pytest.mark.sim

STARTS = 60


@pytest.mark.parametrize("sizes", [(1, 1), (3, 3)], ids=lambda s: f"{s[0]}v{s[1]}")
@pytest.mark.parametrize("cls", ALL_DRILLS, ids=lambda c: c.__name__)
def test_the_simulator_agrees_with_what_every_drill_wrote(engine, cls, sizes):
    drill = cls(seed=1)
    for _ in range(25):
        start, _, shared = begin(engine, drill, sizes)
        placed = {agent: (car.on_ground, car.has_jumped) for agent, car in start.cars.items()}
        resting = cls in (Dribble, FlickDefence)  # the ball starts in contact with a roof
        rolling = start.ball.position[2] < 100.0 or cls is WallToAirChain  # ...or with a surface
        (tick,) = play(engine, start, shared, {}, 1 / 120, tick_skip=1)
        for agent, car in tick.cars.items():
            assert np.isfinite(car.physics.position).all()
            # One tick on, wheels that were put on a surface are on it and the others are not.
            assert (car.on_ground, car.has_jumped) == placed[agent]
        if not (resting or rolling):  # free flight: only gravity and drag acted on the ball
            change = tick.ball.linear_velocity - start.ball.linear_velocity
            change[2] += GRAVITY / 120
            assert np.linalg.norm(change) < 3.0
        *_, later = play(engine, tick, shared, {}, 0.5)
        assert np.isfinite(later.ball.position).all()
        assert all(np.isfinite(car.physics.position).all() for car in later.cars.values())


def test_a_placed_car_keeps_its_dodge_and_a_car_marked_as_jumped_loses_it_on_time(engine):
    def dodges(air_time: float | None, wait: float) -> bool:
        state = engine.create_base_state()
        state.cars = empty_state(1, 0).cars
        set_ball(state, (0.0, 3000.0, 93.15), (0.0, 0.0, 0.0))
        car = state.cars["blue-0"]
        set_car(car, (0.0, -1000.0, 1900.0), (0, 0, 0), (0.0, 1.0, 0.0), boost=0.0, on_ground=False)
        if air_time is not None:
            mark_jumped(car, air_time)
        state = engine.set_state(state, {})
        placed = state.cars["blue-0"]
        assert placed.has_jumped == (air_time is not None)
        waited = np.zeros((round(wait * 120), 8), dtype=np.float32)
        if len(waited):
            engine.step({"blue-0": waited}, {})
        press = np.zeros((2, 8), dtype=np.float32)
        press[:, 2], press[:, 5] = -1.0, 1.0  # stick forward, jump
        return bool(engine.step({"blue-0": press}, {}).cars["blue-0"].has_flipped)

    assert dodges(None, wait=0.0)
    assert dodges(None, wait=1.4)  # never jumped: the dodge is there for as long as it stays up
    assert dodges(0.5, wait=0.0)
    assert dodges(1.2, wait=0.0)
    assert not dodges(1.3, wait=0.0)  # past the 1.25 s window
    assert not dodges(0.5, wait=1.0)  # the window keeps running


@pytest.mark.parametrize("stage", ["attacking", "defending"])
def test_takeoff_ball_arrives_and_a_boosted_jump_reaches_it(engine, stage):
    shares = physical.takeoff(engine, stage, STARTS, seed=3)
    assert shares["ball_within_40uu_of_target_at_target_time"] == 1.0
    assert shares["largest_distance_from_target"] < 15.0
    assert shares["jump_within_130uu_of_target_height_in_time"] >= 0.85
    assert shares["jump_at_target_height_in_time"] >= 0.6
    assert shares["jump_touches_ball"] >= 0.1  # without steering: the jump alone gets some


@pytest.mark.parametrize("tick_skip", [8, 1])
def test_flip_reset_finish_is_one_forward_dodge_from_a_shot(engine, tick_skip):
    shares = physical.flip_reset_finish(engine, STARTS, seed=5, tick_skip=tick_skip)
    assert shares["starts_holding_a_dodge"] == 1.0
    assert shares["dodge_adds_300_toward_goal"] >= 0.9
    assert shares["dodge_is_counted_as_a_shot"] >= 0.8
    assert shares["goal_within_3s"] >= 0.6


@pytest.mark.parametrize("tick_skip", [8, 1])
def test_flip_reset_stage_gives_a_counted_reset_to_a_car_that_coasts(engine, tick_skip):
    shares = physical.flip_reset_contact(engine, STARTS, seed=5, tick_skip=tick_skip)
    assert shares["starts_in_the_air_having_jumped"] == 1.0
    assert shares["starts_with_the_dodge_spent"] == pytest.approx(0.75, abs=0.15)
    assert shares["flip_reset_counted"] >= 0.9
    assert shares["flip_reset_counted_exactly_once"] >= 0.9
    assert shares["holds_a_dodge_after_the_reset"] >= 0.95


def test_flip_reset_needs_the_jump_the_drill_gives_the_car(engine):
    """The same starts with the jump taken away: wheels on the ball give nothing to count."""

    def never_jumped(state, info):
        car = state.cars[info.agent]
        car.has_jumped, car.air_time_since_jump = False, 0.0

    drill = FlipResetChain(stages="reset", seed=5)
    counted = wheels_on_ball = 0
    for _ in range(40):
        start, info, shared = begin(engine, drill, prepare=never_jumped)
        assert start.cars[info.agent].can_flip  # it holds a dodge it never earned
        tracker = SkillTracker()
        tracker.reset(start)
        touched = False
        for frame in play(engine, start, shared, {}, 1.0):
            counted += any(e.skill == "flip_reset" for e in tracker.step(frame))
            touched |= frame.cars[info.agent].on_ground
        wheels_on_ball += touched
    assert wheels_on_ball >= 36  # the contact happens all the same
    assert counted == 0


@pytest.mark.parametrize(
    ("stage", "where"),
    [
        ("approach", "on_the_floor_after_8_ticks"),
        ("setup", "on_a_wall_after_8_ticks"),
        ("launch", "on_a_wall_after_8_ticks"),
        ("finish", "in_the_air_after_8_ticks"),
    ],
)
def test_wall_to_air_stages_start_where_they_say(engine, stage, where):
    assert physical.wall_to_air(engine, stage, STARTS, seed=2)[where] >= 0.98


@pytest.mark.parametrize("fast", [False, True], ids=["Save", "FastSave"])
def test_save_shots_go_in_when_nobody_stops_them(engine, fast):
    shares = physical.save(engine, fast, STARTS, seed=7)
    assert shares["on_target_at_the_start"] == 1.0
    assert shares["goes_in_untouched_within_5s"] >= 0.97
    assert shares["lifted_shots_cross_within_30uu_of_target"] >= 0.95
    assert shares["lifted_shots_cross_within_0.03s_of_target_time"] >= 0.9


def test_dribble_ball_stays_on_a_car_that_keeps_under_it(engine):
    shares = physical.dribble(engine, STARTS, seed=3)
    assert shares["carry_ball_on_roof_after_0.5s"] >= 0.95
    assert shares["carry_ball_on_roof_after_1.0s"] >= 0.9
    assert shares["carry_ball_on_roof_after_2.0s"] >= 0.6
    assert shares["coasting_ball_on_roof_after_0.5s"] >= 0.7
    assert shares["coasting_ball_on_roof_after_1.0s"] <= 0.1  # a car that coasts loses it


def test_ceiling_car_drops_by_itself_and_keeps_its_dodge(engine):
    shares = physical.ceiling(engine, STARTS, seed=3)
    assert shares["wheels_on_the_ceiling_after_8_ticks"] >= 0.95
    assert shares["off_the_ceiling_within_0.6s"] == 1.0
    assert shares["ceiling_drop_counted"] >= 0.95
    assert shares["holds_a_dodge_in_the_air_after_0.6s"] >= 0.95


@pytest.mark.parametrize("stage", ["airborne", "grounded"])
def test_backboard_ball_comes_off_the_wall_above_the_goal(engine, stage):
    shares = physical.backboard(engine, stage, STARTS, seed=3)
    assert shares["bounces_off_their_backboard_above_the_goal"] >= 0.98
    assert shares["passes_within_30uu_of_target"] >= 0.98
    assert shares["bounces_within_0.05s_of_target_time"] >= 0.95


@pytest.mark.parametrize("tick_skip", [8, 1])
def test_a_scripted_wavedash_from_the_landing_drill_is_counted(engine, tick_skip):
    shares = physical.wavedash(engine, STARTS, seed=5, tick_skip=tick_skip)
    assert shares["script_gains_350_by_the_landing"] >= 0.9
    assert shares["wavedash_counted"] >= 0.9
    assert shares["wavedash_counted_for_the_idle_car"] == 0.0
