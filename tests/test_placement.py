"""Writing the ball and cars into a state, and giving an airborne car a jump in its past."""

from __future__ import annotations

import numpy as np
import pytest
from rlgym.rocket_league.common_values import (
    BLUE_TEAM,
    BOOST_LOCATIONS,
    DOMINUS,
    DOUBLEJUMP_MAX_DELAY,
    ORANGE_TEAM,
)

from rocket_drills.placement import empty_state, mark_jumped, set_ball, set_car


def test_empty_state_has_the_cars_rlgym_would_create():
    state = empty_state(2, 3, hitbox=DOMINUS)
    assert list(state.cars) == ["blue-0", "blue-1", "orange-0", "orange-1", "orange-2"]
    assert [car.team_num for car in state.cars.values()] == [BLUE_TEAM] * 2 + [ORANGE_TEAM] * 3
    assert all(car.hitbox_type == DOMINUS for car in state.cars.values())
    assert state.boost_pad_timers.shape == (len(BOOST_LOCATIONS),)
    assert state.tick_count == 0
    assert not state.goal_scored
    assert state.config.gravity == 1.0


def test_set_ball_copies_three_vectors():
    state = empty_state(1, 0)
    position = np.array([1.0, 2.0, 300.0])
    set_ball(state, position, (10.0, 0.0, -5.0))
    position[0] = 99.0
    assert state.ball.position.tolist() == [1.0, 2.0, 300.0]
    assert state.ball.linear_velocity.tolist() == [10.0, 0.0, -5.0]
    assert state.ball.angular_velocity.tolist() == [0.0, 0.0, 0.0]
    assert state.ball.position.dtype == np.float32
    with pytest.raises(ValueError, match="three numbers"):
        set_ball(state, (1.0, 2.0), (0.0, 0.0, 0.0))


def test_set_car_places_it_and_clears_what_it_remembers():
    car = empty_state(1, 0).cars["blue-0"]
    car.has_jumped = car.has_flipped = car.has_double_jumped = car.is_jumping = True
    car.air_time_since_jump, car.flip_time, car.demo_respawn_timer = 1.0, 0.3, 2.0
    car.ball_touches = 4
    set_car(car, (100.0, 200.0, 17.0), (0.0, 500.0, 0.0), (0.0, 1.0, 0.0), boost=140.0)
    assert car.physics.position.tolist() == [100.0, 200.0, 17.0]
    assert np.allclose(car.physics.forward, (0.0, 1.0, 0.0))
    assert np.allclose(car.physics.up, (0.0, 0.0, 1.0))
    assert car.boost_amount == 100.0  # clipped
    assert car.on_ground
    assert not car.is_demoed
    assert car.ball_touches == 0
    assert not car.has_jumped
    assert not car.has_flipped
    assert not car.has_double_jumped
    assert not car.is_jumping
    assert car.air_time_since_jump == 0.0
    assert car.flip_time == 0.0


def test_a_car_placed_in_the_air_has_never_jumped_and_keeps_its_dodge():
    car = empty_state(1, 0).cars["blue-0"]
    set_car(car, (0.0, 0.0, 900.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), boost=50.0, on_ground=False)
    assert not car.on_ground
    assert not car.has_jumped
    assert car.has_flip
    assert car.can_flip


@pytest.mark.parametrize(
    ("air_time", "dodge_left"),
    [(0.0, True), (0.5, True), (1.2, True), (DOUBLEJUMP_MAX_DELAY, False), (2.0, False)],
)
def test_mark_jumped_leaves_a_dodge_only_inside_the_window(air_time, dodge_left):
    car = empty_state(1, 0).cars["blue-0"]
    set_car(car, (0.0, 0.0, 900.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), boost=50.0, on_ground=False)
    mark_jumped(car, air_time)
    assert car.has_jumped
    assert not car.is_jumping
    assert not car.is_holding_jump
    assert car.air_time_since_jump == air_time
    assert not car.has_flipped
    assert not car.has_double_jumped
    assert car.has_flip is dodge_left
    assert car.can_flip is dodge_left


def test_mark_jumped_refuses_a_car_on_the_ground_and_a_negative_time():
    car = empty_state(1, 0).cars["blue-0"]
    set_car(car, (0.0, 0.0, 17.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0), boost=50.0)
    with pytest.raises(ValueError, match="on_ground=False"):
        mark_jumped(car, 0.5)
    car.on_ground = False
    with pytest.raises(ValueError, match="negative"):
        mark_jumped(car, -0.1)
