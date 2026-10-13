"""Field helpers, rotations and ball-flight maths."""

from __future__ import annotations

import numpy as np
import pytest
from rlgym.rocket_league.api import PhysicsObject
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BLUE_TEAM,
    CEILING_Z,
    GRAVITY,
    ORANGE_TEAM,
    SIDE_WALL_X,
)

from rocket_drills.geometry import (
    BALL_DRAG,
    CORNER_SUM,
    attacked_goal_y,
    ball_flight,
    clamp_to_field,
    defended_goal_y,
    goal_crossing,
    in_field,
    launch_velocity,
    on_target,
    rotation_matrix,
    team_sign,
    unit,
    xy_in_field,
)


def test_blue_attacks_positive_y_and_orange_negative_y():
    assert team_sign(BLUE_TEAM) == 1.0
    assert team_sign(ORANGE_TEAM) == -1.0
    assert attacked_goal_y(BLUE_TEAM) == BACK_WALL_Y == defended_goal_y(ORANGE_TEAM)
    assert attacked_goal_y(ORANGE_TEAM) == -BACK_WALL_Y == defended_goal_y(BLUE_TEAM)


def test_unit_scales_to_length_one_and_rejects_the_zero_vector():
    assert np.allclose(unit((3.0, 0.0, 4.0)), (0.6, 0.0, 0.8))
    with pytest.raises(ValueError, match="zero vector"):
        unit((0.0, 0.0, 0.0))


def test_rotation_matrix_has_forward_right_and_up_as_columns():
    matrix = rotation_matrix((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    assert matrix.dtype == np.float32
    assert np.allclose(matrix[:, 0], (0.0, 1.0, 0.0))
    assert np.allclose(matrix[:, 1], (-1.0, 0.0, 0.0))  # RLGym's "right" of a car that faces +y
    assert np.allclose(matrix[:, 2], (0.0, 0.0, 1.0))


def test_rotation_matrix_reads_back_through_rlgym_and_matches_its_euler_angles():
    rng = np.random.default_rng(1)
    for _ in range(50):
        forward = unit(rng.normal(size=3))
        up = rng.normal(size=3)
        if abs(float(unit(up) @ forward)) > 0.95:
            continue
        physics = PhysicsObject()
        physics.rotation_mtx = rotation_matrix(forward, up)
        assert np.allclose(physics.forward, forward, atol=1e-6)
        assert abs(float(physics.up @ forward)) < 1e-6
        assert float(physics.up @ up) > 0.0
        assert np.allclose(physics.rotation_mtx.T @ physics.rotation_mtx, np.eye(3), atol=1e-5)
        assert np.linalg.det(physics.rotation_mtx) == pytest.approx(1.0, abs=1e-5)

    yawed = PhysicsObject()
    yawed.euler_angles = np.array([0.0, 0.7, 0.0])  # pitch, yaw, roll
    built = rotation_matrix((np.cos(0.7), np.sin(0.7), 0.0))
    assert np.allclose(yawed.rotation_mtx, built, atol=1e-6)


def test_rotation_matrix_rejects_an_up_along_forward():
    with pytest.raises(ValueError, match="parallel"):
        rotation_matrix((0.0, 0.0, 1.0), (0.0, 0.0, 2.0))


def test_ball_flight_without_drag_is_a_parabola():
    position, velocity = ball_flight((0.0, 0.0, 500.0), (100.0, 200.0, 300.0), 2.0, drag=0.0)
    assert np.allclose(position, (200.0, 400.0, 500.0 + 600.0 - 0.5 * GRAVITY * 4.0))
    assert np.allclose(velocity, (100.0, 200.0, 300.0 - GRAVITY * 2.0))


def test_drag_shortens_the_flight_a_little():
    free, _ = ball_flight((0.0, 0.0, 500.0), (0.0, 2000.0, 0.0), 2.0, drag=0.0)
    dragged, velocity = ball_flight((0.0, 0.0, 500.0), (0.0, 2000.0, 0.0), 2.0)
    assert 0.0 < free[1] - dragged[1] < 0.05 * free[1]
    assert velocity[1] == pytest.approx(2000.0 * 0.97**2)
    assert pytest.approx(0.03046, abs=1e-4) == BALL_DRAG


@pytest.mark.parametrize("drag", [0.0, BALL_DRAG])
def test_launch_velocity_is_the_inverse_of_ball_flight(drag):
    rng = np.random.default_rng(2)
    for _ in range(20):
        start = rng.uniform(-3000.0, 3000.0, 3)
        target = rng.uniform(-3000.0, 3000.0, 3)
        time = rng.uniform(0.3, 2.5)
        arrived, _ = ball_flight(start, launch_velocity(start, target, time, drag), time, drag)
        assert np.allclose(arrived, target, atol=1e-6)
    with pytest.raises(ValueError, match="positive"):
        launch_velocity((0, 0, 0), (1, 1, 1), 0.0)


def test_on_target_wants_the_goal_mouth_within_the_horizon():
    goal = BACK_WALL_Y
    assert on_target((0.0, 3000.0, 93.0), (0.0, 2000.0, 0.0), goal)  # rolling in
    assert not on_target((0.0, 3000.0, 93.0), (0.0, 2000.0, 0.0), -goal)  # the other goal
    assert not on_target((0.0, 3000.0, 93.0), (0.0, 500.0, 0.0), goal)  # too slow: over 2.5 s
    assert on_target((0.0, 3000.0, 93.0), (0.0, 500.0, 0.0), goal, horizon=5.0)
    assert not on_target((0.0, 3000.0, 93.0), (1000.0, 2000.0, 0.0), goal)  # wide of the post
    assert not on_target((0.0, 4000.0, 800.0), (0.0, 3000.0, 300.0), goal)  # over the bar
    assert on_target((0.0, 4000.0, 800.0), (0.0, 1500.0, -100.0), goal)  # dropping under it
    assert not on_target((0.0, 3000.0, 93.0), (0.0, 0.0, 0.0), goal)  # not moving


def test_goal_crossing_gives_time_and_place():
    time, x, z = goal_crossing((100.0, 4120.0, 300.0), (50.0, 1000.0, 0.0), BACK_WALL_Y)
    assert time == pytest.approx(1.0)
    assert x == pytest.approx(150.0)
    assert z == pytest.approx(300.0 - 0.5 * GRAVITY)
    assert goal_crossing((0.0, 0.0, 93.0), (0.0, -1000.0, 0.0), BACK_WALL_Y) is None


def test_field_bounds_include_the_diagonal_corners():
    assert CORNER_SUM == 8064
    assert in_field((0.0, 0.0, 100.0))
    assert not in_field((SIDE_WALL_X + 1.0, 0.0, 100.0))
    assert not in_field((0.0, 0.0, CEILING_Z + 1.0))
    assert not in_field((0.0, 0.0, 50.0), margin=100.0)
    assert xy_in_field(3600.0, 4400.0)
    assert not xy_in_field(3600.0, 4600.0)  # inside both walls, beyond the corner wall
    assert not xy_in_field(4000.0, 0.0, margin=100.0)


def test_clamp_to_field_pulls_a_point_inside_every_wall():
    rng = np.random.default_rng(3)
    for _ in range(200):
        x, y = rng.uniform(-6000.0, 6000.0, 2)
        clamped = clamp_to_field(x, y, margin=200.0)
        assert xy_in_field(*clamped, margin=199.0)
    assert clamp_to_field(100.0, -200.0, margin=200.0) == (100.0, -200.0)
