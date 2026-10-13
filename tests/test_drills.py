"""What every drill writes, checked without the simulator: valid states, symmetry, parameters."""

from __future__ import annotations

import math

import numpy as np
import pytest
from rlgym.rocket_league.common_values import (
    BACK_WALL_Y,
    BALL_MAX_SPEED,
    BLUE_TEAM,
    CAR_MAX_SPEED,
    CEILING_Z,
    DOMINUS,
    GOAL_HEIGHT,
    ORANGE_TEAM,
    SIDE_WALL_X,
)

from helpers import starts, surface_distance, to_ball
from rocket_drills.drills import (
    ALL_DRILLS,
    Aerial,
    Backboard,
    Ceiling,
    Dribble,
    DrillInfo,
    FastSave,
    FlickDefence,
    FlipResetChain,
    Save,
    Takeoff,
    WallToAirChain,
    WavedashLanding,
)
from rocket_drills.drills.takeoff import reachable_height
from rocket_drills.geometry import (
    BALL_ON_ROOF_HEIGHT,
    CAR_REST_HEIGHT,
    ball_flight,
    defended_goal_y,
    in_field,
    on_target,
    team_sign,
    unit,
    xy_in_field,
)
from rocket_drills.placement import empty_state
from rocket_drills.skills import FlickDetector

SIZES = [(1, 1), (2, 2), (3, 3)]


@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: f"{s[0]}v{s[1]}")
@pytest.mark.parametrize("cls", ALL_DRILLS, ids=lambda c: c.__name__)
def test_every_start_is_a_valid_state(cls, sizes):
    drill = cls(seed=1)
    for state, info, _ in starts(drill, 120, sizes):
        assert isinstance(info, DrillInfo)
        assert info.name == cls.__name__
        assert info.stage in cls.STAGES
        assert info.team == state.cars[info.agent].team_num
        first = next(a for a, car in state.cars.items() if car.team_num == info.team)
        assert info.agent == first
        assert state.tick_count == 0
        assert not state.boost_pad_timers.any()

        ball = state.ball
        for vector in (ball.position, ball.linear_velocity, ball.angular_velocity):
            assert vector.shape == (3,)
            assert vector.dtype == np.float32
            assert np.isfinite(vector).all()
        assert in_field(ball.position, margin=90.0)
        assert 1.0 < np.linalg.norm(ball.linear_velocity) < BALL_MAX_SPEED  # never left hanging

        positions = []
        for car in state.cars.values():
            physics = car.physics
            assert np.isfinite(physics.position).all()
            assert in_field(physics.position, margin=15.0)
            assert np.linalg.norm(physics.linear_velocity) <= CAR_MAX_SPEED
            assert not physics.angular_velocity.any()
            matrix = physics.rotation_mtx
            assert np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-5)
            assert np.linalg.det(matrix) == pytest.approx(1.0, abs=1e-5)
            assert 0.0 <= car.boost_amount <= 100.0
            assert not car.is_demoed
            assert not car.has_flipped
            assert not car.has_double_jumped
            assert not car.is_jumping
            from_surface = surface_distance(physics.position)
            if car.on_ground:  # wheels on the floor, a wall or the ceiling
                assert from_surface < 20.0
                assert not car.has_jumped
                assert car.air_time_since_jump == 0.0
            else:
                assert from_surface > 50.0
            if car.has_jumped:
                assert car.air_time_since_jump > 0.0
            assert np.linalg.norm(ball.position - physics.position) > 125.0
            positions.append(physics.position)
        for index, here in enumerate(positions):
            for there in positions[index + 1 :]:
                assert np.linalg.norm(here - there) > 200.0


@pytest.mark.parametrize("sizes", [(1, 1), (3, 3)], ids=lambda s: f"{s[0]}v{s[1]}")
@pytest.mark.parametrize("cls", ALL_DRILLS, ids=lambda c: c.__name__)
def test_an_orange_start_is_the_blue_start_turned_half_a_turn(cls, sizes):
    blue_drill, orange_drill = cls(team=BLUE_TEAM, seed=11), cls(team=ORANGE_TEAM, seed=11)
    flip = np.array([-1.0, -1.0, 1.0], dtype=np.float32)
    pairs = zip(starts(blue_drill, 25, sizes), starts(orange_drill, 25, sizes), strict=True)
    for (blue, blue_info, _), (orange, orange_info, _) in pairs:
        assert (blue_info.team, orange_info.team) == (BLUE_TEAM, ORANGE_TEAM)
        assert blue_info.stage == orange_info.stage
        assert np.allclose(orange.ball.position, blue.ball.position * flip)
        assert np.allclose(orange.ball.linear_velocity, blue.ball.linear_velocity * flip)
        for index in range(sizes[0]):
            for ours, theirs in (
                (f"blue-{index}", f"orange-{index}"),
                (f"orange-{index}", f"blue-{index}"),
            ):
                a, b = blue.cars[ours], orange.cars[theirs]
                assert np.allclose(b.physics.position, a.physics.position * flip)
                assert np.allclose(b.physics.linear_velocity, a.physics.linear_velocity * flip)
                assert np.allclose(b.physics.forward, a.physics.forward * flip, atol=1e-6)
                assert np.allclose(b.physics.up, a.physics.up * flip, atol=1e-6)
                assert b.boost_amount == a.boost_amount
                assert (b.on_ground, b.has_jumped) == (a.on_ground, a.has_jumped)
                assert b.air_time_since_jump == a.air_time_since_jump


@pytest.mark.parametrize("cls", ALL_DRILLS, ids=lambda c: c.__name__)
def test_a_seed_fixes_the_starts(cls):
    def positions(drill):
        return [
            np.concatenate([s.ball.position, *(c.physics.position for c in s.cars.values())])
            for s, _, _ in starts(drill, 5, (2, 2))
        ]

    first, again, other = positions(cls(seed=7)), positions(cls(seed=7)), positions(cls(seed=8))
    assert all(np.array_equal(a, b) for a, b in zip(first, again, strict=True))
    assert not all(np.array_equal(a, b) for a, b in zip(first, other, strict=True))
    drill = cls(seed=7)
    positions(drill)
    drill.reseed(7)
    assert all(np.array_equal(a, b) for a, b in zip(first, positions(drill), strict=True))


@pytest.mark.parametrize("cls", ALL_DRILLS, ids=lambda c: c.__name__)
def test_the_team_is_the_one_asked_for_or_either(cls):
    for team in (BLUE_TEAM, ORANGE_TEAM):
        assert {info.team for _, info, _ in starts(cls(team=team, seed=2), 12)} == {team}
    assert {info.team for _, info, _ in starts(cls(seed=2), 60)} == {BLUE_TEAM, ORANGE_TEAM}


STAGED = [cls for cls in ALL_DRILLS if len(cls.STAGES) > 1]


def test_drills_have_named_stages_or_a_single_default_one():
    assert [cls.__name__ for cls in STAGED] == [
        "Takeoff",
        "FlipResetChain",
        "WallToAirChain",
        "Backboard",
        "Save",
        "FastSave",
    ]
    for cls in ALL_DRILLS:
        if cls not in STAGED:
            assert dict(cls.STAGES) == {"default": 1.0}
            assert {info.stage for _, info, _ in starts(cls(seed=3), 5)} == {"default"}


@pytest.mark.parametrize("cls", STAGED, ids=lambda c: c.__name__)
def test_stages_can_be_picked_singly_by_weight_or_left_at_their_defaults(cls):
    for name in cls.STAGES:
        assert {info.stage for _, info, _ in starts(cls(stages=name, seed=3), 10)} == {name}
    default = cls(seed=3)
    assert default.stages == pytest.approx(
        {name: weight / sum(cls.STAGES.values()) for name, weight in cls.STAGES.items()}
    )
    drawn = [info.stage for _, info, _ in starts(default, 500)]
    for name, probability in default.stages.items():
        assert drawn.count(name) / len(drawn) == pytest.approx(probability, abs=0.07)
    names = list(cls.STAGES)
    assert cls(stages=names).stages == pytest.approx(dict.fromkeys(names, 1.0 / len(names)))
    weighted = cls(stages={names[0]: 3.0, names[-1]: 1.0})
    assert weighted.stages[names[0]] == pytest.approx(0.75)
    assert weighted.stages[names[-1]] == pytest.approx(0.25)
    assert {info.stage for _, info, _ in starts(weighted, 60)} == {names[0], names[-1]}


def test_bad_arguments_are_refused_with_a_reason():
    with pytest.raises(ValueError, match="unknown stage"):
        FlipResetChain(stages="landing")
    with pytest.raises(ValueError, match="positive weight"):
        FlipResetChain(stages={"finish": 0.0})
    with pytest.raises(ValueError, match="negative"):
        FlipResetChain(stages={"finish": -1.0, "reset": 2.0})
    with pytest.raises(ValueError, match="unknown wall"):
        WallToAirChain(walls="ceiling")
    with pytest.raises(ValueError, match="team must be"):
        Takeoff(team=2)
    with pytest.raises(ValueError, match="opponents must be"):
        Takeoff(opponents="behind")
    with pytest.raises(ValueError, match="teammates must be"):
        Takeoff(teammates="nowhere")
    with pytest.raises(ValueError, match="low <= high"):
        Takeoff(arrival_time=(1.5, 0.8))
    with pytest.raises(ValueError, match="height must"):
        Takeoff(height=(100.0, 500.0))
    with pytest.raises(ValueError, match="height must stay"):
        Takeoff(height=(300.0, 1500.0))
    with pytest.raises(ValueError, match="between 0 and 1"):
        FlipResetChain(spent_dodge_share=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
        Save(airborne_share=-0.1)


def test_a_drill_needs_cars_and_says_which():
    with pytest.raises(ValueError, match="found no cars"):
        Takeoff().apply(empty_state(0, 0), {})
    with pytest.raises(ValueError, match="has no cars"):
        Takeoff(team=ORANGE_TEAM).apply(empty_state(1, 0), {})
    with pytest.raises(ValueError, match="each team"):
        FlickDefence().apply(empty_state(1, 0), {})


@pytest.mark.parametrize(
    "cls", [c for c in ALL_DRILLS if c is not FlickDefence], ids=lambda c: c.__name__
)
def test_a_lone_car_can_drill(cls):
    for state, info, car in starts(cls(seed=4), 20, (0, 1)):
        assert list(state.cars) == ["orange-0"]
        assert info.team == ORANGE_TEAM
        assert np.isfinite(car.physics.position).all()


def test_layouts_put_the_other_cars_where_they_say():
    parked = Takeoff(seed=5, opponents="away", teammates="away")
    for state, info, _ in starts(parked, 40, (3, 3)):
        sign = team_sign(info.team)
        for agent, car in state.cars.items():
            if agent == info.agent:
                continue
            side = 1.0 if car.team_num != info.team else -1.0
            assert sign * car.physics.position[1] == pytest.approx(side * 4600.0)
            assert not car.physics.linear_velocity.any()

    behind = defending = 0
    for state, info, me in starts(Takeoff(seed=5), 60, (3, 3)):
        sign = team_sign(info.team)
        for car in state.cars.values():
            if car is me:
                continue
            if car.team_num == info.team:
                behind += sign * (car.physics.position[1] - me.physics.position[1]) < 0.0
            else:
                defending += sign * car.physics.position[1] > 400.0
    assert behind == 120  # both team-mates, every start
    assert defending == 180  # all three opponents in their own half, every start

    everywhere = [
        team_sign(info.team) * car.physics.position[1]
        for state, info, me in starts(Aerial(seed=5), 60, (3, 3))
        for car in state.cars.values()
        if car is not me
    ]
    assert min(everywhere) < -3000.0
    assert max(everywhere) > 3000.0


def test_cars_take_their_height_from_their_hitbox():
    for state, _, car in starts(Dribble(seed=6), 5, (1, 1), hitbox=DOMINUS):
        assert car.physics.position[2] == pytest.approx(CAR_REST_HEIGHT[DOMINUS])
        assert state.ball.position[2] == pytest.approx(BALL_ON_ROOF_HEIGHT[DOMINUS])
    for _, _, car in starts(Ceiling(seed=6), 5, (1, 1), hitbox=DOMINUS):
        assert car.physics.position[2] == pytest.approx(CEILING_Z - CAR_REST_HEIGHT[DOMINUS])


# --- the drills one by one -----------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["attacking", "defending"])
def test_takeoff_puts_a_reachable_ball_ahead_of_a_grounded_car(stage):
    ahead = 0
    for state, info, car in starts(Takeoff(stages=stage, seed=3), 200, (2, 2)):
        sign = team_sign(info.team)
        target, time = info.details["target"], info.details["target_time"]
        assert car.on_ground
        assert 45.0 <= car.boost_amount <= 100.0
        assert 500.0 <= np.linalg.norm(car.physics.linear_velocity) <= 1700.0
        assert 0.8 <= time <= 1.5
        assert 300.0 <= target[2] <= reachable_height(time) + 1e-3
        assert state.ball.position[2] > 200.0  # the ball starts in the air
        arrives, _ = ball_flight(state.ball.position, state.ball.linear_velocity, time)
        assert np.allclose(arrives, target, atol=0.05)
        own_half = sign * car.physics.position[1] < -1100.0
        assert own_half if stage == "defending" else sign * car.physics.position[1] > -900.0
        ahead += sign * (target[1] - car.physics.position[1]) > 200.0
        for other in state.cars.values():
            if other.team_num != info.team:
                assert sign * other.physics.position[1] > 400.0  # the other team is at home
    assert ahead / 200 > 0.97


def test_takeoff_parameters_narrow_the_start():
    drill = Takeoff(seed=3, arrival_time=(1.0, 1.1), height=(400.0, 500.0), boost=(80.0, 90.0))
    for _, info, car in starts(drill, 50):
        assert 1.0 <= info.details["target_time"] <= 1.1
        assert 400.0 <= info.details["target"][2] <= 500.0
        assert 80.0 <= car.boost_amount <= 90.0
    for _, info, _ in starts(Takeoff(seed=3, upper_heights=True), 100):
        top = reachable_height(info.details["target_time"])
        assert 0.7 * top - 1e-3 <= info.details["target"][2] <= top + 1e-3


def test_flip_reset_chain_stages_are_what_they_claim():
    spent = []
    for stage in ("approach", "reset", "finish"):
        for state, info, car in starts(FlipResetChain(stages=stage, seed=5), 200):
            sign = team_sign(info.team)
            direction, distance = to_ball(car, state)
            aim = unit(info.details["aim"] - state.ball.position)
            assert sign * state.ball.position[1] > 2299.0  # the ball hangs in their third
            assert 450.0 <= state.ball.position[2] <= 1000.0
            assert sign * info.details["aim"][1] == pytest.approx(BACK_WALL_Y)
            assert not car.on_ground
            closing = float((car.physics.linear_velocity - state.ball.linear_velocity) @ direction)
            if stage == "finish":
                assert 169.0 <= distance <= 221.0
                assert float(car.physics.forward @ aim) > 0.999  # nose along the line to the goal
                assert float(direction @ aim) > 0.999  # and the car behind the ball on that line
                assert not car.has_jumped
                assert car.can_flip
                assert 49.0 <= closing <= 351.0
            elif stage == "reset":
                assert 174.0 <= distance <= 261.0
                assert float(-car.physics.up @ direction) > 0.999  # wheels at the ball
                assert car.has_jumped
                assert 299.0 <= closing <= 651.0
                spent.append(not car.has_flip)
            else:
                assert 400.0 <= distance <= 950.0
                assert float(car.physics.forward @ direction) > 0.999
                assert car.has_jumped
                assert 0.3 <= car.air_time_since_jump <= 0.9
                assert car.has_flip
    assert np.mean(spent) == pytest.approx(0.75, abs=0.08)
    assert all(car.on_ground for _, _, car in starts(FlipResetChain(stages="takeoff", seed=5), 30))


def test_flip_reset_chain_parameters():
    always = FlipResetChain(stages="reset", spent_dodge_share=1.0, seed=5)
    assert not any(car.has_flip for _, _, car in starts(always, 40))
    never = FlipResetChain(stages="reset", spent_dodge_share=0.0, seed=5)
    assert all(car.has_flip for _, _, car in starts(never, 40))
    high = FlipResetChain(stages="finish", ball_height=(900.0, 950.0), seed=5)
    assert all(900.0 <= state.ball.position[2] <= 950.0 for state, _, _ in starts(high, 40))

    def keeper_wide(share: float) -> float:
        drill = FlipResetChain(stages="finish", keeper_out_share=share, seed=5)
        wide = [
            abs(
                state.cars[f"{'orange' if info.team == BLUE_TEAM else 'blue'}-0"].physics.position[
                    0
                ]
            )
            >= 2000.0
            for state, info, _ in starts(drill, 100, (2, 2))
        ]
        return float(np.mean(wide))

    assert keeper_wide(1.0) == 1.0
    assert keeper_wide(0.0) < 0.35  # (a keeper moved off the ball's spot can end up wide too)


@pytest.mark.parametrize("sizes", [(1, 1), (2, 2)], ids=lambda s: f"{s[0]}v{s[1]}")
def test_wall_to_air_stages_are_what_they_claim(sizes):
    def wall_distance(position):
        return min(SIDE_WALL_X - abs(position[0]), BACK_WALL_Y - abs(position[1]))

    for stage in ("setup", "launch"):
        for state, _, car in starts(WallToAirChain(stages=stage, seed=2), 150, sizes):
            assert car.on_ground  # wheels on the wall
            assert wall_distance(car.physics.position) == pytest.approx(17.03, abs=0.01)
            assert abs(car.physics.up[2]) < 1e-6  # the roof points off the wall
            assert car.physics.forward[2] > 0.0 or stage == "launch"
            assert wall_distance(state.ball.position) < (100.0 if stage == "setup" else 560.0)
            assert state.ball.position[2] > car.physics.position[2]

    for state, _, car in starts(WallToAirChain(stages="approach", seed=2), 150, sizes):
        assert car.on_ground
        assert car.physics.position[2] < 20.0
        ball, velocity = state.ball.position, state.ball.linear_velocity
        on_floor, on_wall = ball[2] < 100.0, wall_distance(ball) < 140.0
        assert on_floor or on_wall
        if not on_floor:
            assert velocity[2] > 250.0  # rolling up the wall
        heading = ball + 0.4 * velocity - car.physics.position
        assert float(car.physics.forward[:2] @ unit(heading[:2])) > 0.999

    for state, _, car in starts(WallToAirChain(stages="finish", seed=2), 150, sizes):
        direction, distance = to_ball(car, state)
        assert not car.on_ground
        assert 259.0 <= distance <= 501.0
        assert float(car.physics.forward @ direction) > 0.999
        assert state.ball.position[2] > 890.0


def test_wall_to_air_uses_every_wall_and_only_the_ones_asked_for():
    def kinds(drill):
        return [info.details["wall"] for _, info, _ in starts(drill, 300)]

    default = kinds(WallToAirChain(stages="setup", seed=2))
    assert default.count("side") / 300 == pytest.approx(0.55, abs=0.09)
    assert default.count("their_back") / 300 == pytest.approx(0.25, abs=0.08)
    assert default.count("own_back") / 300 == pytest.approx(0.20, abs=0.08)
    assert set(kinds(WallToAirChain(stages="setup", walls="their_back", seed=2))) == {"their_back"}

    low_on_a_back_wall = [
        abs(state.ball.position[0])
        for state, info, _ in starts(WallToAirChain(stages="setup", walls="own_back", seed=2), 100)
    ]
    assert min(low_on_a_back_wall) >= 1150.0  # beside the goal, not in front of it
    side = [
        state.ball.position
        for state, _, _ in starts(WallToAirChain(stages="setup", walls="side", seed=2), 200)
    ]
    assert np.ptp([p[1] for p in side]) > 5000.0  # spread along the wall
    assert {np.sign(p[0]) for p in side} == {-1.0, 1.0}  # both of them


def test_wall_to_air_challenger_boost_and_jump_shares():
    for stage in ("launch", "finish"):
        challenged = low = jumped = 0
        for state, info, car in starts(WallToAirChain(stages=stage, seed=4), 400, (2, 2)):
            other = "orange" if info.team == BLUE_TEAM else "blue"
            keeper, challenger = state.cars[f"{other}-0"], state.cars[f"{other}-1"]
            assert keeper.on_ground  # the first defender stays down
            if not challenger.on_ground:
                challenged += 1
                assert challenger.has_jumped
                assert challenger.physics.position[2] >= 300.0
                assert np.linalg.norm(challenger.physics.position - car.physics.position) > 300.0
            low += car.boost_amount <= 20.0
            jumped += car.has_jumped
        assert challenged / 400 == pytest.approx(0.35, abs=0.08)
        assert low / 400 == pytest.approx(0.25, abs=0.08)
        if stage == "finish":
            assert jumped / 400 == pytest.approx(0.5, abs=0.08)
    quiet = WallToAirChain(
        stages="finish", challenger_share=0.0, low_boost_share=0.0, jumped_share=1.0, seed=4
    )
    for state, _, car in starts(quiet, 60):
        assert all(c.on_ground for c in state.cars.values() if c is not car)
        assert car.boost_amount >= 40.0
        assert car.has_jumped


@pytest.mark.parametrize("cls", [Save, FastSave], ids=lambda c: c.__name__)
@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: f"{s[0]}v{s[1]}")
def test_save_drills_put_a_shot_on_target_at_the_defenders_net(cls, sizes):
    fast = cls is FastSave
    lifted = low_boost = 0
    for state, info, defender in starts(cls(seed=3), 300, sizes):
        sign = team_sign(info.team)
        ball, velocity = state.ball.position, state.ball.linear_velocity
        assert on_target(ball, velocity, defended_goal_y(info.team))
        assert not on_target(ball, velocity, -defended_goal_y(info.team))
        assert sign * ball[1] < -799.0  # in the defender's half
        speed = float(np.linalg.norm(velocity[:2]))
        assert (2499.0 if fast else 1399.0) <= speed <= (3950.0 if fast else 3300.0)
        target, time = info.details["target"], info.details["target_time"]
        assert sign * target[1] == pytest.approx(-BACK_WALL_Y)
        assert abs(target[0]) <= 750.0
        assert target[2] < GOAL_HEIGHT - 91.25
        assert 0.2 < time <= 2.2 + 1e-9
        is_lifted = ball[2] > 150.0 or velocity[2] > 100.0
        lifted += is_lifted
        if is_lifted:
            arrives, _ = ball_flight(ball, velocity, time)
            assert np.allclose(arrives, target, atol=0.1)
        assert defender.on_ground
        assert sign * defender.physics.position[1] < 0.0  # on its own side
        if info.stage == "in_goal":
            assert sign * defender.physics.position[1] < -4299.0
            assert abs(defender.physics.position[0]) <= 800.0
        low_boost += defender.boost_amount < 15.0
        other = "orange" if info.team == BLUE_TEAM else "blue"
        shooter = state.cars[f"{other}-0"]
        assert sign * shooter.physics.position[1] > sign * ball[1]  # behind the ball...
        assert sign * shooter.physics.linear_velocity[1] < 0.0  # ...following it in
    assert lifted / 300 == pytest.approx(0.75 if fast else 0.4, abs=0.09)
    assert low_boost / 300 == pytest.approx(0.5, abs=0.09) if fast else low_boost == 0


def test_save_parameters():
    slow = Save(seed=3, speed=(1000.0, 1100.0), airborne_share=1.0, boost=(0.0, 5.0))
    for state, info, defender in starts(slow, 60):
        assert state.ball.position[2] >= 200.0
        assert defender.boost_amount <= 5.0
        assert info.details["target_time"] <= 2.2 + 1e-9
    assert all(
        defender.boost_amount <= 15.0
        for _, _, defender in starts(FastSave(seed=3, low_boost_share=1.0), 60)
    )
    alone = Save(seed=3, shooter=False, opponents="away")
    for state, info, _ in starts(alone, 20):
        other = "orange" if info.team == BLUE_TEAM else "blue"
        assert abs(state.cars[f"{other}-0"].physics.position[1]) == pytest.approx(4600.0)


@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: f"{s[0]}v{s[1]}")
def test_dribble_starts_with_the_ball_riding_on_the_roof(sizes):
    toward_goal = 0
    for state, info, car in starts(Dribble(seed=3), 200, sizes):
        assert car.on_ground
        assert FlickDetector.on_roof(car, state.ball)
        assert np.allclose(state.ball.linear_velocity, car.physics.linear_velocity)
        assert 200.0 <= np.linalg.norm(car.physics.linear_velocity) <= 1300.0
        toward_goal += team_sign(info.team) * car.physics.linear_velocity[1] > 0.0
        for other in state.cars.values():
            if other is not car:
                assert np.linalg.norm((other.physics.position - car.physics.position)[:2]) >= 799.0
    assert toward_goal / 200 > 0.9


@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: f"{s[0]}v{s[1]}")
def test_flick_defence_puts_the_defender_in_front_of_a_carried_ball(sizes):
    faces_ball = 0
    for state, info, defender in starts(FlickDefence(seed=4), 300, sizes):
        sign = team_sign(info.team)
        carrier = state.cars[info.details["carrier"]]
        assert carrier.team_num != info.team
        assert FlickDetector.on_roof(carrier, state.ball)
        assert np.allclose(state.ball.linear_velocity, carrier.physics.linear_velocity)
        assert 699.0 <= np.linalg.norm(carrier.physics.linear_velocity) <= 1501.0
        assert sign * carrier.physics.linear_velocity[1] < 0.0  # carried at the defender's net
        gap = np.linalg.norm((defender.physics.position - carrier.physics.position)[:2])
        assert 600.0 < gap < 2500.0
        assert (
            sign * (defender.physics.position[1] - carrier.physics.position[1]) < 0.0
        )  # goal-side
        assert sign * defender.physics.linear_velocity[1] < 0.0  # backing off toward its net
        direction, _ = to_ball(defender, state)
        faces_ball += float(defender.physics.forward[:2] @ unit(direction[:2])) > 0.9
    assert faces_ball / 300 == pytest.approx(0.5, abs=0.1)


def test_aerial_puts_a_high_ball_near_a_grounded_car_that_faces_it():
    facing = 0
    for state, _, car in starts(Aerial(seed=3), 200):
        offset = (state.ball.position - car.physics.position)[:2]
        assert car.on_ground
        assert 50.0 <= car.boost_amount <= 100.0
        assert 450.0 <= state.ball.position[2] <= 1500.0
        assert np.linalg.norm(offset) <= 1600.0 + 1e-3
        facing += float(car.physics.forward[:2] @ unit(offset)) > 0.8
    assert facing / 200 > 0.9
    narrow = Aerial(seed=3, height=(1000.0, 1100.0), distance=(500.0, 600.0), boost=(10.0, 20.0))
    for state, _, car in starts(narrow, 50):
        assert 1000.0 <= state.ball.position[2] <= 1100.0
        assert car.boost_amount <= 20.0


def test_ceiling_starts_on_the_ceiling_with_the_ball_below_and_ahead():
    for state, info, car in starts(Ceiling(seed=3), 200):
        sign = team_sign(info.team)
        assert car.on_ground
        assert car.physics.position[2] == pytest.approx(CEILING_Z - 17.03)
        assert np.allclose(car.physics.up, (0.0, 0.0, -1.0), atol=1e-6)
        assert sign * car.physics.forward[1] > 0.6  # heading for the other team's half
        assert not car.has_jumped
        assert state.ball.position[2] < car.physics.position[2] - 390.0
        assert sign * (state.ball.position[1] - car.physics.position[1]) >= 400.0


@pytest.mark.parametrize("stage", ["airborne", "grounded"])
def test_backboard_sends_the_ball_at_a_point_above_their_goal(stage):
    for state, info, car in starts(Backboard(stages=stage, seed=3), 200):
        sign = team_sign(info.team)
        target, time = info.details["target"], info.details["target_time"]
        assert sign * target[1] == pytest.approx(BACK_WALL_Y - 94.75)
        assert 750.0 <= target[2] <= 1500.0
        assert target[2] - 91.25 > GOAL_HEIGHT  # the whole ball above the crossbar
        assert sign * state.ball.linear_velocity[1] > 600.0
        arrives, _ = ball_flight(state.ball.position, state.ball.linear_velocity, time)
        assert np.allclose(arrives, target, atol=0.1)
        assert sign * (state.ball.position[1] - car.physics.position[1]) >= 499.0  # it follows
        assert car.on_ground == (stage == "grounded")
        assert car.has_jumped == (stage == "airborne")
        if stage == "airborne":
            assert 0.3 <= car.air_time_since_jump <= 0.9
    low = Backboard(seed=3, impact_height=(800.0, 850.0))
    assert all(800.0 <= info.details["target"][2] <= 850.0 for _, info, _ in starts(low, 30))


@pytest.mark.parametrize("sizes", SIZES, ids=lambda s: f"{s[0]}v{s[1]}")
def test_wavedash_landing_is_a_fair_race_between_two_cars_coming_down(sizes):
    jumped, nose_up, fair = [], [], []
    for state, _, _ in starts(WavedashLanding(seed=6), 200, sizes):
        ball = state.ball.position
        assert ball[2] == pytest.approx(93.15)
        assert np.linalg.norm(state.ball.linear_velocity) < 900.0
        racers = [state.cars["blue-0"], state.cars["orange-0"]]
        distances = []
        for side, car in zip((-1.0, 1.0), racers, strict=True):  # blue from -y, orange from +y
            position, velocity = car.physics.position, car.physics.linear_velocity
            assert not car.on_ground
            assert 59.0 <= position[2] - 17.03 <= 351.0
            assert velocity[2] <= -99.0  # coming down
            assert car.boost_amount <= 20.0
            assert side * (position[1] - ball[1]) >= -1.0  # each on its own side of the ball
            assert float(unit(velocity[:2]) @ unit((ball - position)[:2])) > 0.7  # heading at it
            assert abs(car.physics.forward[2]) <= math.sin(0.45) + 1e-6
            assert car.can_flip
            jumped.append(car.has_jumped)
            nose_up.append(car.physics.forward[2] > math.sin(0.1))
            distances.append(float(np.linalg.norm((ball - position)[:2])))
        inside = all(xy_in_field(*c.physics.position[:2], margin=310.0) for c in racers)
        if inside:  # a wall did not cut either run short
            fair.append(abs(distances[0] / distances[1] - 1.0) < 0.12)
            assert min(distances) > 1400.0
            assert max(distances) < 4250.0
    assert np.mean(jumped) == pytest.approx(0.5, abs=0.08)
    assert np.mean(nose_up) == pytest.approx(0.5, abs=0.08)
    assert len(fair) > 100
    assert all(fair)
