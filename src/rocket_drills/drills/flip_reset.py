"""The flip-reset chain: take-off, approach, wheels to the ball, and the finish."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from rlgym.rocket_league.common_values import BACK_WALL_Y, DOUBLEJUMP_MAX_DELAY

from rocket_drills.drills.base import Drill, Range, Seed, Stages, check_range, check_share
from rocket_drills.drills.scene import OpponentLayout, Scene, TeammateLayout
from rocket_drills.drills.takeoff import MAX_HEIGHT, MIN_HEIGHT, place_takeoff
from rocket_drills.geometry import UP, unit


class FlipResetChain(Drill):
    """Every stage of a flip reset as a start of its own.

    A flip reset is a long play: get up to a high ball, get the wheels onto it, come off it with
    the dodge back, and use that dodge to shoot. Practised from the ground only, the last links
    are almost never reached. This drill starts episodes at each link, so the last one can be
    learned first and the earlier ones lead into states that are already worth something.

    In every stage but the take-off the ball hangs in the other team's third, drifting slowly
    toward a point in their goal mouth, and the drill car is on the far side of it from that goal.

    Stages:
        ``"takeoff"`` (default 10%): on the ground under a high ball ahead: the
        :class:`~rocket_drills.drills.Takeoff` start, attacking, at the upper heights.

        ``"approach"`` (default 20%): in the air 450 to 900 uu behind and a little under the
        ball, nose at it, closing at 500 to 1000 uu/s, with a jump 0.3 to 0.9 s behind it.

        ``"reset"`` (default 30%): wheels toward the ball and the goal, 175 to 260 uu from the
        ball's centre and closing at 300 to 650 uu/s, so that the wheels meet the ball within a
        few ticks. The car *has jumped*; in ``spent_dodge_share`` of these starts its dodge has
        run out, so the contact is what gives it the dodge that the finish needs.

        ``"finish"`` (default 40%): nose on the ball along the line to the goal, 170 to 220 uu
        behind the ball's centre, closing slowly, holding a dodge (the state of a car that has
        just had a reset). One forward dodge is a shot. In ``keeper_out_share`` of these starts
        the other team's first car is out wide instead of in its net.

    Args:
        stages: See :class:`Drill`.
        ball_height: Range of the height at which the ball hangs (all stages but the take-off).
        spent_dodge_share: Share of the ``"reset"`` starts in which the car jumped 1.3 to 2.5 s
            ago and has no dodge left; in the rest it jumped 0.5 to 1.2 s ago and still has one.
        keeper_out_share: Share of the ``"finish"`` starts with the other team's first car out
            wide. Only applies with ``opponents="defending"``.
        team: See :class:`Drill`.
        opponents: See :class:`Drill`.
        teammates: See :class:`Drill`.
        seed: See :class:`Drill`.
    """

    STAGES: ClassVar[Mapping[str, float]] = {
        "takeoff": 0.1,
        "approach": 0.2,
        "reset": 0.3,
        "finish": 0.4,
    }

    def __init__(
        self,
        *,
        stages: Stages = None,
        ball_height: Range = (450.0, 1000.0),
        spent_dodge_share: float = 0.75,
        keeper_out_share: float = 0.5,
        team: int | None = None,
        opponents: OpponentLayout = "defending",
        teammates: TeammateLayout = "behind",
        seed: Seed = None,
    ) -> None:
        super().__init__(
            stages=stages, team=team, opponents=opponents, teammates=teammates, seed=seed
        )
        self.ball_height = check_range("ball_height", ball_height, lowest=350.0)
        self.spent_dodge_share = check_share("spent_dodge_share", spent_dodge_share)
        self.keeper_out_share = check_share("keeper_out_share", keeper_out_share)

    def _build(self, scene: Scene, stage: str) -> Mapping[str, Any]:
        rng = scene.rng
        if stage == "takeoff":
            spot, tau = place_takeoff(
                scene,
                attacking=True,
                arrival_time=(0.8, 1.5),
                height=(MIN_HEIGHT, MAX_HEIGHT),
                upper_heights=True,
                boost=(45.0, 100.0),
            )
            return {"target": scene.to_world(spot), "target_time": tau}

        ball = np.array(
            [
                rng.uniform(-1400.0, 1400.0),
                rng.uniform(2300.0, 4200.0),
                rng.uniform(*self.ball_height),
            ]
        )
        aim = np.array([rng.uniform(-500.0, 500.0), BACK_WALL_Y, rng.uniform(150.0, 500.0)])
        line = unit(aim - ball)  # from the ball to the point in the goal mouth
        ball_velocity = line * rng.uniform(0.0, 300.0) + UP * rng.uniform(-100.0, 150.0)
        roof = unit(UP - line * line[2])  # "up" for a car whose nose lies along the line
        scene.ball(ball, ball_velocity)
        scene.keep_clear(ball[0], ball[1], 1500.0)

        me = scene.me
        if stage == "finish":
            position = ball - line * rng.uniform(170.0, 220.0)
            velocity = ball_velocity + line * rng.uniform(50.0, 350.0)
            scene.air_car(me, position, velocity, line, roof, rng.uniform(20.0, 100.0))
        elif stage == "reset":
            position = ball - line * rng.uniform(175.0, 260.0)
            velocity = ball_velocity + line * rng.uniform(300.0, 650.0)
            spent = rng.random() < self.spent_dodge_share
            late = rng.uniform(DOUBLEJUMP_MAX_DELAY + 0.05, 2.5)
            early = rng.uniform(0.5, DOUBLEJUMP_MAX_DELAY - 0.05)
            scene.air_car(  # nose up, roof away from the ball: the wheels lead
                me,
                position,
                velocity,
                roof,
                -line,
                rng.uniform(20.0, 100.0),
                air_time=late if spent else early,
            )
        else:  # approach
            position = ball - line * rng.uniform(450.0, 900.0) - UP * rng.uniform(0.0, 250.0)
            position[2] = max(position[2], 250.0)
            toward = unit(ball - position)
            velocity = ball_velocity + toward * rng.uniform(500.0, 1000.0)
            scene.air_car(
                me,
                position,
                velocity,
                toward,
                unit(UP - toward * toward[2]),
                rng.uniform(40.0, 100.0),
                air_time=rng.uniform(0.3, 0.9),
            )

        keeper_out = rng.random() < self.keeper_out_share
        if stage == "finish" and keeper_out and scene.opponents and self.opponents == "defending":
            scene.fill(self.opponents, self.teammates)
            for _ in range(8):
                wide_x = rng.choice([-1.0, 1.0]) * rng.uniform(2000.0, 3200.0)
                wide_y = rng.uniform(2500.0, 4600.0)
                if scene.is_free(wide_x, wide_y):
                    break
            boost = rng.uniform(20.0, 100.0)
            scene.ground_car(scene.opponents[0], wide_x, wide_y, -np.pi / 2.0, 0.0, boost)
        return {"aim": scene.to_world(aim)}
