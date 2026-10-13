"""The skill detectors: one small class per mechanic.

A detector looks at two consecutive game states (a :class:`~rocket_drills.skills.Step`) and emits
an event for every car that did its mechanic between them. Each docstring states exactly what
counts. Times are in seconds, so the definitions hold whatever the number of ticks per step;
``docs/skills.md`` lists what each detector is known to miss or to count wrongly.
"""

from __future__ import annotations

from collections.abc import Hashable

import numpy as np
from rlgym.rocket_league.api import Car, GameState, PhysicsObject
from rlgym.rocket_league.common_values import BACK_WALL_Y

from rocket_drills.geometry import (
    ON_TARGET_HORIZON,
    attacked_goal_y,
    defended_goal_y,
    on_target,
    team_sign,
    xy_in_field,
)
from rocket_drills.skills.events import Step

_NEVER = float("-inf")
_BOUNCE_DEPTH = 520.0  # a ball coming off a back wall is seen no further from it than this
_BOUNCE_HEIGHT = 650.0  # ...and with its centre above this: just over the crossbar


def wheels_on_ball(
    car: Car, ball: PhysicsObject, reach: float = 230.0, alignment: float = 0.7
) -> bool:
    """Tell whether the car is standing on the ball.

    True when at least three wheels have contact, the ball's centre is within ``reach`` of the
    car's and the direction of the wheels (straight down from the roof) is within
    ``acos(alignment)`` of the direction to the ball.
    """
    return car.on_ground and _ball_under_wheels(car, ball, reach, alignment)


def in_the_air(car: Car, ball: PhysicsObject) -> bool:
    """Tell whether the car is off every surface.

    True when none of its wheels has contact, or when the ball is what they touch: the ball's
    centre is within 230 uu and on the wheels' side of the car (a ball that rests on the surface a
    car drives on is always on its roof's side, since its centre is further from that surface).
    A car with one or two wheels on the floor or on a wall is not in the air, although RLGym's
    ``Car.on_ground`` is false for it.
    """
    return not any(car.wheels_with_contact) or _ball_under_wheels(car, ball, alignment=0.2)


def _ball_under_wheels(
    car: Car, ball: PhysicsObject, reach: float = 230.0, alignment: float = 0.7
) -> bool:
    to_ball = np.asarray(ball.position, dtype=np.float64) - car.physics.position
    distance = float(np.linalg.norm(to_ball))
    return distance < reach and float(-(car.physics.up @ to_ball)) > alignment * distance


def _ground_speed(car: Car) -> float:
    return float(np.linalg.norm(car.physics.linear_velocity[:2]))


def _used_a_jump(car: Car) -> bool:
    return bool(car.has_jumped or car.has_flipped or car.has_double_jumped)


class Detector:
    """Base class of the skill detectors.

    Attributes:
        skills: Names of the events the detector can emit.
    """

    skills: tuple[str, ...] = ()

    def reset(self, state: GameState) -> None:
        """Forget everything. Called at the start of every episode with its first state."""

    def update(self, step: Step) -> None:
        """Look at one step and emit the events in it."""
        raise NotImplementedError


class TouchDetector(Detector):
    """``touch``: the car begins a touch of the ball.

    A touch begins when the car is in contact with the ball after at least ``touch_gap`` seconds
    (a setting of the tracker, 0.05 s by default) without contact. A hit that lasts several ticks
    is one touch; so is a ball carried on the roof, until it has been off the car for that long.
    """

    skills = ("touch",)

    def update(self, step: Step) -> None:
        """Emit ``touch`` for every car that began a touch."""
        for agent in step.state.cars:
            if step.new_touch(agent):
                step.emit("touch", agent)


class ShotDetector(Detector):
    """``shot``: the car's touch puts the ball on target for the goal its team attacks.

    Counted in a step in which the car is in contact with the ball, the ball is on target after
    the step and was not on target before it. On target means that the ball's flight under gravity
    alone (no drag, bounces or walls) crosses the goal line within ``horizon`` seconds, between
    the posts and under the crossbar, each narrowed by the ball's radius
    (:func:`rocket_drills.geometry.on_target`).

    Args:
        horizon: How many seconds ahead the flight is followed.
    """

    skills = ("shot",)

    def __init__(self, horizon: float = ON_TARGET_HORIZON) -> None:
        self.horizon = horizon

    def update(self, step: Step) -> None:
        """Emit ``shot`` for every car whose touch put the ball on target."""
        before, after = step.prev.ball, step.state.ball
        for agent, car in step.state.cars.items():
            if not step.touching(agent):
                continue
            goal_y = attacked_goal_y(car.team_num)
            was = on_target(before.position, before.linear_velocity, goal_y, self.horizon)
            now = on_target(after.position, after.linear_velocity, goal_y, self.horizon)
            if now and not was:
                step.emit("shot", agent)


class SaveDetector(Detector):
    """``save``: the car's touch takes the ball off target for the goal its team defends.

    Counted in a step in which the car is in contact with the ball, the ball was on target for the
    car's own goal before the step and is not after it. On target is defined as for
    :class:`ShotDetector`.

    Args:
        horizon: How many seconds ahead the flight is followed.
    """

    skills = ("save",)

    def __init__(self, horizon: float = ON_TARGET_HORIZON) -> None:
        self.horizon = horizon

    def update(self, step: Step) -> None:
        """Emit ``save`` for every car whose touch took the ball off target."""
        before, after = step.prev.ball, step.state.ball
        for agent, car in step.state.cars.items():
            if not step.touching(agent):
                continue
            goal_y = defended_goal_y(car.team_num)
            was = on_target(before.position, before.linear_velocity, goal_y, self.horizon)
            now = on_target(after.position, after.linear_velocity, goal_y, self.horizon)
            if was and not now:
                step.emit("save", agent)


class GoalDetector(Detector):
    """``goal``: a goal is scored, credited to the scoring team's car that touched the ball last.

    Counted in the first step in which ``GameState.goal_scored`` is true. If no car of the scoring
    team has touched the ball since the reset (an own goal off a kickoff, say) the event has no
    car and counts for the team only.
    """

    skills = ("goal",)

    def __init__(self) -> None:
        self._last_touch: dict[int, Hashable] = {}

    def reset(self, state: GameState) -> None:
        """Forget who touched the ball."""
        self._last_touch = {}

    def update(self, step: Step) -> None:
        """Emit ``goal`` when the ball crosses a goal line."""
        for agent, car in step.state.cars.items():
            if step.touching(agent):
                self._last_touch[car.team_num] = agent
        if step.state.goal_scored and not step.prev.goal_scored:
            team = step.state.scoring_team
            scorer = self._last_touch.get(team)
            if scorer is None:
                step.emit_for_team("goal", team)
            else:
                step.emit("goal", scorer)


class AerialTouchDetector(Detector):
    """``aerial_touch`` and ``air_dribble_touch``: touches of a ball in the air by a car in the air.

    An air touch is a touch that begins (see :class:`TouchDetector`) while none of the car's wheels
    has contact with anything but the ball (:func:`in_the_air`) and the centre of the ball is
    higher than ``min_ball_height``. The first air touch after the car leaves a surface is an
    ``aerial_touch``; every later one before it lands again is an ``air_dribble_touch``.

    Args:
        min_ball_height: Height the ball's centre must be above.
    """

    skills = ("aerial_touch", "air_dribble_touch")

    def __init__(self, min_ball_height: float = 250.0) -> None:
        self.min_ball_height = min_ball_height
        self._streak: dict[Hashable, int] = {}

    def reset(self, state: GameState) -> None:
        """Forget the touches of the airborne stretches in progress."""
        self._streak = {}

    def update(self, step: Step) -> None:
        """Emit the air touches that began in this step."""
        ball = step.state.ball
        for agent, car in step.state.cars.items():
            if not in_the_air(car, ball):
                self._streak[agent] = 0
                continue
            if step.new_touch(agent) and ball.position[2] > self.min_ball_height:
                count = self._streak.get(agent, 0) + 1
                self._streak[agent] = count
                step.emit("aerial_touch" if count == 1 else "air_dribble_touch", agent)


class FlipResetDetector(Detector):
    """``flip_reset``: the car gets its jump and dodge back off the ball, in the air.

    Counted in a step when all of these hold:

    * before the step the car had no wheel contact and had used a jump since it last stood on
      something (``has_jumped``, ``has_flipped`` or ``has_double_jumped``);
    * after the step all three of those flags are clear, which the simulator does when three or
      more wheels touch something;
    * after the step the ball's centre is within ``reach`` of the car's, and the car is either in
      the air again or still has its wheels on the ball (pointing within ``acos(alignment)`` of
      the direction to the ball);
    * the car is higher than ``min_height``.

    The second form of the third condition matters: the wheels usually stay on the ball for
    several ticks, and a detector that waits for the car to be airborne again misses most resets
    when a step ends during the contact.

    Args:
        min_height: Height the car's origin must be above, which rules out the floor.
        reach: Largest distance between the centres of the car and the ball.
        alignment: Smallest cosine between the wheels' direction and the direction to the ball
            while the wheels are still on it.
    """

    skills = ("flip_reset",)

    def __init__(
        self, min_height: float = 300.0, reach: float = 230.0, alignment: float = 0.7
    ) -> None:
        self.min_height = min_height
        self.reach = reach
        self.alignment = alignment

    def update(self, step: Step) -> None:
        """Emit ``flip_reset`` for every car that got one in this step."""
        ball = step.state.ball
        for agent, before, car in step.cars():
            if before.on_ground or not _used_a_jump(before) or _used_a_jump(car):
                continue
            if car.physics.position[2] <= self.min_height:
                continue
            distance = float(np.linalg.norm(ball.position - car.physics.position))
            if distance >= self.reach:
                continue
            if not car.on_ground or wheels_on_ball(car, ball, self.reach, self.alignment):
                step.emit("flip_reset", agent)


class CeilingDropDetector(Detector):
    """``ceiling_drop``: the car comes off the ceiling.

    Counted in a step before which the car had its wheels on the ceiling (wheel contact, roof
    pointing down with ``up.z < -0.7``, origin above ``min_height``) and after which it has no
    wheel contact. Whether it fell or jumped off is not looked at.

    Args:
        min_height: Height the car's origin must be above to be on the ceiling.
    """

    skills = ("ceiling_drop",)

    def __init__(self, min_height: float = 1800.0) -> None:
        self.min_height = min_height

    def update(self, step: Step) -> None:
        """Emit ``ceiling_drop`` for every car that left the ceiling in this step."""
        for agent, before, car in step.cars():
            on_ceiling = (
                before.on_ground
                and before.physics.up[2] < -0.7
                and before.physics.position[2] > self.min_height
            )
            if on_ceiling and not car.on_ground:
                step.emit("ceiling_drop", agent)


class BackboardBounceDetector(Detector):
    """``backboard_bounce``: the ball comes off the backboard the car attacks, after its touch.

    Counted for the car that touched the ball last, in the step in which the ball bounces off the
    back wall its team attacks, above the goal, if that touch was no more than ``touch_window``
    seconds earlier. The bounce is seen in the ball's velocity: toward that wall at more than 300
    uu/s before the step and away from it at more than 100 uu/s after it, with the ball within 520
    uu of the wall and its centre above 650 uu.

    This is the first half of a double tap and nothing more: any hard touch that sends the ball
    onto the wall above the goal counts, whether or not the car goes on to read the rebound.

    Args:
        touch_window: Longest time between the car's touch and the bounce.
    """

    skills = ("backboard_bounce",)

    def __init__(self, touch_window: float = 3.0) -> None:
        self.touch_window = touch_window

    def update(self, step: Step) -> None:
        """Emit ``backboard_bounce`` when the last toucher's ball comes off the backboard."""
        agent = step.last_toucher
        if agent is None or agent not in step.state.cars:
            return
        sign = team_sign(step.state.cars[agent].team_num)
        before, after = step.prev.ball, step.state.ball
        bounced = (
            sign * before.linear_velocity[1] > 300.0
            and sign * after.linear_velocity[1] < -100.0
            and sign * after.position[1] > BACK_WALL_Y - _BOUNCE_DEPTH
            and after.position[2] > _BOUNCE_HEIGHT
        )
        touched = step.last_contact(agent)
        if bounced and touched is not None and step.time - touched <= self.touch_window:
            step.emit("backboard_bounce", agent)


class FlickDetector(Detector):
    """``flick``: the car dodges into a ball it was carrying on its roof.

    Counted once per carry, in a step in which the car is in contact with the ball while dodging
    (``is_flipping``, or ``has_flipped`` newly set), if the ball rode on its roof no more than
    ``carry_window`` seconds before. The ball rides on the roof when the car has wheel contact and
    the ball's centre is 100 to 230 uu above the car's origin along the car's up axis and less
    than 110 uu from that axis.

    Args:
        carry_window: Longest time between the ball riding on the roof and the dodge touch.
    """

    skills = ("flick",)

    def __init__(self, carry_window: float = 0.7) -> None:
        self.carry_window = carry_window
        self._carried: dict[Hashable, float] = {}

    def reset(self, state: GameState) -> None:
        """Forget the carries in progress."""
        self._carried = {}

    @staticmethod
    def on_roof(car: Car, ball: PhysicsObject) -> bool:
        """Tell whether the ball rides on the roof of a car that stands on its wheels."""
        if not car.on_ground:
            return False
        offset = np.asarray(ball.position, dtype=np.float64) - car.physics.position
        height = float(offset @ car.physics.up)
        sideways = float(np.linalg.norm(offset - height * car.physics.up))
        return 100.0 < height < 230.0 and sideways < 110.0

    def update(self, step: Step) -> None:
        """Emit ``flick`` for every car whose dodge hit the ball it carried."""
        for agent, before, car in step.cars():
            if self.on_roof(before, step.prev.ball):
                self._carried[agent] = step.time - step.dt
            dodging = car.is_flipping or (car.has_flipped and not before.has_flipped)
            carried = self._carried.get(agent, _NEVER)
            if step.touching(agent) and dodging and step.time - carried <= self.carry_window:
                step.emit("flick", agent)
                self._carried[agent] = _NEVER


class WavedashDetector(Detector):
    """``wavedash``: the car lands with a sudden gain in speed along the ground.

    Counted in a step before which the car had no wheel contact and after which it has (and is not
    standing on the ball), if its horizontal speed is at least ``min_gain`` higher than before the
    dodge. A plain landing loses speed; the dodge of a wavedash adds about 500 uu/s.

    The dodge is seen in one of two ways. If ``has_flipped`` became set while the car was in the
    air, no more than ``dodge_window`` seconds before the landing, the gain is measured from the
    speed just before that. Otherwise the dodge, if there was one, began and ended inside the
    landing step and left no flag behind; the gain is then measured over that step, and counts
    only for a clean landing: on the flat floor at least 300 uu from every wall, with no touch of
    the ball and no bump in that step. (Coming down the curve at the foot of a wall, or being
    bumped, also adds horizontal speed.)

    Args:
        min_gain: Smallest gain in horizontal speed, in uu/s.
        dodge_window: Longest time between the start of the dodge and the landing.
    """

    skills = ("wavedash",)

    def __init__(self, min_gain: float = 350.0, dodge_window: float = 0.3) -> None:
        self.min_gain = min_gain
        self.dodge_window = dodge_window
        self._dodge: dict[Hashable, tuple[float, float]] = {}

    def reset(self, state: GameState) -> None:
        """Forget the dodges in progress."""
        self._dodge = {}

    @staticmethod
    def _clean_landing(state: GameState, agent: Hashable) -> bool:
        car = state.cars[agent]
        position = car.physics.position
        flat = (
            position[2] < 50.0
            and car.physics.up[2] > 0.9
            and xy_in_field(position[0], position[1], margin=300.0)
        )
        bumped = car.bump_victim_id is not None or any(
            other.bump_victim_id == agent for other in state.cars.values()
        )
        return flat and not bumped and car.ball_touches == 0

    def update(self, step: Step) -> None:
        """Emit ``wavedash`` for every car that landed with the speed of a dodge."""
        for agent, before, car in step.cars():
            speed = _ground_speed(car)
            landed = not before.on_ground and car.on_ground
            if landed and not wheels_on_ball(car, step.state.ball):
                started, speed_then = self._dodge.get(agent, (_NEVER, 0.0))
                if step.time - started <= self.dodge_window + 1e-9:
                    gain = speed - speed_then
                elif self._clean_landing(step.state, agent):
                    gain = speed - _ground_speed(before)
                else:
                    gain = 0.0
                if gain >= self.min_gain:
                    step.emit("wavedash", agent)
                    self._dodge.pop(agent, None)  # one dodge is one wavedash, however it bounces
            if car.has_flipped and not before.has_flipped and not car.on_ground:
                self._dodge[agent] = (step.time, _ground_speed(before))


class ShotAfter(Detector):
    """A shot that follows another event of the same car within a time limit.

    This is how the mechanics that are only worth something when they end in a shot are counted:
    ``flip_reset_shot``, ``ceiling_shot``, ``double_tap`` and ``flick_shot`` are all instances.
    The event is emitted in a step in which the car has a ``shot`` (so a :class:`ShotDetector`
    must run earlier in the same tracker), if the car's latest ``trigger`` event is no more than
    ``within`` seconds old. ``within=0`` asks for the trigger in the same step.

    Args:
        skill: Name of the event to emit.
        trigger: Name of the event that must come first.
        within: Longest time between the trigger and the shot.
        airborne: Also require the car to have no wheel contact after the step.
    """

    def __init__(self, skill: str, trigger: str, within: float, *, airborne: bool = False) -> None:
        self.skills = (skill,)
        self.trigger = trigger
        self.within = within
        self.airborne = airborne

    def update(self, step: Step) -> None:
        """Emit the event for every car whose shot follows its trigger closely enough."""
        for agent, car in step.state.cars.items():
            if not step.happened("shot", agent) or (self.airborne and car.on_ground):
                continue
            triggered = step.last_time(self.trigger, agent)
            if triggered is not None and step.time - triggered <= self.within + 1e-9:
                step.emit(self.skills[0], agent)


def default_detectors() -> list[Detector]:
    """Return a fresh set of every detector, in an order that satisfies their dependencies.

    Besides the plain detectors this includes four :class:`ShotAfter` instances:

    * ``flip_reset_shot``: a shot within 2 s of the car's flip reset;
    * ``ceiling_shot``: a shot within 2 s of the car coming off the ceiling;
    * ``double_tap``: a shot by a car in the air within 2 s of its backboard bounce;
    * ``flick_shot``: a shot in the same step as the car's flick or within 0.5 s after it.
    """
    return [
        TouchDetector(),
        ShotDetector(),
        SaveDetector(),
        GoalDetector(),
        AerialTouchDetector(),
        FlipResetDetector(),
        CeilingDropDetector(),
        BackboardBounceDetector(),
        FlickDetector(),
        WavedashDetector(),
        ShotAfter("flip_reset_shot", "flip_reset", 2.0),
        ShotAfter("ceiling_shot", "ceiling_drop", 2.0),
        ShotAfter("double_tap", "backboard_bounce", 2.0, airborne=True),
        ShotAfter("flick_shot", "flick", 0.5),
    ]
