# Skill counters

`SkillTracker` watches consecutive `GameState` objects and counts mechanics. This page gives, for
each counter, the exact conditions, the thresholds, what was verified, and what it is known to
miss or to count wrongly.

- [How the tracker works](#how-the-tracker-works)
- [What was verified, and how](#what-was-verified-and-how)
- [touch](#touch) · [shot](#shot) · [save](#save) · [goal](#goal) ·
  [aerial_touch and air_dribble_touch](#aerial_touch-and-air_dribble_touch) ·
  [flip_reset](#flip_reset) · [ceiling_drop](#ceiling_drop) ·
  [backboard_bounce](#backboard_bounce) · [flick](#flick) · [wavedash](#wavedash) ·
  [the shots that follow something](#the-shots-that-follow-something)
- [Counters of your own](#counters-of-your-own)
- [SkillReward](#skillreward)

## How the tracker works

```python
tracker = SkillTracker()
tracker.reset(state)  # first state of an episode
events = tracker.step(next_state)  # every state after it
```

`step` compares the new state with the previous one, runs each detector on the pair and returns
the `SkillEvent`s of that step. Totals build up across episodes until `clear()`. `summary()` gives
counts and car-minutes per car, and events per car-minute per car and per team (a team's events
divided by the minutes of all its cars together, so that team sizes can be compared).

**Time.** Every threshold is in seconds. The time between two states is taken from their
`tick_count` at 120 ticks per second, or from the `dt` you pass to `step`. The tracker can be
stepped every tick or every few ticks; what changes with the step length is said per counter.

**Touches.** RLGym reports `Car.ball_touches`, the number of contacts between the car's body and
the ball since the previous state. The tracker turns that into two things the detectors use:
*touching* (any contact in this step) and a *new touch* (contact after at least `touch_gap`
seconds without contact; `SkillTracker(touch_gap=0.05)` by default). With 8 ticks per step, one
clean step between two contacts makes the second a new touch; stepped every tick, six clean ticks
do.

**In the air.** `Car.on_ground` in RLGym means three or more wheels have contact. That is the
simulator's own rule for giving the jump back, and the flip-reset and wavedash counters use it.
For touches the counters use the stricter `in_the_air(car, ball)`: no wheel has contact with
anything but the ball. A car that rides a wall with two wheels down is not in the air. The ball
is taken to be what the wheels touch when its centre is within 230 uu and on the wheels' side of
the car; a ball that lies on the surface a car drives on is always on the roof's side.

**On target.** `shot` and `save` use `rocket_drills.geometry.on_target`: the ball's flight under
gravity alone, with no drag, bounces or walls, crosses the goal line within 2.5 s, between the
posts and under the crossbar, each narrowed by the ball's radius. A ball rolling along the floor
counts.

**Coordinates.** Blue attacks `+y`. "The goal a car attacks" and "its own goal" follow from
`Car.team_num`.

## What was verified, and how

Three kinds of evidence, named per counter below.

*Hand-built sequences* (`tests/test_detectors.py`): `GameState` sequences written by hand, for
every condition of every detector and for the cases that must not count. These always run.

*The simulator* (`tests/test_sim_skills.py`, `tests/test_sim_drills.py`, and
`tools/measure_drills.py` for the larger samples): scripted play in RocketSim 2.2.1 through
RLGym's engine, with the tracker stepped every 8 ticks and, where it says so, every tick.

*Ground-only play*: a check on false positives. Four cars that only steer at the ball on the
floor (`scripted.drive_at_ball`, 2v2, 400 starts with everything on the floor, 15 s each, 362
car-minutes) never jump, so they can do none of the air mechanics on purpose. Counted in one
such run: 1618 touches, 25 shots, 4 saves, 68 goals, 15 backboard bounces, 1 aerial touch, and
no flip reset, ceiling drop, flick, wavedash or air-dribble touch. This check is not part of the
test suite. It does not repeat exactly: three runs of the same script, with the same seeds,
differed by a few touches and by one save and one goal. The cause was not looked into; it only
showed where several cars collide. The measurements in which one car plays, which are most of
what is quoted here and in `docs/drills.md`, came out the same every time.

`flip_reset_shot`, `ceiling_shot`, `double_tap` and `flick_shot` were verified on hand-built
sequences only: no scripted policy here produces them in the simulator.

## touch

**Event:** `touch` · **Detector:** `TouchDetector`

**Counted when** the car begins a new touch (see above).

- One hit that lasts several ticks is one touch. A ball that rests on the car is one touch until
  it has been off the car for `touch_gap`.
- If two cars touch the ball in the same step, both get a touch.
- Not counted: contact between the wheels and the ball without the body. In 145 `FlipResetChain`
  starts from the `reset` stage, where a coasting car always gets its wheels onto the ball, only
  45.5% produced a touch.

## shot

**Event:** `shot` · **Detector:** `ShotDetector(horizon=2.5)`

**Counted when**, in one step, the car is touching the ball, the ball is on target for the goal
the car attacks after the step, and it was not on target before the step.

- Verified in the simulator: a forward dodge into the ball from the `finish` stage of
  `FlipResetChain` is counted as a shot in 95.4% of 500 starts (it adds more than 300 uu/s toward
  the goal in all of them; the rest were not on target by the definition above).
- Not counted: a touch on a ball that was already on target (adding power to a shot that is
  going in); a shot that needs more than `horizon` seconds to reach the line; a shot that goes in
  off a wall, the ceiling or the crossbar. In the ground-only run there were 68 goals and 25
  shots: most of those goals were slow rolls and deflections.
- Counted although it may not go in: the flight ignores drag, bounces and defenders. A ball that
  will bounce over the bar, or that a defender is standing in front of, is still a shot.
- If two team-mates touch the ball in the step that puts it on target, both get a shot.

## save

**Event:** `save` · **Detector:** `SaveDetector(horizon=2.5)`

**Counted when**, in one step, the car is touching the ball, the ball was on target for the car's
own goal before the step, and is not after it.

- Verified in the simulator through `evaluate_drill` on the `Save` drill (a car that drives at
  the ball saves 31.4% of 220 `in_goal` starts), and by the fact that `conceded` plus `save`
  covers nearly every start.
- A touch that only delays the goal counts: the ball is knocked up, is "over the bar" by the
  flight rule, and drops in afterwards.
- Not counted: stopping a shot that would have taken more than `horizon` seconds to arrive, and
  a block by a car that the ball hits without the ball leaving the target (it still goes in).

## goal

**Event:** `goal` · **Detector:** `GoalDetector`

**Counted when** `GameState.goal_scored` becomes true; the event goes to the car of the scoring
team that touched the ball last since the reset.

- If no car of the scoring team has touched the ball in the episode, the event has `agent=None`
  and counts for the team only. Verified in the simulator: a `Save` shot that nobody stops gives
  exactly that.
- There is no time limit on "last": an own goal is credited to whichever car of the scoring team
  touched the ball most recently, however long ago.
- The scoring team is RLGym's `GameState.scoring_team`.

## aerial_touch and air_dribble_touch

**Events:** `aerial_touch`, `air_dribble_touch` · **Detector:** `AerialTouchDetector(min_ball_height=250)`

**Counted when** the car begins a new touch while it is in the air (see above) and the ball's
centre is above `min_ball_height`. The first such touch after the car leaves a surface is an
`aerial_touch`; each later one before it has a wheel on a surface again is an `air_dribble_touch`.

- The car's own height is not checked. The ball's is: a ball whose centre is above 250 uu cannot
  be reached from the floor without leaving it.
- A car that fell off a wall or the ceiling, or was bumped into the air, is in the air like one
  that jumped. In the ground-only run this gave 1 aerial touch in 362 car-minutes.
- A flick often meets the ball above 250 uu and is then an aerial touch as well.
- Standing on the ball with the wheels does not end the airborne stretch.
- Not counted: wheel contact without the body (see [touch](#touch)); a touch by a car with a
  wheel on a wall.
- Two touches less than `touch_gap` apart are one, so a ball that is carried in continuous
  contact gives one `aerial_touch` and no air-dribble touches.

## flip_reset

**Event:** `flip_reset` · **Detector:** `FlipResetDetector(min_height=300, reach=230, alignment=0.7)`

**Counted when**, in one step, all of these hold:

1. before the step the car was off the ground (`on_ground` false) and had used a jump since it
   last stood on something: `has_jumped`, `has_flipped` or `has_double_jumped` was set;
2. after the step all three flags are clear (the simulator clears them when three or more wheels
   touch something);
3. after the step the ball's centre is within `reach` of the car's, and the car is either off
   the ground again, or still has its wheels on the ball, pointing within `acos(alignment)` (about
   45 degrees) of the direction to it;
4. the car's origin is higher than `min_height`.

- Verified in the simulator: a car that coasts from the `reset` stage of `FlipResetChain` gets
  exactly one counted flip reset in 100% of 500 starts, stepped every 8 ticks and every tick, and
  holds a dodge afterwards in all of them. With the jump taken away from the same starts (a car
  that was only placed), the wheels still touch the ball and nothing is counted, in 40 of 40. A
  forward dodge into the ball with the nose, from the `finish` stage, is never counted.
- The second form of condition 3 matters. The wheels usually stay on the ball for several ticks;
  a counter that waits for the car to be airborne again, with a touch in the same step, misses
  the resets in which a step ends during the contact, and the ones in which only the wheels
  touched.
- Not counted: a car that reached the ball without ever using its jump (it rolled off a wall or
  fell off the ceiling). It gets the same contact, but it already held its dodge and nothing
  changes. Also not counted: anything at or below 300 uu, which keeps the floor out.
- Counted wrongly, in principle: a car that has jumped, brushes a wall, the ceiling or another
  car with three wheels and is off it again within one step, while the ball is within 230 uu.
  Not seen in any run here.
- The counter says nothing about whether the dodge was then used; `flip_reset_shot` does.

**Compared with `rlgym-tools`.** Its `FlipResetReward` (version 2.6.5) pays when the car has a
ball touch in the step, `has_flip` goes from false to true, and the underside faces the ball
within 45 degrees. It therefore needs the body to touch the ball, and does not count a reset that
arrives while the first dodge is still available. The counter here needs neither.

## ceiling_drop

**Event:** `ceiling_drop` · **Detector:** `CeilingDropDetector(min_height=1800)`

**Counted when** before the step the car had its wheels on the ceiling (`on_ground`, `up.z <
-0.7`, origin above `min_height`) and after it the car is off the ground.

- Verified in the simulator: counted in 100% of 500 `Ceiling` starts.
- Jumping off the ceiling counts as well as falling off it, although only the fall keeps the
  dodge.
- Not counted: a car that touches the ceiling for less than one step, since it is never seen on
  it.

## backboard_bounce

**Event:** `backboard_bounce` · **Detector:** `BackboardBounceDetector(touch_window=3.0)`

**Counted when** the ball bounces off the back wall that the last car to touch it attacks, above
the goal, no more than `touch_window` seconds after that touch. The bounce is read from the ball's
velocity along the pitch: toward that wall at more than 300 uu/s before the step, away from it at
more than 100 uu/s after, with the ball within 520 uu of the wall and its centre above 650 uu.

- This is the first half of a double tap and no more than that. Any hard touch that sends the
  ball onto the wall above the goal counts: the ground-only cars did it 15 times in 362
  car-minutes.
- Verified in the simulator: a car flying into the ball in front of the other team's backboard
  gets `touch`, `aerial_touch`, `backboard_bounce` in that order.
- Only the last toucher is credited, and only if it attacks that wall: a defender's deflection
  onto its own backboard counts for nobody.
- Not counted: bounces off the corner walls; bounces with the ball's centre at or below 650 uu
  (the crossbar is at 643); with steps so long that the ball is more than 520 uu from the wall
  again when it is next seen (0.17 s at 3000 uu/s).

## flick

**Event:** `flick` · **Detector:** `FlickDetector(carry_window=0.7)`

**Counted when** the car is touching the ball while dodging (`is_flipping`, or `has_flipped` newly
set) no more than `carry_window` seconds after the ball rode on its roof. Once per carry. The ball
rides on the roof when the car is on the ground and the ball's centre is 100 to 230 uu above the
car's origin along the car's up axis and less than 110 uu from that axis
(`FlickDetector.on_roof`).

- Verified in the simulator: 32 scripted attempts (carry, jump, then a backflip or a front flip,
  with different speeds, ball positions and timings), each run with the tracker stepped every
  tick and every 8 ticks. A flick was counted exactly in the attempts in which the dodge touched
  the ball.
- Any dodge touch counts, however weak. `flick_shot` is the one that asks for a result.
- Not counted: popping the ball with a jump alone; a dodge more than 0.7 s after the ball left
  the roof.
- The roof test fits the six standard hitboxes, on which the ball rests 120 to 137 uu above the
  car's origin.

## wavedash

**Event:** `wavedash` · **Detector:** `WavedashDetector(min_gain=350, dodge_window=0.3)`

**Counted when** the car lands (off the ground before the step, on it after, and not standing on
the ball) with at least `min_gain` more horizontal speed than it had before its dodge. The dodge
is seen in one of two ways:

1. `has_flipped` became set while the car was in the air, no more than `dodge_window` seconds
   before the landing. The gain is measured from the speed in the state before that.
2. No such flag was seen: the dodge, if there was one, began and ended inside the landing step.
   The gain is measured over that step, and counts only for a clean landing: on the flat floor
   (origin below 50 uu, `up.z > 0.9`), at least 300 uu from every wall, with no touch of the ball
   and no bump in that step.

One dodge counts once, however the car bounces.

- Verified in the simulator: `scripted.wavedash` from the `WavedashLanding` drill gains at least
  350 uu/s in 100% of 500 starts and is counted in 99.2% stepped every 8 ticks, 99.8% stepped
  every tick; the car beside it that does nothing is never counted. The misses landed more than
  0.3 s after the dodge began. The ground-only run gave no wavedash.
- Why the conditions on a landing with no dodge seen: before they were added, the ground-only
  cars were counted 7 times in 363 car-minutes, for landings on the curve at the foot of a wall,
  for driving on a wall, and for being bumped. Stepped every tick, a dodge is always seen in the
  air first, so the second way never applies to a real one.
- Not counted: a dodge more than `dodge_window` before the wheels touch; a sideways or diagonal
  dash at speed, which adds to the car's velocity without adding 350 uu/s to its *speed* (at 1400
  uu/s a 500 uu/s push to the side raises the speed by under 100); a dash on a wall, where the
  gain is not horizontal; with steps longer than a tick, a dash near a wall whose dodge fell
  inside the landing step.
- Counted although it is not a wavedash: any dodge that is followed by a wheels-down landing
  within 0.3 s with the speed kept, for instance a low dodge that happened to land flat. Random
  actions do this now and then: 15 times in 200 car-minutes of 2v2 from kickoffs.

**Compared with `rlgym-tools`.** Its `WavedashReward` (version 2.6.5) pays when the car lands
while it or its previous state `is_flipping`. That has no speed condition and no time limit
within the 0.65 s of a flip, and it cannot see a dodge that begins and lands inside one step.

## The shots that follow something

**Events:** the four in the table below · **Detector:** `ShotAfter(skill, trigger, within, airborne=False)`

**Counted when** the car has a `shot` in this step and its latest `trigger` event is no more than
`within` seconds old; with `airborne=True` the car must also be off the ground after the step. The
default tracker has four:

| Skill | Trigger | Within | Also |
|---|---|---|---|
| `flip_reset_shot` | `flip_reset` | 2 s | |
| `ceiling_shot` | `ceiling_drop` | 2 s | |
| `double_tap` | `backboard_bounce` | 2 s | the car is off the ground |
| `flick_shot` | `flick` | 0.5 s | |

- These inherit everything said about `shot` and about their trigger. In particular a second
  touch on a rebound that is already going in is not a new shot, so it is not a `double_tap`.
- Nothing checks what happened in between. An opponent may have touched the ball between the
  backboard bounce and the shot; the car may have landed and jumped again between the flip reset
  and the shot.
- These are the counters to watch, or to pay for, when the bare mechanic can be collected without
  doing anything useful with it.
- Verified on hand-built sequences only.

## Counters of your own

A detector is a class with a `skills` tuple, an optional `reset(state)` and an
`update(step)`. The `Step` gives it the two states, the time, the touches and the events emitted
so far in this step:

```python
from rocket_drills.skills import Detector, SkillTracker, default_detectors


class Supersonic(Detector):
    """``went_supersonic``: the car reaches supersonic speed."""

    skills = ("went_supersonic",)

    def update(self, step):
        for agent, before, car in step.cars():
            if car.is_supersonic and not before.is_supersonic:
                step.emit("went_supersonic", agent)


tracker = SkillTracker([*default_detectors(), Supersonic()])
```

Detectors run in the order given, and one that builds on another's events
(`step.happened(skill, agent)`, `step.last_time(skill, agent)`) goes after it. Two detectors may
not emit the same name.

## SkillReward

```python
reward_fn = CombinedReward(
    (GoalReward(), 10.0),
    SkillReward({"flip_reset_shot": 2.0, "double_tap": 2.0, "wavedash": 0.2}),
)
```

An RLGym `RewardFunction` that owns a tracker, steps it once per environment step and pays each
car the weight of every event it produced. The events of the last step are in
`shared_info["skill_events"]`; the totals are in `reward_fn.tracker`. Give it a tracker of your
own to change detectors or thresholds, and do not step that tracker yourself as well.

A policy collects whatever is counted in the cheapest way it finds, so everything listed above as
"counted although" is a way to be paid for nothing.
