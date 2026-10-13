# Drills

A drill is an RLGym `StateMutator` that starts an episode in one situation. This page says, for
each drill, what the start looks like, what its parameters do, what a competent policy does from
it, and what was measured in the simulator.

- [Common behaviour](#common-behaviour)
- [How the facts were measured](#how-the-facts-were-measured)
- [Takeoff](#takeoff) · [Aerial](#aerial) · [FlipResetChain](#flipresetchain) ·
  [WallToAirChain](#walltoairchain) · [Ceiling](#ceiling) · [Backboard](#backboard) ·
  [Dribble](#dribble) · [FlickDefence](#flickdefence) · [Save](#save) · [FastSave](#fastsave) ·
  [WavedashLanding](#wavedashlanding)
- [DrillMix](#drillmix)
- [Writing your own drill](#writing-your-own-drill)

## Common behaviour

**Where a drill goes.** After a mutator that creates the cars:

```python
MutatorSequence(FixedTeamSizeMutator(2, 2), Takeoff())
```

A drill writes the ball and every car, so anything an earlier mutator did to them is replaced.
It does not touch the boost pads, the tick count or the game configuration.

**The drill car.** The start is built around the first car of the drill team. With `team=None`
(the default) the team is drawn afresh on every reset. Drills work with any team sizes, including
a lone car, except `FlickDefence`, which needs a car on each team.

**The frame.** A drill is written for a team that attacks `+y`. When the drill team is orange the
whole start is turned half a turn about the centre of the pitch. With the same seed,
`team=ORANGE_TEAM` gives exactly the blue start mirrored that way: positions, velocities,
orientations, boost and jump state, car for car.

**The other cars.** Cars with no part in the drill are placed by layout:

| Layout | For | Where |
|---|---|---|
| `"defending"` | either team | In the car's own half, on the floor, facing up the pitch at 0 to 700 uu/s with 20 to 100 boost. The first such car of a team is in or near its goal mouth, the others between the halfway line and the net. |
| `"behind"` | team-mates | On the floor 400 to 2600 uu behind the drill car (toward its own net), facing up the pitch at 0 to 900 uu/s with 10 to 100 boost. |
| `"random"` | either team | Anywhere on the floor, facing any way, at 0 to 1400 uu/s with 0 to 100 boost. |
| `"away"` | either team | Parked, standing still, in a row across the pitch 520 uu in front of its own goal line, out of the play. |

A car that would start within 300 uu of another car or of the ball, or inside an area the drill
keeps clear (given below per drill), is drawn again, up to 24 times, and parked as an `"away"`
car if that keeps failing. A `"defending"` car that has to be drawn again may land anywhere in its
half, so the keeper is not always in net.

**Stages.** Drills with more than one kind of start call them stages. `stages` takes one name, a
list of names (equal weights) or a mapping from name to weight; the class attribute `STAGES` holds
the defaults.

**What a drill reports.** After `apply`, `shared_info["drill"]` is a `DrillInfo`:

| Field | Meaning |
|---|---|
| `name` | The drill's class name. |
| `stage` | The stage drawn (`"default"` for drills with one). |
| `agent` | The drill car's id. |
| `team` | Its team. |
| `details` | Drill-specific promises, in arena coordinates: see each drill. |

**Cars in the air.** A car placed in the air counts, to the simulator, as never having jumped: it
keeps its dodge until it lands. Where the situation implies a jump, the drill gives the car one
with `mark_jumped(car, air_time)`, which sets `has_jumped` and `air_time_since_jump`; the dodge is
then available for the rest of the 1.25 s window and not after. Checked in the simulator: a car
marked 0.5 s or 1.2 s into the window dodges when asked, one marked 1.3 s does not, one marked
0.5 s and left for another second does not, and a car that was only placed still dodges 1.4 s
later.

**A ball at rest stays put.** RocketSim does not move a ball whose velocity is exactly zero, even
in mid-air. No drill here leaves the ball without velocity.

## How the facts were measured

`python tools/measure_drills.py --starts 500 --seed 0` plays 500 starts per check in RocketSim and
prints the share of starts in which each claim held. The numbers below are that output, with
`rlgym-rocket-league` 2.0.1 and `rocketsim` 2.2.1, in 1v1, with decisions every 8 ticks and no
action delay (`RocketSimEngine(rlbot_delay=False)`). The checks are in `tests/physical.py`, and
`tests/test_sim_drills.py` asserts lower bounds on the same quantities over 60 starts.

"Scripted" means one of the stateless policies in `rocket_drills.scripted`. They are crude on
purpose: the question is whether the start allows the play, not how well it can be played.

One check covers every drill: in 25 starts of each, in 1v1 and 3v3, every car's wheel contact and
jump state after one tick of physics equal what the drill wrote (a car put on a surface is on it,
a car put in the air is not), and a ball in free flight has changed its velocity by gravity and
drag only.

The baseline tables further down come from `evaluate_drill(..., episodes=500)` with `seed=0`
against an opponent that does nothing, on the same engine (no action delay); their outcomes are
defined in the function's docstring. The table in the README was made with RLGym's default
one-tick action delay instead, which is why a few of its figures differ slightly from these.

All of these runs came out the same every time they were repeated. A run in which four cars
chase the ball and collide did not (see `docs/skills.md`).

## Takeoff

**The start.** The drill car is on the floor, driving up the pitch at 500 to 1700 uu/s, heading
within 0.6 rad of straight ahead, with 45 to 100 boost. The ball is in the air and on its way to
a spot ahead of the car: 550 to 1900 uu along the car's heading, give or take 120 uu, at a height
of 300 uu up to what a boosted jump has climbed by the time the ball arrives (at most 1000 uu).
The ball arrives 0.8 to 1.5 s after the start, drifting at up to 450 uu/s.

| Stage | Default | The car starts |
|---|---|---|
| `attacking` | 70% | between its own third and the other team's box (`y` from -800 to 3000 in its own frame) |
| `defending` | 30% | deep in its own half (`y` from -4300 to -1200) |

By default the other team is in its own half and none of its cars is within 1500 uu of the spot.

**Parameters.** `arrival_time=(0.8, 1.5)` seconds until the ball is at the spot.
`height=(300, 1000)` range of the spot's height; the top is lowered to the reachable height for
the arrival time (`rocket_drills.drills.takeoff.reachable_height`). `upper_heights=False`: keep to
the upper 30% of the reachable heights. `boost=(45, 100)`.

**Details.** `"target"`: the spot. `"target_time"`: seconds until the ball is there.

**What a competent policy does.** Jumps early, tilts back and boosts to meet the ball at the spot
before anyone else can, and sends it on toward the other net. A policy that waits for the ball to
come down gives the play away, which is the habit this drill is for.

**Measured.**

| Claim | `attacking` | `defending` |
|---|---|---|
| Left alone, the ball is within 40 uu of the spot at the stated time | 100% (furthest: 7.8 uu) | 100% (furthest: 7.5 uu) |
| `scripted.boosted_jump` has the car within 130 uu below the spot's height by 0.07 s after that time | 94.8% | 97.0% |
| ... at or above the spot's height by then | 84.0% | 82.6% |
| ... and touches the ball (it does not steer) | 34.0% | 35.0% |

## Aerial

**The start.** The ball is anywhere over the pitch (within 3000 uu of the middle across and 4000
along), 450 to 1500 uu up, drifting at a few hundred uu/s. The drill car is on the floor 400 to
1600 uu from the point below the ball, facing it to within about 0.3 rad, rolling at up to 1000
uu/s, with 50 to 100 boost. The other cars are anywhere.

**Parameters.** `height=(450, 1500)`, `distance=(400, 1600)` along the floor, `boost=(50, 100)`.

**What a competent policy does.** Goes up and hits the ball. There is no promised spot and no
direction to play in: this is the plain aerial, for learning to get to a high ball at all.
`Takeoff` is the timed, directed version.

**Measured.** Only the general check above, and one baseline: `scripted.boosted_jump`, which
jumps without aiming, touches the ball in the air in 7.4% of 500 starts. Nothing else about this
drill was measured in the simulator.

## FlipResetChain

**The start.** Four links of one play. In all but the take-off the ball hangs in the other
team's third (`x` within 1400, `y` from 2300 to 4200, 450 to 1000 uu up), drifting at up to 300
uu/s along the line to a point in their goal mouth, and the drill car is on the far side of the
ball from that goal.

| Stage | Default | The drill car |
|---|---|---|
| `takeoff` | 10% | On the floor under a high ball ahead: the `Takeoff` start, attacking, upper heights. |
| `approach` | 20% | In the air 450 to 900 uu behind the ball and up to 250 uu under that line, nose at the ball, closing at 500 to 1000 uu/s, 40 to 100 boost. It jumped 0.3 to 0.9 s ago and still has its dodge. |
| `reset` | 30% | Wheels toward the ball and the goal, nose up, 175 to 260 uu from the ball's centre, closing at 300 to 650 uu/s: the wheels meet the ball within a few ticks. It has jumped. In 75% of these starts that was 1.3 to 2.5 s ago and the dodge is gone; in the rest 0.5 to 1.2 s ago. |
| `finish` | 40% | Nose on the ball along the line to the goal, 170 to 220 uu behind the ball's centre, closing at 50 to 350 uu/s, 20 to 100 boost. It has not jumped, so it holds a dodge: the state of a car just off a reset. In half of these starts the other team's first car is out wide rather than in net. |

No car of the other team starts within 1500 uu of the ball.

**Parameters.** `ball_height=(450, 1000)`. `spent_dodge_share=0.75`: share of `reset` starts
with the dodge already gone. `keeper_out_share=0.5`: share of `finish` starts with their first
car out wide (only with `opponents="defending"`).

**Details.** `"aim"`: the point in the goal mouth that the ball drifts toward (all stages but
`takeoff`, which gives `"target"` and `"target_time"` as `Takeoff` does).

**What a competent policy does.** From `finish`: dodges into the ball at once, which is a shot.
From `reset`: lets the wheels touch, comes off the ball with the dodge back, turns its nose onto
the ball and uses the dodge. From `approach`: rolls so that its wheels lead and arrives as in
`reset`. Each stage ends where the next begins, so the value of the finish can reach back through
the chain without the starts ever being moved.

**Why the car has jumped.** With a car that was only placed, the `reset` stage teaches nothing:
the car already holds a dodge, the contact changes no flag, and no flip reset is counted. That
case is in the tests: the same 40 starts with the jump removed give wheel contact in at least 36
and no counted reset.

**Measured.**

| Claim | Share of 500 |
|---|---|
| `finish`: the car starts holding a dodge | 100% |
| `finish`: `scripted.forward_dodge` adds more than 300 uu/s to the ball's speed toward the goal | 100% |
| `finish`: that touch is counted as a `shot` | 95.4% |
| `finish`: it is a goal within 3 s (their car idle, out of its net in half of the starts) | 84.2% |
| `reset`: the car starts in the air having jumped | 100% |
| `reset`: its dodge is spent at the start | 74.6% |
| `reset`: doing nothing, the car gets a counted `flip_reset` | 100% |
| `reset`: ... counted exactly once | 100% |
| `reset`: ... and holds a dodge afterwards | 100% |

The `reset` rows are the same with the tracker stepped every tick instead of every 8 ticks.

Baselines over the default mix of stages (500 starts, 3 s, opponent idle):

| Stage | Starts | Policy | touch | flip_reset | shot | goal |
|---|---|---|---|---|---|---|
| `finish` | 194 | `idle` | 1.000 | 0.000 | 0.139 | 0.062 |
| `finish` | 194 | `forward_dodge` | 1.000 | 0.000 | 0.912 | 0.789 |
| `approach` | 114 | `idle` | 1.000 | 0.000 | 0.939 | 0.675 |
| `approach` | 114 | `forward_dodge` | 0.763 | 0.044 | 0.684 | 0.632 |
| `reset` | 145 | `idle` | 0.455 | 1.000 | 0.000 | 0.000 |
| `reset` | 145 | `forward_dodge` | 0.724 | 0.828 | 0.172 | 0.117 |
| `takeoff` | 47 | `idle` | 0.128 | 0.000 | 0.021 | 0.000 |

Two things to know from this table. Against a car that does nothing, coasting from `approach`
already scores in two starts out of three, because the ball is on the line to the goal: the stage
only asks for a reset once somebody defends. And wheels on the ball are a `touch` in fewer than
half of the resets, because RocketSim reports a touch for contact with the car's body, not its
wheels.

## WallToAirChain

**The start.** The wall-to-air dribble in four links, on either side wall anywhere along it, the
other team's back wall and the drill team's own back wall. Low on a back wall the play stays at
least 1150 uu from the middle, beside the goal.

| Stage | Default | The start |
|---|---|---|
| `approach` | 10% | On the floor. In 70% the ball rolls at the wall at 400 to 1300 uu/s from 250 to 1300 uu out, the car 350 to 1100 uu further out and behind it; in 30% the ball is just starting up the wall (150 to 350 uu high, rising at 500 to 1100 uu/s) and the car is 250 to 800 uu from the wall. The car heads for where the ball will be, at 700 to 1600 uu/s. |
| `setup` | 20% | The ball rolls up the wall, 450 to 1000 uu high, rising at 600 to 1400 uu/s. The car is on the wall 260 to 450 uu below it, wheels on the wall, driving up at 900 to 1500 uu/s. |
| `launch` | 30% | The ball is 1450 to 1800 uu high, 180 to 450 uu off the wall and leaving it at 150 to 650 uu/s. The car is on the wall, 1000 to 1450 uu high, driving up at 600 to 1200 uu/s. |
| `finish` | 40% | The ball is high (900 to 1750 uu) and drifting away from the wall. The car is in the air 260 to 500 uu from it on any side, nose at it, moving with it. |

**Parameters.**

- `walls`: weights over `"side"`, `"their_back"` and `"own_back"` (defaults 55%, 25%, 20%; the
  two side walls share theirs).
- `challenger_share=0.35`: share of `launch` and `finish` starts in which a defender is already
  in the air between the ball's flight and its own net, 0.3 to 0.9 s after its jump. With more
  than one opponent it is the last of them, and the first stays on the ground. A challenger is
  never put within 400 uu of the drill car or of the ball; a start in which twelve draws find no
  such place goes without one (6 of 4000 `launch` starts, none of 4000 `finish` starts).
- `low_boost_share=0.25`: share of starts in which the drill car has 0 to 20 boost instead of
  enough for the play (40 to 60 and up, by stage), so that a policy also meets the case where
  going up is wrong.
- `jumped_share=0.5`: share of `finish` starts in which the car left the wall with a jump 0.2 to
  1.0 s ago; in the others it rolled off and holds its dodge.

**Details.** `"wall"`: `"side"`, `"their_back"` or `"own_back"`.

**What a competent policy does.** Takes the ball up the wall under control, leaves the wall with
it near the top so that the ball drops toward the other team's half, and carries it through the
air to the net, changing its timing when a challenger is up. With an empty tank it does something
else.

**Measured.**

| Claim | Share of 500 |
|---|---|
| `approach`: the car's wheels are down after 8 ticks | 100% (99.6% on the flat floor; the other two cars were on the curve at the foot of a side wall) |
| `setup`: the car is on a wall, wheels in contact, after 8 ticks | 100% |
| `launch`: the same | 100% |
| `finish`: the car is in the air after 8 ticks | 100% |

A ball placed on the curve at the foot of a side wall (the 30% branch of `approach`) is put on the
curve, moving along it; it still takes one bump of up to about 300 uu/s within its first few
ticks, as a ball rolling into that curve does on its own, because the arena's curve is made of
flat facets. Whether the play itself is on from each stage (that a car can carry the ball from `setup`
through `finish`) was not measured: no scripted policy here can do it.

## Ceiling

**The start.** The drill car is on the ceiling, wheels in contact, anywhere over the middle of
the pitch, driving toward the other team's half at 400 to 1200 uu/s with 30 to 100 boost. The
ball is 400 to 900 uu ahead of it and 400 to 900 uu below it (never lower than 300 uu), drifting
up the pitch at up to 600 uu/s.

**Parameters.** `boost=(30, 100)`.

**What a competent policy does.** A car cannot stay on the ceiling. It falls off within a
quarter of a second whatever it does, and because it leaves without jumping it keeps its dodge
all the way down. The play is to drop, line up on the ball and spend the dodge on it: a ceiling
shot.

**Measured.**

| Claim | Share of 500 |
|---|---|
| The car's wheels are on the ceiling after 8 ticks | 100% |
| Doing nothing, it is off the ceiling within 0.6 s | 100% |
| The tracker counts a `ceiling_drop` | 100% |
| 0.6 s after the start it is in the air, has not jumped and can dodge | 100% |

## Backboard

**The start.** The ball is 1100 to 2100 uu from the other team's back wall, within 900 uu of the
middle, 700 to 1300 uu up, and on its way to a point on that wall within 1000 uu of the middle
and 750 to 1500 uu high. It travels at 1200 to 2000 uu/s along the floor and its climb is set so
that it gets there. The default heights keep the whole ball above the crossbar.

| Stage | Default | The drill car |
|---|---|---|
| `airborne` | 50% | In the air 500 to 1100 uu behind the ball and up to 300 uu below it, nose at it, at 70% to 90% of its velocity. It jumped 0.3 to 0.9 s ago, so its dodge runs out about when the ball comes off the wall. |
| `grounded` | 50% | On the floor 600 to 1200 uu behind the ball, driving at it at 800 to 1500 uu/s. |

**Parameters.** `impact_height=(750, 1500)`, `boost=(40, 100)`.

**Details.** `"target"`: the point on the wall. `"target_time"`: seconds until the ball is there.

**What a competent policy does.** Reads where the rebound goes and hits it before it comes down:
in the air from the `airborne` start, with a well-timed jump from the `grounded` one. The same
read, after the car's own touch put the ball on the wall, is a double tap.

**Measured.**

| Claim | `airborne` | `grounded` |
|---|---|---|
| Left alone, the ball bounces off their back wall with its centre above 650 uu | 100% | 100% |
| It passes within 30 uu of the stated point | 100% | 100% |
| It bounces within 0.05 s of the stated time | 100% | 100% |

Whether a car can reach the rebound from these starts was not measured.

## Dribble

**The start.** The drill car is on the floor anywhere from deep in its own half to the edge of
the other team's box, heading at their goal to within about 0.5 rad, at 200 to 1300 uu/s, with
the ball resting on its roof and moving with it. The ball sits up to 20 uu behind or 30 uu ahead
of the car's centre, at the height at which it rests on that hitbox's roof (measured per hitbox:
150.3 uu for the Octane). No other car starts within 800 uu.

**Parameters.** `speed=(200, 1300)`, `boost=(0, 100)`.

**What a competent policy does.** Keeps the ball on the roof by matching its speed, carries it
toward the net, and flicks it when a defender commits. For the other team this is practice at
challenging a carry.

**Measured.**

| Claim | Share of 500 |
|---|---|
| With `scripted.carry`, the ball is still on the roof after 0.53 s | 100% |
| ... after 1 s | 98.4% |
| ... after 2 s | 83.4% |
| With no controls at all, it is still there after 0.53 s | 90.2% |
| ... after 1 s | 0.2% |

"On the roof" is the test the flick counter uses (`FlickDetector.on_roof`). A car that only
coasts slows down and the ball rolls off its nose.

No scripted flick is offered. In 32 scripted jump-and-dodge attempts from a carry (see
`tests/test_sim_skills.py`) the dodge reached the ball in some and not in others, depending on
timing, speed and where the ball sat, and a stateless script that flicks reliably from every
start of this drill was not found.

## FlickDefence

**The start.** The other team's first car has the ball on its roof and drives at the drill
team's goal at 700 to 1500 uu/s, aimed inside the posts, with 20 to 100 boost. The drill car is
goal-side of it and on its line, 900 to 2400 uu ahead and up to 350 uu to the side, moving the
same way at 50% to 110% of the carrier's speed. In half of the starts it faces the ball (it is
reversing); in the other half it faces its own goal. No other car starts within 800 uu of either.

**Parameters.** `gap=(900, 2400)`, `boost=(10, 100)`.

**Details.** `"carrier"`: the id of the car that has the ball.

**What a competent policy does.** Holds its ground long enough to make the carrier commit, and
when the flick comes, goes up to meet the ball instead of backing off underneath it.

**Measured.** Only the general check above, plus what the unit tests assert about the layout
(the ball is on the carrier's roof by the flick counter's test, the defender is goal-side in
every start, half of the defenders face the ball). What happens next depends on two policies and
was not measured.

## Save

**The start.** The ball is in the drill team's half (800 to 3800 uu from the halfway line),
rolling or in the air, and already on its way to a point inside the posts at 1400 to 3200 uu/s. A
shot that would take longer than 2.2 s is sped up until it does not. The other team's first car
follows the shot in, 400 to 1400 uu behind the ball (further back if that would put it within
400 uu of the defender). The defender never starts within 350 uu of the ball along the floor.

| Stage | Default | The drill car (the defender) |
|---|---|---|
| `in_goal` | 45% | On its goal line or up to 420 uu in front of it, within 700 uu of the middle, facing the ball. |
| `shadowing` | 35% | Between the ball and the net, 35% to 75% of the way back, moving back with the shot at 300 to 1400 uu/s, facing the ball or its own goal. |
| `rotating` | 20% | 1500 to 3000 uu out to one side, heading for its goal at 600 to 1800 uu/s. |

**Parameters.** `speed=(1400, 3200)`. `airborne_share=0.4`: share of shots in the air.
`boost=(20, 100)` for the defender. `shooter=True`: put the other team's first car behind the
ball.

**Details.** `"target"`: where the ball crosses the goal line. `"target_time"`: seconds until it
does; exact for shots in the air, a lower bound for rolling shots, which slow down on the floor.

**What a competent policy does.** Gets to the ball's path before the ball gets to the line and
clears it; from `rotating`, gets back first.

**Measured.**

| Claim | Share of 500 |
|---|---|
| The ball is on target for the defender's goal at the start, by the `shot` counter's definition | 100% |
| With the defender moved out of the way, the ball goes in within 5 s | 100% |
| Shots in the air cross the line within 30 uu of the stated point | 100% |
| ... and within 0.03 s of the stated time | 100% |

Baselines (500 starts, 4 s, the shooter idle):

| Stage | Starts | Defender | save | conceded |
|---|---|---|---|---|
| `in_goal` | 220 | `idle` | 0.073 | 0.932 |
| `in_goal` | 220 | `drive_at_ball` | 0.314 | 0.686 |
| `shadowing` | 173 | `idle` | 0.197 | 0.809 |
| `shadowing` | 173 | `drive_at_ball` | 0.220 | 0.780 |
| `rotating` | 107 | `idle` | 0.047 | 0.972 |
| `rotating` | 107 | `drive_at_ball` | 0.065 | 0.935 |

A defender that does nothing still makes some saves by being in the way, and a save does not
end the play: the two columns can add up to more than one.

## FastSave

**The start.** As `Save`, but the shot travels at 2500 to 3800 uu/s from 900 to 4100 uu up the
pitch, which leaves 0.3 to 1.6 s. 40% of the shots are struck off the ground and rising, 35% are already
in the air, 25% run along the floor. The defender has 0 to 15 boost in half of the starts, 15 to
40 in a quarter and 40 to 100 in a quarter.

| Stage | Default | The drill car (the defender) |
|---|---|---|
| `in_goal` | 50% | Goal-side near its net: on the line or up to 800 uu off it, within 800 uu of the middle, roughly facing the ball, at up to 800 uu/s. |
| `shadowing` | 35% | Between the ball and the net, 30% to 80% of the way back, moving back with the shot. |
| `rotating` | 15% | 1200 to 3000 uu out to one side, heading for its goal at 800 to 2000 uu/s. |

**Parameters.** `speed=(2500, 3800)`. `low_boost_share=0.5`: share of starts with 0 to 15 boost;
the rest are split evenly between 15 to 40 and 40 to 100. `shooter=True`.

**Details.** As `Save`.

**What a competent policy does.** Reacts at once, usually with a jump or a dodge at the ball,
since there is no time to drive to it and often no boost to fly to it.

**Measured.** The same four claims as `Save`, each 100% of 500. Baselines (500 starts, 4 s):

| Stage | Starts | Defender | save | conceded |
|---|---|---|---|---|
| `in_goal` | 254 | `idle` | 0.043 | 0.953 |
| `in_goal` | 254 | `drive_at_ball` | 0.067 | 0.929 |
| `shadowing` | 172 | `idle` | 0.047 | 0.948 |
| `shadowing` | 172 | `drive_at_ball` | 0.064 | 0.930 |
| `rotating` | 74 | `idle` | 0.000 | 1.000 |
| `rotating` | 74 | `drive_at_ball` | 0.014 | 0.986 |

## WavedashLanding

**The start.** The ball lies in the middle of the pitch, barely moving. Each team's first car is
on its own side of it, 1500 to 4000 uu away (the two within about 10% of each other unless a wall
cuts one run short), heading at the ball at 300 to 1400 uu/s, 60 to 350 uu off the floor and
sinking at 100 to 600 uu/s, level but for a tilt of up to 0.45 rad (nose up in half of the
starts) and a roll of up to 0.3 rad. Both have 0 to 20 boost. Half of these cars have a jump 0.15
to 0.5 s behind them; the other half have not jumped. All of them can still dodge at the start.

No other car starts within 1200 uu of the ball or 600 uu of either racer. With no opponent the
drill car makes the same start alone.

**Parameters.** `distance=(1500, 4000)` from each car to the ball. `height=(60, 350)` of the
wheels above the floor. `jumped_share=0.5`.

**What a competent policy does.** Dodges forward just before its wheels touch, which keeps the
dodge's speed (about 500 uu/s, at once) through the landing, and gets to the ball first. With an
empty tank it is the only speed on offer.

**Measured.**

| Claim | Share of 500 |
|---|---|
| `scripted.wavedash` has gained at least 350 uu/s of horizontal speed by 16 ticks after its wheels touch | 100% |
| The tracker counts a `wavedash` for it, stepped every 8 ticks | 99.2% |
| ... stepped every tick | 99.8% |
| The tracker counts a `wavedash` for the other car, which does nothing | 0% |

The few scripted landings that were not counted came down on their wheels more than 0.3 s after
the dodge began, which is outside the counter's window.

## DrillMix

```python
mix = DrillMix({KickoffMutator(): 0.6, Takeoff(): 0.2, FlipResetChain(): 0.2}, seed=1)
```

Each reset picks one entry by weight and applies it. Weights are relative. An entry can be a
drill, any RLGym `StateMutator`, or another `DrillMix`; pass `(mutator, weight)` pairs instead of
a mapping to use the same object twice. `mix.counts` is how often each entry has been picked.

`shared_info["drill"]` always describes the pick: the drill's own `DrillInfo`, or
`DrillInfo(name="KickoffMutator")` for a mutator that is not a drill.

With a `seed`, the choice and every drill in the mix are seeded from it, so the sequence of
starts repeats. Mutators that are not drills keep their own randomness (RLGym's `KickoffMutator`
draws from Python's `random` module).

`rlgym-tools` has `WeightedSampleMutator`, which also picks a mutator by weight. Drills work in
it as well; it neither seeds them nor records the pick for mutators that are not drills.

## Writing your own drill

Subclass `Drill` and implement `_build(scene, stage)`. The `scene` is the start in the frame
where the drill team attacks `+y`; whatever you place is mirrored for orange. Place the ball and
the cars that have a part; the rest are placed by the layouts afterwards.

```python
from rocket_drills.drills import Drill, Scene


class KickoffChase(Drill):
    """The ball rolls away from the drill car toward the other team's net."""

    def _build(self, scene: Scene, stage: str):
        rng = scene.rng
        x, y = rng.uniform(-2000, 2000), rng.uniform(-3000, 0)
        scene.ball((x, y + 600, 93.15), (0, rng.uniform(500, 1500), 0))
        scene.ground_car(scene.me, x, y, yaw=1.5708, speed=rng.uniform(0, 1000), boost=50)
        scene.keep_clear(x, y, 1000)  # nobody else within 1000 uu of the drill car
        return {"note": "anything you return ends up in DrillInfo.details"}
```

`scene.ground_car` puts a car on the floor, `scene.surface_car` on a wall or the ceiling, and
`scene.air_car(..., air_time=...)` in the air with or without a jump behind it. `scene.me`,
`scene.mates` and `scene.opponents` are the car ids. For a drill with stages, set the class
attribute `STAGES = {"name": weight, ...}` and accept `stages` in `__init__`.

To write a start without the `Drill` machinery, any `StateMutator` can use the helpers in
`rocket_drills.placement` directly: `set_ball`, `set_car` and `mark_jumped`.
`rocket_drills.placement.empty_state(blue, orange)` makes a state to try a mutator on without
starting the simulator.
