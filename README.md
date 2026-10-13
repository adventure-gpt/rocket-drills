# rocket-drills

[![tests](https://github.com/adventure-gpt/rocket-drills/actions/workflows/tests.yml/badge.svg)](https://github.com/adventure-gpt/rocket-drills/actions/workflows/tests.yml)
[![licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](LICENSE)

Practice scenarios and skill counters for Rocket League bots trained with
[RLGym](https://rlgym.org) v2.

- **Drills** are state mutators that start an episode in one situation: a ball in the air ahead of
  a car on the ground, a car about to get a flip reset, a shot coming at a defender, a ball
  rolling up the wall. Long mechanics come as chains in which every stage is a start of its own.
- **Skill counters** watch consecutive game states and count mechanics (flip resets, wavedashes,
  aerial touches, ceiling shots, double taps, flicks, shots, saves), so you can measure how often
  a policy does each of them per car-minute.

Both are plain RLGym v2 components. A drill goes wherever a `StateMutator` goes; the tracker reads
nothing but `GameState` objects.

## Install

```
pip install git+https://github.com/adventure-gpt/rocket-drills
```

This needs Python 3.10 or newer and installs `rlgym-rocket-league[sim]` (RLGym's Rocket League
package with RocketSim). It was written and tested against `rlgym-rocket-league` 2.0.1,
`rlgym-api` 2.0.0 and `rocketsim` 2.2.1 on Python 3.12. `rlgym-rocket-league` 2.0.1 requires
numpy below 2, so use a Python version for which numpy 1.x is available (3.10 to 3.12).

## Quick start

An RLGym environment that starts half of its episodes from drills, with the counters running:

```python
import numpy as np
from rlgym.api import RLGym
from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
from rlgym.rocket_league.done_conditions import GoalCondition, TimeoutCondition
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import GoalReward
from rlgym.rocket_league.sim import RocketSimEngine
from rlgym.rocket_league.state_mutators import FixedTeamSizeMutator, KickoffMutator, MutatorSequence

from rocket_drills import DrillMix, SkillTracker
from rocket_drills.drills import FlipResetChain, Save, Takeoff

mix = DrillMix({KickoffMutator(): 0.5, Takeoff(): 0.2, FlipResetChain(): 0.2, Save(): 0.1})
env = RLGym(
    state_mutator=MutatorSequence(FixedTeamSizeMutator(1, 1), mix),
    obs_builder=DefaultObs(zero_padding=1),
    action_parser=RepeatAction(LookupTableAction(), repeats=8),
    reward_fn=GoalReward(),
    transition_engine=RocketSimEngine(),
    termination_cond=GoalCondition(),
    truncation_cond=TimeoutCondition(6.0),
)
tracker = SkillTracker()
for _ in range(50):
    env.reset()
    tracker.reset(env.state)
    done = False
    while not done:
        actions = {agent: np.array([np.random.randint(90)]) for agent in env.agents}  # your policy
        _, _, terminated, truncated = env.step(actions)
        tracker.step(env.state)
        done = any(terminated.values()) or any(truncated.values())
print(tracker.summary().format())
```

It prints events per car-minute for each team, one row per skill. With random actions most rows
are zero; the few flip resets come from the `FlipResetChain` starts in which the car only has to
coast into the ball.

`examples/quickstart.py` is a longer version of this, and `examples/evaluate_drill.py` runs a
policy on one drill and prints per-stage outcome rates.

## Drills

Every drill is an RLGym `StateMutator`. Apply it after a mutator that creates the cars; it works
for any team sizes. The start is built around one car, the *drill car* (the first car of the
drill team). After each reset `shared_info["drill"]` holds a `DrillInfo` with the drill's name,
the stage, the drill car and its team.

| Drill | The start | Stages | Parameters of its own |
|---|---|---|---|
| `Takeoff` | A car on the ground; the ball in the air, on its way to a spot ahead of the car that a jump with boost reaches by the time the ball is there. | `attacking` 70%, `defending` 30% (which half the car is in) | `arrival_time`, `height`, `upper_heights`, `boost` |
| `Aerial` | A slow ball 450 to 1500 uu up and a car on the ground 400 to 1600 uu from the point below it, facing it. | | `height`, `distance`, `boost` |
| `FlipResetChain` | Every link of a flip reset: on the ground under a high ball; in the air behind it; wheels about to meet it (the car has jumped, its dodge usually spent); behind the ball on the line to the goal with a dodge in hand. | `takeoff` 10%, `approach` 20%, `reset` 30%, `finish` 40% | `ball_height`, `spent_dodge_share`, `keeper_out_share` |
| `WallToAirChain` | The wall-to-air dribble on the side walls and both back walls: ball rolling at the wall; ball going up it with the car below; ball coming off near the top with the car on the wall; car in the air next to the ball. | `approach` 10%, `setup` 20%, `launch` 30%, `finish` 40% | `walls`, `challenger_share`, `low_boost_share`, `jumped_share` |
| `Ceiling` | A car driving on the ceiling, the ball below and ahead. It drops off by itself with its dodge intact. | | `boost` |
| `Backboard` | The ball on its way to a point on the other team's back wall above the goal; the car following it, in the air or on the ground. | `airborne` 50%, `grounded` 50% | `impact_height`, `boost` |
| `Dribble` | A car carrying the ball on its roof toward the other team's net. | | `speed`, `boost` |
| `FlickDefence` | The other team's car carries the ball at the drill team's net; the drill car is goal-side of it, backing off. | | `gap`, `boost` |
| `Save` | A shot at 1400 to 3200 uu/s already on target; the defender in goal, shadowing, or rotating back. | `in_goal` 45%, `shadowing` 35%, `rotating` 20% | `speed`, `airborne_share`, `boost`, `shooter` |
| `FastSave` | A shot at 2500 to 3800 uu/s, three in four off the ground, at a defender who has under 15 boost half of the time. | `in_goal` 50%, `shadowing` 35%, `rotating` 15% | `speed`, `low_boost_share`, `shooter` |
| `WavedashLanding` | Each team's first car comes down from a hop with an empty tank, racing to a ball ahead: a dodge into the landing is the only speed to be had. | | `distance`, `height`, `jumped_share` |

All drills also take:

- `team`: `BLUE_TEAM`, `ORANGE_TEAM` or `None` (pick one at random each reset). An orange start is
  the blue start turned half a turn about the centre of the pitch.
- `opponents`: where the other team's cars start unless they have a part in the drill:
  `"defending"` (in their half, the first in or near its net), `"random"` or `"away"` (parked
  in front of their own goal, out of the play).
- `teammates`: `"behind"` (on the floor behind the drill car), `"defending"`, `"random"` or
  `"away"`.
- `seed`: an integer or a `numpy.random.Generator`.
- `stages` (drills that have stages): one name, several names, or a mapping from name to weight.

```python
FlipResetChain(stages="finish")  # one stage alone
FlipResetChain(stages={"reset": 1, "finish": 3})  # your own weighting
WallToAirChain(walls="side", challenger_share=0.0)  # side walls only, nobody challenging
Takeoff(team=BLUE_TEAM, opponents="away", height=(600, 1000), seed=7)
```

`DrillMix` picks one of several mutators by weight on every reset. Its entries can be drills,
RLGym's own mutators, or another `DrillMix`.

[docs/drills.md](docs/drills.md) describes each drill in full: what the start looks like, what a
competent policy does from it, and what was measured in the simulator (for instance: left alone,
the `Takeoff` ball is within 8 uu of the promised spot at the promised time in all of 1000 starts,
and a scripted jump with boost is within 130 uu of that height in time in 95% to 97% of them).

### Cars that start in the air

A car that a state mutator puts in the air has never jumped as far as the simulator knows, so its
dodge never runs out, and wheels on the ball give it nothing. A policy that practises from such
starts can come to rely on a dodge it will not have in a game, and it cannot get a flip reset at
all: the contact happens and changes nothing. The drills here mark airborne cars as having
jumped, with a time since the jump, wherever that is what the situation would be. To do the same
in a drill of your own:

```python
from rocket_drills import mark_jumped, set_car

set_car(car, position, velocity, forward, up, boost=60, on_ground=False)
mark_jumped(car, air_time=1.4)  # jumped 1.4 s ago: no dodge left until a reset or a landing
```

## Skill counters

```python
tracker = SkillTracker()
tracker.reset(first_state)  # at the start of every episode
events = tracker.step(next_state)  # SkillEvent(skill, agent, team, tick) for this step
summary = tracker.summary()  # totals; .per_car(), .per_team(), .format()
```

The tracker can be stepped every tick or every few ticks; all definitions are in seconds.

| Skill | Counted when |
|---|---|
| `touch` | The car begins a touch: contact with the ball after at least 0.05 s without contact. |
| `shot` | The car's touch puts the ball on target for the goal it attacks: the ball's flight under gravity crosses that goal line within 2.5 s, inside the posts and under the bar, and did not before the touch. |
| `save` | The car's touch takes the ball off target (same definition) for its own goal. |
| `goal` | A goal is scored; credited to the scoring team's car that touched the ball last. |
| `aerial_touch` | The car begins a touch while none of its wheels touches a surface and the ball's centre is above 250 uu: the first such touch after leaving a surface. |
| `air_dribble_touch` | The second and each later such touch before the car lands. |
| `flip_reset` | A car that was in the air and had used its jump has its jump flags cleared with the ball within 230 uu, and is either in the air again or still standing on the ball with its wheels; higher than 300 uu. |
| `flip_reset_shot` | A shot within 2 s of the car's flip reset. |
| `ceiling_drop` | The car leaves the ceiling. |
| `ceiling_shot` | A shot within 2 s of the car's ceiling drop. |
| `backboard_bounce` | The ball bounces off the back wall the car attacks, above the goal, within 3 s of that car's touch, with no other car touching it in between. The first half of a double tap. |
| `double_tap` | A shot by a car in the air within 2 s of its backboard bounce. |
| `flick` | The car's dodge touches the ball within 0.7 s of the ball riding on its roof; once per carry. |
| `flick_shot` | A shot in the same step as the car's flick or within 0.5 s after it. |
| `wavedash` | The car lands with at least 350 uu/s more horizontal speed than it had before the dodge it began up to 0.3 s earlier (or, if the dodge fell inside the landing step, than it had a step earlier, for a clean landing on the flat floor). |

[docs/skills.md](docs/skills.md) gives the exact conditions and thresholds of each detector and
what it is known to miss or to count wrongly.

Each skill is a small detector class (`FlipResetDetector`, `WavedashDetector`, ...) with its
thresholds as constructor arguments; pass your own list to `SkillTracker(detectors=[...])`.
`ShotAfter("my_skill", trigger, within)` builds a "followed by a shot" counter for any event.

`SkillReward({"flip_reset_shot": 2.0, "wavedash": 0.2})` is an RLGym `RewardFunction` that pays a
fixed amount per detected event; combine it with your other rewards in `CombinedReward`.

## Evaluating a policy on a drill

```python
from rocket_drills import evaluate_drill, scripted
from rocket_drills.drills import FlipResetChain

report = evaluate_drill(FlipResetChain(seed=0), scripted.forward_dodge, episodes=500, seconds=3)
print(report.format())
```

```
share of starts with each outcome within 3 s
stage                      starts     airborne  first_touch        touch    air_touch   flip_reset        dodge         shot         save         goal     conceded
FlipResetChain / finish       194        1.000        1.000        1.000        1.000        0.000        1.000        0.912        0.000        0.789        0.000
FlipResetChain / approach     114        1.000        0.772        0.772        0.772        0.035        1.000        0.649        0.000        0.667        0.000
FlipResetChain / reset        145        1.000        0.703        0.703        0.703        0.821        1.000        0.179        0.000        0.124        0.000
FlipResetChain / takeoff       47        0.000        0.128        0.128        0.000        0.000        0.000        0.021        0.000        0.000        0.000
```

A policy is any function `policy(state, agent)` that returns the eight controls for one car, so
a trained network fits in by wrapping its observation builder and action parser.
`rocket_drills.scripted` has a few stateless baselines (`idle`, `drive_at_ball`, `boosted_jump`,
`forward_dodge`, `wavedash`, `carry`): the table above is what "dodge forward as soon as you can"
scores against a car that does nothing.

## Collision meshes

RocketSim needs the arena's collision meshes to step the physics. This package ships none.

With `rlgym-rocket-league` 2.0.1 there is nothing to set up: that package carries the soccar
meshes, and RLGym's `RocketSimEngine` loads them from its own folder.

To use other meshes, or if your install has none, RocketSim has to be given a folder (one that
contains `soccar/*.cmf`) before the first engine of the process is created.
`rocket_drills.sim.make_engine(meshes="path/to/collision_meshes")` does that and returns the
engine; so does setting the `ROCKET_DRILLS_MESHES` environment variable and calling
`make_engine()`. RocketSim takes its meshes once per process, so the first folder it is given
stays in use. When the arena cannot be built, `make_engine` raises a
`SimulatorUnavailableError` that says where it looked. The tests use `make_engine` and skip
everything that steps the simulator when it fails.

## Development

```
pip install -e ".[dev]"
ruff check . && ruff format --check .
pytest                      # everything; the simulator tests skip if RocketSim cannot start
pytest -m "not sim"         # only the tests that need no simulator
python tools/measure_drills.py   # the measured facts quoted in docs/drills.md
```

## See also

- [RLGym](https://github.com/lucas-emery/rocket-league-gym), the library this builds on, and its
  documentation at [rlgym.org](https://rlgym.org).
- [rlgym-tools](https://pypi.org/project/rlgym-tools/), the community's collection of RLGym
  components. It meets this package in a few places: `WeightedSampleMutator` is a plain weighted
  choice between mutators (`DrillMix` adds seeding and records the pick); `FlipResetReward` and
  `WavedashReward` are reward functions for two of the mechanics counted here, with definitions
  of their own (compared in [docs/skills.md](docs/skills.md)); `TrainingPackMutator` and
  `ReplayMutator` start episodes from training packs and replays, which mix well with drills in
  a `DrillMix`; `AugmentMutator` mirrors states at random.

## What we learned using them

[docs/findings.md](docs/findings.md) is a short write-up of what we measured while training the bot these
come from: goal-only reward against a shaped one, why a drill has to pay at its last step, what a few
ticks of reaction time are worth, and a few traps in measuring bots against each other.

## Acknowledgements

The drills and counters come out of the Whiskers bot project.

## Licence

MIT. See [LICENSE](LICENSE).
