# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## 0.1.0

First release. Written and tested against `rlgym-rocket-league` 2.0.1, `rlgym-api` 2.0.0 and
`rocketsim` 2.2.1 on Python 3.12.

### Added

- Drills, as RLGym v2 state mutators: `Takeoff`, `Aerial`, `FlipResetChain`, `WallToAirChain`,
  `Ceiling`, `Backboard`, `Dribble`, `FlickDefence`, `Save`, `FastSave` and `WavedashLanding`,
  with weighted stages, a choice of drill team, layouts for the cars that have no part in the
  drill, and seeding.
- `DrillMix`, a weighted mixture of drills and other state mutators that records which one each
  episode started from.
- `DrillInfo` in `shared_info["drill"]`: the drill, the stage, the drill car and what the drill
  promises about the start.
- `mark_jumped`, `set_car`, `set_ball` and `empty_state` for writing states by hand, including
  cars that start in the air with a jump behind them.
- `SkillTracker` with detectors for `touch`, `shot`, `save`, `goal`, `aerial_touch`,
  `air_dribble_touch`, `flip_reset`, `ceiling_drop`, `backboard_bounce`, `flick` and `wavedash`,
  and the shots that follow them: `flip_reset_shot`, `ceiling_shot`, `double_tap` and
  `flick_shot`.
- `SkillReward`, an RLGym reward function that pays per detected event.
- `evaluate_drill`, which runs a policy on a drill and reports per-stage outcome rates, and a
  few scripted baseline policies in `rocket_drills.scripted`.
- `rocket_drills.sim.make_engine`, which starts RLGym's RocketSim engine and reports clearly
  when the collision meshes are missing.
