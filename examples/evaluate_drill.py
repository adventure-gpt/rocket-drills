"""Run a policy on one drill and print what it makes of each stage.

Examples:
    python examples/evaluate_drill.py FlipResetChain --policy forward_dodge --seconds 3
    python examples/evaluate_drill.py Takeoff --policy boosted_jump --episodes 500
    python examples/evaluate_drill.py Save --policy drive_at_ball --opponent idle --team-size 2
    python examples/evaluate_drill.py WallToAirChain --stages launch finish --policy idle

The policies offered here are the scripted baselines in ``rocket_drills.scripted``. To evaluate
your own, write a function that takes the game state and the car to control and returns that
car's eight controls, and pass it to ``evaluate_drill`` in their place::

    def my_policy(state, agent):
        obs = obs_builder.build_obs([agent], state, {})[agent]
        action = model.act(obs)                     # your network
        return lookup_table[action]                 # eight controls for this car

    report = evaluate_drill(FlipResetChain(), my_policy, opponent_policy=my_policy)
"""

import argparse

from rocket_drills import evaluate_drill, scripted
from rocket_drills.drills import ALL_DRILLS
from rocket_drills.sim import make_engine

DRILLS = {cls.__name__: cls for cls in ALL_DRILLS}
POLICIES = {
    name: getattr(scripted, name)
    for name in ("idle", "drive_at_ball", "boosted_jump", "forward_dodge", "wavedash", "carry")
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-stage outcome rates of a policy on a drill.")
    parser.add_argument("drill", choices=sorted(DRILLS), help="the drill to start from")
    parser.add_argument("--stages", nargs="+", help="only these stages (default: the drill's mix)")
    parser.add_argument("--policy", choices=POLICIES, default="idle", help="drives the drill team")
    parser.add_argument("--opponent", choices=POLICIES, default="idle", help="drives the others")
    parser.add_argument("--episodes", type=int, default=200, help="number of starts")
    parser.add_argument("--seconds", type=float, default=5.0, help="length of an episode")
    parser.add_argument("--team-size", type=int, default=1, help="cars per team")
    parser.add_argument("--seed", type=int, default=0, help="seed of the drill")
    parser.add_argument("--meshes", help="folder of collision meshes, if RLGym's are not found")
    args = parser.parse_args()

    options = {"seed": args.seed}
    if args.stages:
        options["stages"] = args.stages
    drill = DRILLS[args.drill](**options)
    report = evaluate_drill(
        drill,
        POLICIES[args.policy],
        opponent_policy=POLICIES[args.opponent],
        episodes=args.episodes,
        seconds=args.seconds,
        team_size=args.team_size,
        engine=make_engine(args.meshes),
    )
    print(f"{args.drill}: {args.policy} against {args.opponent}, {args.episodes} starts")
    print(report.format())


if __name__ == "__main__":
    main()
