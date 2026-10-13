"""Measure what really happens from each drill's starts, in the simulator.

    python tools/measure_drills.py [--starts 500] [--seed 0]

Prints, for every drill that has a physical check, the share of starts in which each claim held.
The figures in docs/drills.md are this script's output. The checks themselves live in
tests/physical.py, where the test suite asserts lower bounds on the same quantities.
"""

import argparse
import sys
import time
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))

import physical
from rocket_drills.sim import make_engine


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--starts", type=int, default=500, help="starts per check")
    parser.add_argument("--seed", type=int, default=0, help="seed of every drill")
    parser.add_argument("--meshes", help="folder of collision meshes, if RLGym's are not found")
    args = parser.parse_args()
    engine = make_engine(args.meshes, rlbot_delay=False)
    count, seed = args.starts, args.seed

    checks = [
        ("Takeoff / attacking", lambda: physical.takeoff(engine, "attacking", count, seed)),
        ("Takeoff / defending", lambda: physical.takeoff(engine, "defending", count, seed)),
        ("FlipResetChain / finish", lambda: physical.flip_reset_finish(engine, count, seed)),
        ("FlipResetChain / reset", lambda: physical.flip_reset_contact(engine, count, seed)),
        (
            "FlipResetChain / reset, tracker stepped every tick",
            lambda: physical.flip_reset_contact(engine, count, seed, tick_skip=1),
        ),
        *(
            (
                f"WallToAirChain / {stage}",
                lambda s=stage: physical.wall_to_air(engine, s, count, seed),
            )
            for stage in ("approach", "setup", "launch", "finish")
        ),
        ("Save", lambda: physical.save(engine, False, count, seed)),
        ("FastSave", lambda: physical.save(engine, True, count, seed)),
        ("Dribble", lambda: physical.dribble(engine, count, seed)),
        ("Ceiling", lambda: physical.ceiling(engine, count, seed)),
        ("Backboard / airborne", lambda: physical.backboard(engine, "airborne", count, seed)),
        ("Backboard / grounded", lambda: physical.backboard(engine, "grounded", count, seed)),
        ("WavedashLanding", lambda: physical.wavedash(engine, count, seed)),
        (
            "WavedashLanding, tracker stepped every tick",
            lambda: physical.wavedash(engine, count, seed, tick_skip=1),
        ),
    ]
    print(
        f"{count} starts per check, seed {seed}; rlgym-rocket-league "
        f"{version('rlgym-rocket-league')}, rocketsim {version('rocketsim')}; decisions every 8 "
        "ticks unless stated, no action delay"
    )
    began = time.perf_counter()
    for name, check in checks:
        print(f"\n{name}")
        for claim, value in check().items():
            print(f"  {claim:50s} {value:8.3f}")
    print(f"\n{time.perf_counter() - began:.0f} s")


if __name__ == "__main__":
    main()
