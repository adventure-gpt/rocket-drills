"""The code in the README and the docs is run, and their tables are checked, so they cannot rot."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
from rlgym.rocket_league.common_values import BLUE_TEAM, ORANGE_TEAM

from helpers import after, quiet_state, skills_of
from rocket_drills.drills import ALL_DRILLS, Drill
from rocket_drills.placement import empty_state
from rocket_drills.skills import SkillTracker, default_detectors

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
DRILLS = (ROOT / "docs" / "drills.md").read_text(encoding="utf-8")
SKILLS = (ROOT / "docs" / "skills.md").read_text(encoding="utf-8")


def code_blocks(text: str) -> list[str]:
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


@pytest.mark.sim
def test_the_quick_start_runs_as_printed(engine, capsys):
    quick_start = code_blocks(README)[0]
    assert quick_start.startswith("import numpy as np")
    exec(compile(quick_start, "README quick start", "exec"), {})
    printed = capsys.readouterr().out.splitlines()
    assert printed[0].split() == ["events", "per", "car-minute", "blue", "orange"]
    assert [line.split()[0] for line in printed[1:-1]] == list(SkillTracker().skills)
    assert printed[-1].startswith("car-minutes")


def test_the_readme_lists_every_drill_and_every_skill():
    for cls in ALL_DRILLS:
        assert f"| `{cls.__name__}` |" in README
        for stage in cls.STAGES:
            if stage != "default":
                assert f"`{stage}`" in README
    for skill in SkillTracker().skills:
        assert f"| `{skill}` |" in README


def test_the_docs_have_a_section_for_every_drill_and_every_detector():
    for cls in ALL_DRILLS:
        assert f"\n## {cls.__name__}\n" in DRILLS
        for stage in cls.STAGES:
            if stage != "default":
                assert f"| `{stage}` |" in DRILLS
    for skill in SkillTracker().skills:
        assert f"`{skill}`" in SKILLS
    for detector in default_detectors():
        assert type(detector).__name__ in SKILLS


def test_the_custom_drill_in_the_docs_works_and_mirrors():
    (example,) = [block for block in code_blocks(DRILLS) if "class KickoffChase" in block]
    namespace: dict = {}
    exec(compile(example, "docs/drills.md custom drill", "exec"), namespace)
    cls = namespace["KickoffChase"]
    assert issubclass(cls, Drill)
    blue, orange = empty_state(2, 2), empty_state(2, 2)
    blue_info, orange_info = {}, {}
    cls(team=BLUE_TEAM, seed=1).apply(blue, blue_info)
    cls(team=ORANGE_TEAM, seed=1).apply(orange, orange_info)
    assert blue_info["drill"].name == "KickoffChase"
    assert blue_info["drill"].details["note"]
    assert orange_info["drill"].agent == "orange-0"
    assert blue.ball.linear_velocity[1] > 0.0  # rolling at the net blue attacks
    flip = np.array([-1.0, -1.0, 1.0])
    assert np.allclose(orange.ball.position, blue.ball.position * flip)
    assert np.allclose(
        orange.cars["orange-0"].physics.position, blue.cars["blue-0"].physics.position * flip
    )
    for car in blue.cars.values():  # nobody else within 1000 uu of the drill car
        if car is not blue.cars["blue-0"]:
            gap = car.physics.position - blue.cars["blue-0"].physics.position
            assert np.linalg.norm(gap[:2]) >= 1000.0


def test_the_custom_detector_in_the_docs_counts():
    (example,) = [block for block in code_blocks(SKILLS) if "class Supersonic" in block]
    namespace: dict = {}
    exec(compile(example, "docs/skills.md custom detector", "exec"), namespace)
    tracker = namespace["tracker"]
    assert tracker.skills[-1] == "went_supersonic"
    start = quiet_state()
    fast = after(start)
    fast.cars["blue-0"].supersonic_time = 0.1
    tracker.reset(start)
    assert skills_of(tracker.step(fast)) == ["went_supersonic"]
    assert tracker.step(after(fast)) == []


def test_a_drill_that_forgets_the_ball_or_its_car_is_told_so():
    class Forgetful(Drill):
        """Places the ball and nothing else."""

        def _build(self, scene, stage):
            scene.ball((0.0, 0.0, 93.15), (0.0, 100.0, 0.0))

    with pytest.raises(RuntimeError, match="must place the ball and the drill car"):
        Forgetful().apply(empty_state(1, 1), {})
