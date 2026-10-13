"""evaluate_drill: per-stage outcome rates of a policy on a drill."""

from __future__ import annotations

import pytest
from rlgym.rocket_league.state_mutators import KickoffMutator

from rocket_drills import DrillMix, DrillReport, evaluate_drill, scripted
from rocket_drills.drills import FlipResetChain, Save, Takeoff
from rocket_drills.evaluation import OUTCOMES, StageResult


def result(stage, starts, **rates):
    return StageResult("FlipResetChain", stage, starts, {o: rates.get(o, 0.0) for o in OUTCOMES})


def test_a_report_pools_stages_by_their_starts_and_formats_as_a_table():
    report = DrillReport(
        seconds=5.0,
        results=(result("finish", 30, goal=0.9, shot=1.0), result("reset", 10, goal=0.1)),
    )
    assert report.rate("goal", "finish") == pytest.approx(0.9)
    assert report.rate("goal") == pytest.approx((0.9 * 30 + 0.1 * 10) / 40)
    with pytest.raises(KeyError, match="unknown outcome"):
        report.rate("assist")
    with pytest.raises(ValueError, match="no starts"):
        report.rate("goal", "takeoff")
    lines = report.format().splitlines()
    assert lines[0] == "share of starts with each outcome within 5 s"
    assert lines[1].split() == ["stage", "starts", *OUTCOMES]
    assert lines[2].split()[:4] == ["FlipResetChain", "/", "finish", "30"]
    assert lines[2].split()[-2] == "0.900"  # goal is the last column but one
    plain = DrillReport(1.0, (StageResult("Aerial", "default", 5, dict.fromkeys(OUTCOMES, 0.0)),))
    assert plain.format().splitlines()[2].split()[:2] == ["Aerial", "5"]  # no "/ default"


@pytest.mark.sim
def test_a_dodge_from_the_finish_stage_scores_and_idling_does_not(engine):
    def run(policy):
        drill = FlipResetChain(stages=["reset", "finish"], seed=5)
        return evaluate_drill(drill, policy, episodes=60, seconds=3.0, engine=engine)

    dodging, idling = run(scripted.forward_dodge), run(scripted.idle)
    assert {r.stage for r in dodging.results} == {"reset", "finish"}
    assert sum(r.starts for r in dodging.results) == 60
    assert all(set(r.rates) == set(OUTCOMES) for r in dodging.results)
    assert dodging.rate("dodge", "finish") == 1.0
    assert dodging.rate("goal", "finish") > 0.6 > 0.2 > idling.rate("goal", "finish")
    assert dodging.rate("shot", "finish") > 0.8
    assert idling.rate("dodge") == 0.0
    assert idling.rate("flip_reset", "reset") > 0.9  # coasting is enough for the reset
    assert idling.rate("flip_reset", "finish") == 0.0
    # (Wheels on the ball are not always a touch: RocketSim counts contact with the body only.)
    assert 0.2 < idling.rate("touch", "reset") == idling.rate("first_touch", "reset")
    assert idling.rate("airborne") == 1.0  # both stages start in the air


@pytest.mark.sim
def test_takeoff_baselines_differ_by_what_they_do(engine):
    def run(policy):
        return evaluate_drill(Takeoff(seed=3), policy, episodes=60, seconds=2.5, engine=engine)

    jumping, driving, idling = (
        run(scripted.boosted_jump),
        run(scripted.drive_at_ball),
        run(scripted.idle),
    )
    assert jumping.rate("airborne") == 1.0
    assert driving.rate("airborne") < 0.1
    assert jumping.rate("air_touch") > 0.15 > driving.rate("air_touch")
    assert idling.rate("air_touch") == 0.0  # (the ball comes down onto some coasting cars)
    assert idling.rate("airborne") == 0.0
    assert {r.stage for r in jumping.results} == {"attacking", "defending"}


@pytest.mark.sim
def test_saves_are_scored_for_the_defender_and_mixes_get_a_row_per_drill(engine):
    report = evaluate_drill(
        Save(seed=7, stages="in_goal"),
        scripted.drive_at_ball,
        episodes=40,
        seconds=4.0,
        team_size=(2, 1),
        engine=engine,
    )
    assert report.rate("conceded") + report.rate("save") > 0.8  # it goes in or is kept out
    assert 0.0 < report.rate("save") <= report.rate("touch")
    assert report.rate("goal") < 0.1

    mix = DrillMix({KickoffMutator(): 1.0, Takeoff(): 1.0}, seed=2)
    mixed = evaluate_drill(mix, scripted.drive_at_ball, episodes=30, seconds=1.0, engine=engine)
    assert {(r.drill, r.stage) for r in mixed.results} == {
        ("KickoffMutator", None),
        ("Takeoff", "attacking"),
        ("Takeoff", "defending"),
    }
    assert sum(r.starts for r in mixed.results) == 30
    assert "KickoffMutator" in mixed.format()


def test_evaluate_drill_checks_its_arguments_before_starting_the_simulator():
    for bad in ({"episodes": 0}, {"tick_skip": 0}, {"seconds": 0.0}):
        with pytest.raises(ValueError, match="at least 1"):
            evaluate_drill(Takeoff(), scripted.idle, **bad)
