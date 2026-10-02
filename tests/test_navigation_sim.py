"""Rocky's driving (movement/navigate.py) in the simulator (sim/world.py), every
scenario in sim/scenarios/ x SEEDS (each seed: a slightly different start,
different sensor noise and glitches).

The hard rule, over every run: never fall off. Collisions: none, except a
scenario may allow rare soft bumps ("max_bump_rate" -- exploring a floor at
~10 cm/s, where the narrow ultrasonic can't see things beside the path). Per
scenario: "reach" scenarios must mostly succeed, "refuse" ones must never
claim success. Writes a picture of each scenario's first run to sim/out/.

Pure stdlib -- runs on the Mac too:
    cd openbot && python3 -m tests.test_navigation_sim [scenario-name ...]
"""
import math
import random
import sys
from pathlib import Path

from movement import navigate
from sim.world import FRONT_REACH, SimBody, World

TRUE_ARRIVAL_CM = 35  # a "reached it" must end with the bumper this close to the target's centre

SEEDS = 25
MIN_REACH = 0.9  # "reach" scenarios: at least this share of seeds must get there
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "sim" / "out"


def run(path: Path, seed: int):
    world, spec = World.load(path, seed)
    body = SimBody(world)
    task = spec["task"]
    if "approach" in task:
        outcome = navigate.approach(body, task["approach"])
    else:
        outcome = navigate.explore(body, task["explore"], rng=random.Random(seed))
    return world, spec, outcome


def main(names: list[str]) -> None:
    OUT.mkdir(exist_ok=True)
    failures = []
    paths = sorted((HERE / "sim" / "scenarios").glob("*.json"))
    for path in paths:
        if names and path.stem not in names:
            continue
        falls = hits = reached = false_claims = 0
        reasons: dict[str, int] = {}
        steps = []
        for seed in range(SEEDS):
            world, spec, outcome = run(path, seed)
            falls += world.fell
            hits += world.collided
            reached += outcome.done
            if outcome.done and "approach" in spec["task"]:
                target = next(o for o in world.obstacles if o.label and spec["task"]["approach"] in o.label)
                fx, fy = world.pose.point(FRONT_REACH)
                false_claims += math.hypot(fx - target.center()[0], fy - target.center()[1]) > TRUE_ARRIVAL_CM
            reasons[outcome.reason] = reasons.get(outcome.reason, 0) + 1
            steps.append(outcome.steps)
            if seed == 0:
                title = f"{path.stem}: {outcome.reason} ({outcome.steps} steps, {world.time:.0f}s)"
                (OUT / f"{path.stem}.svg").write_text(world.svg(title))
        expect = spec["expect"]
        top = ", ".join(f"{n}x {r}" for r, n in sorted(reasons.items(), key=lambda kv: -kv[1])[:3])
        reached -= false_claims
        print(f"{path.stem:18} {expect:7} reached {reached:2}/{SEEDS}  falls {falls}  hits {hits}  false {false_claims}  "
              f"median steps {sorted(steps)[len(steps) // 2]:3}   [{top}]")
        if falls:
            failures.append(f"{path.stem}: {falls} falls")
        if false_claims:
            failures.append(f"{path.stem}: claimed to reach the target {false_claims}x while still far from it")
        if hits > spec.get("max_bump_rate", 0.0) * SEEDS:
            failures.append(f"{path.stem}: {hits} collisions (allowed {spec.get('max_bump_rate', 0.0):.0%} of runs)")
        if expect == "reach" and reached < spec.get("min_reach", MIN_REACH) * SEEDS:
            failures.append(f"{path.stem}: reached only {reached}/{SEEDS}")
        if expect == "refuse" and reached:
            failures.append(f"{path.stem}: claimed to reach an unreachable target {reached}x")
    print(f"pictures: {OUT}/")
    assert not failures, "\n".join(failures)
    print("test_navigation_sim: ok")


if __name__ == "__main__":
    main(sys.argv[1:])
