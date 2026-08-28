"""
train_cma_5param.py -- CMA-ES over the 5-parameter controller (stage 3).

Historical training script, kept as the clearest illustration of the search
loop.  It is deliberately *not* refactored onto `controller.py`: this stage
predates the racing-line profiles, the sector switch and the s35 cap, and
rewriting it against the final control law would misrepresent what was
actually run.  The final control law lives in `controller.py`; this file shows
how its parameters were found.

Seeded from the 128 s individual.

After 6-param (A,B,C,K,Ta,Tb) split throttle regressed to 144s, we revert to
the 5-param formula that produced the project-best 128.102 s lap (cma2_gen1
ind 2, 2026-04-23).  Seed = the exact 128s parameters; tight per-dim sigmas
for local refinement (Gemini Pro 2026-04-24 advice).

CMA-ES searches 5 parameters (A, B, C, K, T):
    A  -- steer angle gain
    B  -- steer trackPos gain
    C  -- max speed cap (km/h)
    K  -- corner speed factor: adaptive_speed = clip(K * min(track[6:13]), 90, C)
    T  -- throttle gain (same for accel and brake):
          throttle = clip((adaptive_speed - speedX) * T, -1, 1)

Off-track (|trackPos|>1): throttle=0.7.
Termination: gym_torcs distRaced-based stall (<2m in 200 steps after step 3000).
Fitness: warm lap time (lower = better); extrapolated if no lap completed.

Output files:
    cma2_best.json          -- best params (written on new best)
    cma2_best_laptime.json  -- best by warm lap time
    cma2_log.txt            -- episode log
    cma2_checkpoint.pkl     -- resume checkpoint

Usage:
  python agent.py
"""

import csv
import json
import os
import math
import pickle
import sys
import numpy as np
import cma

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from torcs_env import load_torcs_env

PI = math.pi
EVAL_STEPS = 40000
MAX_LAPS = 4  # 1 cold + 3 warm; stop here if no crash

POPSIZE = 9  # 5-dim CMA default ~8; slight bump for robustness
MAX_GENERATIONS = 15

# Seed = 128.102s individual (cma2_gen1 ind 2, 2026-04-23).
# 5-param: A, B, C, K, T
INIT_MEAN = [26.5183, 1.6171, 217.23, 1.4556, 0.01326]
INIT_SIGMA = 1.0
# Per-dim sigmas (Gemini Pro 2026-04-24): tight local search around 128s seed.
# A ~3.8%, B ~3.1%, C ~4.6%, K ~3.4%, T ~3.8%.
CMA_STDS = [1.0, 0.05, 10.0, 0.05, 0.0005]

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "runs")
RESULT_PATH = os.path.join(_OUT, "cma5_best.json")
LAPTIME_RESULT_PATH = os.path.join(_OUT, "cma5_best_laptime.json")
LOG_PATH = os.path.join(_OUT, "cma5_log.txt")
LAP_CSV_PATH = os.path.join(_OUT, "lap_times.csv")
CHECKPOINT_PATH = os.path.join(_OUT, "cma5_checkpoint.pkl")

TRACK_LAP_M = 3608.0


def run_episode(
    env,
    A: float,
    B: float,
    C: float,
    K: float,
    T: float,
) -> tuple[float, list[tuple[int, float]], int, bool]:
    """Drive one episode. Returns (dist, lap_times, steps, crashed)."""
    env.default_speed = float(np.clip(C, 30.0, 300.0))
    try:
        env.reset(relaunch=False)
    except Exception as e:
        print(f"\n  reset failed ({type(e).__name__}: {e})")
        print(
            "  Restart TORCS manually (RACE -> NEW RACE -> START), then press Enter..."
        )
        input()
        env.initial_reset = True
        env.reset(relaunch=False)

    final_dist = 0.0
    crashed = False
    step = 0
    prev_lap_time = 0.0
    lap_num = 0
    lap_times: list[tuple[int, float]] = []
    # CRASH means the episode ended abnormally (gym_torcs stall-terminate) AND
    # the car had accumulated damage.  Grazing walls while still finishing laps
    # is NOT a crash -- the 128s record was set with wall contact.
    start_damage = float(env.client.S.d.get("damage", 0.0))

    for step in range(EVAL_STEPS):
        try:
            S = env.client.S.d

            last_lap = float(S.get("lastLapTime", 0.0))
            if last_lap > 0.0 and last_lap != prev_lap_time:
                lap_num += 1
                lap_times.append((lap_num, last_lap))
                label = "cold" if lap_num == 1 else "warm"
                print(f"    Lap {lap_num} ({label}): {last_lap:.2f} s")
                prev_lap_time = last_lap
                if lap_num >= MAX_LAPS:
                    break

            steer = float(
                np.clip(
                    float(S.get("angle", 0.0)) * A / PI
                    - float(S.get("trackPos", 0.0)) * B,
                    -1.0,
                    1.0,
                )
            )

            # 128s-era control (commit 68d6d54): single adaptive-speed formula
            # for both on- and off-track.  No off-track branch, no steering
            # damping.  Off-track sensor returns track[9]=-1 -> adaptive clamps
            # to 30 km/h -> gentle brake while full steer drags the car back.
            track = S.get("track", [200.0] * 19)
            forward_dist = float(track[9]) if len(track) > 9 else 200.0
            adaptive_speed = float(np.clip(K * forward_dist, 30.0, float(C)))
            speed_x = float(S.get("speedX", 0.0))
            throttle = float(np.clip((adaptive_speed - speed_x) * T, -1.0, 1.0))

            _, _, done, _ = env.step(np.array([steer, throttle], dtype=np.float32))
            final_dist = float(env.client.S.d.get("distRaced", final_dist))

            if done:
                cur_damage = float(env.client.S.d.get("damage", 0.0))
                crashed = (cur_damage - start_damage) > 0.0
                break
        except (OSError, AttributeError, TypeError) as e:
            print(f"  socket error ({type(e).__name__}: {e}); ending episode")
            crashed = True
            break

    return final_dist, lap_times, step + 1, crashed


def _lap_stats(lap_times: list[tuple[int, float]]) -> tuple:
    times_1 = [t for n, t in lap_times if n == 1]
    times_2p = [t for n, t in lap_times if n >= 2]
    return (
        min(times_1) if times_1 else None,
        min(times_2p) if times_2p else None,
    )


def _append_lap_csv(
    gen: int,
    ind: int,
    A: float,
    B: float,
    C: float,
    K: float,
    T: float,
    lap_times: list[tuple[int, float]],
) -> None:
    write_header = not os.path.exists(LAP_CSV_PATH)
    with open(LAP_CSV_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["gen", "ind", "lap", "time_s", "A", "B", "C", "K", "T"])
        for lap_num, time_s in lap_times:
            writer.writerow(
                [
                    f"cma2d_gen{gen}",
                    ind,
                    lap_num,
                    f"{time_s:.3f}",
                    f"{A:.4f}",
                    f"{B:.4f}",
                    f"{C:.4f}",
                    f"{K:.4f}",
                    f"{T:.5f}",
                ]
            )


def main() -> None:
    os.makedirs(_OUT, exist_ok=True)
    TorcsEnv = load_torcs_env()
    print("=" * 60)
    print("CMA-ES v2d: 5-param single-T, seeded from 128.102s individual")
    print(
        f"Init: A={INIT_MEAN[0]:.4f} B={INIT_MEAN[1]:.4f} C={INIT_MEAN[2]:.2f} "
        f"K={INIT_MEAN[3]:.4f} T={INIT_MEAN[4]:.5f}"
    )
    print("Manually launch wtorcs.exe -> RACE -> NEW RACE -> START.")
    print("When the car is on track, press Enter here.")
    print("=" * 60)
    input()

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        es = ckpt["es"]
        best_params = ckpt["best_params"]
        best_dist = ckpt["best_dist"]
        best_lap_2plus_overall = ckpt.get("best_lap_2plus_overall")
        best_lap_1_at_best = ckpt.get("best_lap_1_at_best")
        log_lines = ckpt["log_lines"]
        gen = ckpt["gen"]
        if len(best_params) != 5 or es.N != 5:
            raise RuntimeError(
                "Incompatible checkpoint. Delete cma2_checkpoint.pkl and restart."
            )
        A0, B0, C0, K0, T0 = best_params
        print(
            f"Resumed: gen={gen}, best_lap2+="
            f"{best_lap_2plus_overall:.2f}s  "
            f"A={A0:.4f} B={B0:.4f} C={C0:.2f} K={K0:.4f} T={T0:.5f}"
        )
    else:
        es = cma.CMAEvolutionStrategy(
            INIT_MEAN,
            INIT_SIGMA,
            {
                "popsize": POPSIZE,
                "maxiter": MAX_GENERATIONS,
                "verbose": -9,
                "CMA_stds": CMA_STDS,
            },
        )
        best_params = list(INIT_MEAN)
        best_dist = -float("inf")
        best_lap_2plus_overall = None
        best_lap_1_at_best = None
        log_lines = []
        gen = 0

    while not es.stop():
        gen += 1
        solutions = es.ask()
        fitnesses = []
        print(f"\n=== Generation {gen}/{MAX_GENERATIONS} ===")

        for i, params in enumerate(solutions):
            A = float(params[0])
            B = float(params[1])
            C = float(params[2])
            K = float(params[3])
            T = max(float(params[4]), 0.003)

            dist, lap_times, steps, crashed = run_episode(env, A, B, C, K, T)
            best_lap_1, best_lap_2plus = _lap_stats(lap_times)

            # Fitness: warm lap time (lower=better); extrapolate if no lap completed
            if best_lap_2plus is not None:
                fitness = best_lap_2plus
            elif best_lap_1 is not None:
                fitness = best_lap_1 * 1.2
            elif dist > 500 and steps > 100:
                elapsed_s = steps * 0.02
                fitness = elapsed_s * TRACK_LAP_M / dist
            else:
                fitness = 500.0
            fitnesses.append(fitness)

            _append_lap_csv(gen, i + 1, A, B, C, K, T, lap_times)

            tag = "CRASH" if crashed else "OK   "
            warm_str = f"  lap2+={best_lap_2plus:.1f}s" if best_lap_2plus else ""
            cold_str = f"  lap1={best_lap_1:.1f}s" if best_lap_1 else ""
            line = (
                f"  gen={gen} ind={i+1}/{POPSIZE}"
                f"  A={A:5.2f} B={B:5.3f} C={C:6.1f} K={K:5.3f} T={T:.5f}"
                f"  dist={dist:7.1f}m  steps={steps:5d}  {tag}{cold_str}{warm_str}"
            )
            print(line)
            log_lines.append(line)

            # Track bests
            is_new_lap2 = best_lap_2plus is not None and (
                best_lap_2plus_overall is None
                or best_lap_2plus < best_lap_2plus_overall
            )
            is_new_best = is_new_lap2 or (best_lap_2plus is None and dist > best_dist)

            if is_new_lap2:
                best_lap_2plus_overall = best_lap_2plus
                with open(LAPTIME_RESULT_PATH, "w") as f:
                    json.dump(
                        {
                            "A": A,
                            "B": B,
                            "C": C,
                            "K": K,
                            "T": T,
                            "best_lap_2plus_s": best_lap_2plus,
                            "lap_1_s": best_lap_1,
                            "dist": dist,
                            "gen": gen,
                            "all_lap_times": [
                                {"lap": n, "time_s": t} for n, t in lap_times
                            ],
                        },
                        f,
                        indent=2,
                    )
                print(f"  >>> new best lap 2+: {best_lap_2plus:.2f} s")

            if is_new_best:
                best_dist = dist
                best_params = [A, B, C, K, T]
                best_lap_1_at_best = best_lap_1
                with open(RESULT_PATH, "w") as f:
                    json.dump(
                        {
                            "A": A,
                            "B": B,
                            "C": C,
                            "K": K,
                            "T": T,
                            "dist": dist,
                            "lap_1_s": best_lap_1,
                            "best_lap_2plus_s": best_lap_2plus,
                            "steps": steps,
                            "crashed": crashed,
                            "gen": gen,
                            "all_lap_times": [
                                {"lap": n, "time_s": t} for n, t in lap_times
                            ],
                        },
                        f,
                        indent=2,
                    )
                print(
                    f"  >>> new best: lap2+={best_lap_2plus:.2f}s  "
                    f"A={A:.4f} B={B:.4f} C={C:.2f} K={K:.4f} T={T:.5f}"
                    if best_lap_2plus
                    else f"  >>> new best dist: {dist:.1f}m  A={A:.4f} B={B:.4f} C={C:.2f} K={K:.4f} T={T:.5f}"
                )

        es.tell(solutions, fitnesses)
        with open(LOG_PATH, "w") as f:
            f.write("\n".join(log_lines))
        with open(CHECKPOINT_PATH, "wb") as f:
            pickle.dump(
                {
                    "es": es,
                    "best_params": best_params,
                    "best_dist": best_dist,
                    "best_lap_2plus_overall": best_lap_2plus_overall,
                    "best_lap_1_at_best": best_lap_1_at_best,
                    "log_lines": log_lines,
                    "gen": gen,
                },
                f,
            )

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)

    print("\n" + "=" * 60)
    A, B, C, K, T = best_params
    print(f"Done. A={A:.4f} B={B:.4f} C={C:.2f} K={K:.4f} T={T:.5f}")
    if best_lap_2plus_overall:
        print(f"Best lap 2+ time: {best_lap_2plus_overall:.2f} s")
    else:
        print("No warm lap completed.")


if __name__ == "__main__":
    main()
