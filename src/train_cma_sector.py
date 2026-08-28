"""
train_cma_sector.py  --  8-hour CMA-ES with sector-aware speed control (stage 4).

Historical training script, kept unrefactored for the same reason as
`train_cma_5param.py`: it records what was actually run at this stage.  This
is where the single global lookahead gain K was split in two.

New idea: split the lap into two sectors with different K values.

  dist_in_lap = distRaced % 3608.0
  K_eff = K_final  if dist_in_lap >= switch_dist  else K

Motivation: the final corner (~3300 m) requires ~90 km/h entry speed.
At K=2.14 the car arrives too fast and exits through dirt.
K_final < K forces earlier braking into the final corner while keeping
K_main at 2.14 for the rest of the lap (no time loss elsewhere).

Parameters (8):
  A, B, C, K, T, D   -- same 6-param controller (seeded from 122.06 s best)
  K_final             -- corner speed factor for final sector (seed 1.4)
  switch_dist         -- lap distance (m) where K switches (seed 2900 m)

At seed: K_final=1.4 means at track[9]=100 m, adaptive_speed = 140 km/h
(vs 214 → clamped to 200 with K=2.14).  Braking starts ~300 m earlier.

Seeding: reads results/cma8h_overall_best.json at startup.
         Falls back to hard-coded 122.06 s individual.

Budget: 8 hours. popsize=8 (~50-60 gens).

Output:
  logs/cma_sector_log.txt
  results/cma_sector_best.json
  cma_sector_checkpoint.pkl    (deleted on completion)
  results/lap_times_all.csv    (append)
  results/cma8h_overall_best.json  (updated if improved)
"""

import builtins
import csv
import json
import math
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402
import cma  # noqa: E402
from torcs_env import load_torcs_env  # noqa: E402

# Episode limits, carried over unchanged from the 5-parameter stage.
EVAL_STEPS = 40000
MAX_LAPS = 4  # 1 cold + 3 warm

# ── Budget ────────────────────────────────────────────────────────────────────
BUDGET_S = 8 * 3600
POPSIZE = 8
TRACK_LAP_M = 3608.0
PI = math.pi

# ── Fallback seed (122.06 s, 5h_refine gen1 ind8) ────────────────────────────
_FALLBACK = {
    "A": 26.144885,
    "B": 1.553125,
    "C": 200.2917,
    "K": 2.141852,
    "T": 0.008653,
    "D": 0.082639,
    "best_lap_2plus_s": 122.06,
}
# Track analysis (from corkscrew.xml):
#   s46 straight: 3144-3246m (202m) → s48 double-LEFT R=18-20m at 3268-3286m
#   On s46 straight, track[9] ≈ 200m → K=2.14 gives adaptive_speed=428→clamp=200 → no braking
#   K_final must be low enough so K_final×200 < current_speed → triggers braking on the straight
#   K_final=0.9: adaptive_speed=180 → brakes from 200 km/h → arrives at s48 ~60-80 km/h
#   switch_dist=3050: activates at start of s46 straight (after s45 R=57m corners)
_SEED_K_FINAL = 0.9
_SEED_SWITCH = 3050.0

# ── Bounds [lo, hi] per param: A B C K T D K_final switch_dist ───────────────
BOUNDS = [
    [20.0, 32.0],  # A
    [0.8, 2.5],  # B
    [160.0, 220.0],  # C
    [1.5, 2.45],  # K  (main sector)
    [0.003, 0.018],  # T
    [0.0, 0.14],  # D
    [0.3, 2.0],  # K_final (low end allows braking on s46 straight; high end=no effect)
    [2700.0, 3400.0],  # switch_dist: s45 area (2980) to just before s48 (3268)
]

# ── Per-dim initial sigmas (INIT_SIGMA=1.0, so these are absolute std) ───────
# Tight on known-good params; wider on new sector params.
STDS = [0.07, 0.013, 1.5, 0.013, 0.00016, 0.016, 0.20, 80.0]
INIT_SIGMA = 1.0

# ── Files ─────────────────────────────────────────────────────────────────────
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_REPO, "runs")
LOG_PATH = os.path.join(_OUT, "cma_sector_log.txt")
LAP_CSV_PATH = os.path.join(_OUT, "lap_times.csv")
CKPT_PATH = os.path.join(_OUT, "cma_sector_checkpoint.pkl")
BEST_JSON = os.path.join(_OUT, "cma_sector_best.json")
# Seed for the search: the best 6-parameter individual from the previous stage.
OVERALL_BEST = os.path.join(_REPO, "results", "stage3_cma_6param_deadband.json")

# ── Logging ───────────────────────────────────────────────────────────────────
_log_lines: list[str] = []


def _log(line: str) -> None:
    _log_lines.append(line)
    print(line, flush=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))


def _load_existing_log() -> None:
    global _log_lines
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            _log_lines = f.read().splitlines()


# ── Seed selection ────────────────────────────────────────────────────────────
def _pick_seed() -> tuple[list[float], float]:
    """Return (8-param mean, prior_best_s).  Prefers cma8h_overall_best.json."""
    src = _FALLBACK.copy()
    if os.path.exists(OVERALL_BEST):
        with open(OVERALL_BEST) as f:
            d = json.load(f)
        if d.get("best_lap_2plus_s") is not None:
            src = d
    warm = src["best_lap_2plus_s"]
    mean8 = [
        src["A"],
        src["B"],
        src["C"],
        src["K"],
        src["T"],
        src["D"],
        _SEED_K_FINAL,
        _SEED_SWITCH,
    ]
    return mean8, warm


# ── Episode (8-param, sector-aware) ──────────────────────────────────────────
def run_episode_sector(env, A, B, C, K, T, D, K_final, switch_dist):
    """Drive one episode with sector-aware K switching at switch_dist m into lap."""
    env.default_speed = float(np.clip(C, 30.0, 300.0))
    try:
        env.reset(relaunch=False)
    except Exception as e:
        print(f"\n  reset failed ({type(e).__name__}: {e})  -- treating as crash")
        return 0.0, [], 0, True

    final_dist = 0.0
    crashed = False
    step = 0
    prev_lap_time = 0.0
    lap_num = 0
    lap_times: list[tuple[int, float]] = []

    for step in range(EVAL_STEPS):
        try:
            S = env.client.S.d

            last_lap = float(S.get("lastLapTime", 0.0))
            if last_lap > 0.0 and last_lap != prev_lap_time:
                lap_num += 1
                lap_times.append((lap_num, last_lap))
                label = "cold" if lap_num == 1 else "warm"
                print(f"    Lap {lap_num} ({label}): {last_lap:.3f} s")
                prev_lap_time = last_lap
                if lap_num >= MAX_LAPS:
                    break

            angle = float(S.get("angle", 0.0))
            tp = float(S.get("trackPos", 0.0))

            # trackPos deadband
            if abs(tp) <= D:
                tp_eff = 0.0
            else:
                tp_eff = tp - math.copysign(D, tp)
            steer = float(np.clip(angle * A / PI - tp_eff * B, -1.0, 1.0))

            # Sector-aware K: switch to K_final in the final sector
            dist_in_lap = float(S.get("distRaced", 0.0)) % TRACK_LAP_M
            K_eff = K_final if dist_in_lap >= switch_dist else K

            track = S.get("track", [200.0] * 19)
            forward_dist = float(track[9]) if len(track) > 9 else 200.0
            adaptive_speed = float(np.clip(K_eff * forward_dist, 30.0, float(C)))

            speed_x = float(S.get("speedX", 0.0))
            throttle = float(np.clip((adaptive_speed - speed_x) * T, -1.0, 1.0))

            _, _, done, _ = env.step(np.array([steer, throttle], dtype=np.float32))
            final_dist = float(env.client.S.d.get("distRaced", final_dist))

            if done:
                crashed = True
                break
        except (OSError, AttributeError, TypeError) as e:
            print(f"  socket error ({type(e).__name__}: {e}); ending episode")
            crashed = True
            break

    return final_dist, lap_times, step + 1, crashed


# ── Helpers ───────────────────────────────────────────────────────────────────
def _lap_stats(lap_times):
    """(best cold lap, best warm lap) from a list of (lap_number, time)."""
    cold = [t for n, t in lap_times if n == 1]
    warm = [t for n, t in lap_times if n >= 2]
    return (min(cold) if cold else None, min(warm) if warm else None)


def _to_dict(params) -> dict:
    A, B, C, K, T, D, K_final, sw = (float(x) for x in params)
    return {
        "A": A,
        "B": B,
        "C": C,
        "K": K,
        "T": max(T, 0.001),
        "D": max(D, 0.0),
        "K_final": max(K_final, 0.3),
        "switch_dist": float(np.clip(sw, 2000.0, 3500.0)),
    }


def _fitness(warm, cold, dist, steps) -> float:
    if warm is not None:
        return warm
    if cold is not None:
        return cold * 1.2
    if dist > 500 and steps > 100:
        return (steps * 0.02) * TRACK_LAP_M / dist
    return 500.0


def _append_lap_csv(gen, ind, d, lap_times) -> None:
    fields = [
        "gen",
        "ind",
        "lap",
        "time_s",
        "A",
        "B",
        "C",
        "K",
        "T",
        "D",
        "K_final",
        "switch_dist",
    ]
    write_header = not os.path.exists(LAP_CSV_PATH)
    with open(LAP_CSV_PATH, "a", newline="") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(fields)
        for n, t in lap_times:
            w.writerow(
                [
                    f"sector_gen{gen}",
                    ind,
                    n,
                    f"{t:.3f}",
                    f"{d['A']:.4f}",
                    f"{d['B']:.4f}",
                    f"{d['C']:.4f}",
                    f"{d['K']:.4f}",
                    f"{d['T']:.5f}",
                    f"{d['D']:.4f}",
                    f"{d['K_final']:.4f}",
                    f"{d['switch_dist']:.1f}",
                ]
            )


def _maybe_update_overall_best(payload) -> bool:
    warm = payload.get("best_lap_2plus_s")
    if warm is None:
        return False
    if os.path.exists(OVERALL_BEST):
        with open(OVERALL_BEST) as f:
            cur = json.load(f)
        if cur.get("best_lap_2plus_s") is not None and cur["best_lap_2plus_s"] <= warm:
            return False
    with open(OVERALL_BEST, "w") as f:
        json.dump(payload, f, indent=2)
    return True


def _eval(env, params):
    d = _to_dict(params)
    orig_input = builtins.input

    def _no_op(*a, **k):
        raise RuntimeError("auto-skip")

    builtins.input = _no_op
    try:
        return run_episode_sector(
            env,
            d["A"],
            d["B"],
            d["C"],
            d["K"],
            d["T"],
            d["D"],
            d["K_final"],
            d["switch_dist"],
        )
    except RuntimeError:
        return 0.0, [], 0, True
    finally:
        builtins.input = orig_input


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(_OUT, exist_ok=True)
    _load_existing_log()

    seed_mean, prior_best_s = _pick_seed()
    A0, B0, C0, K0, T0, D0, Kf0, sw0 = seed_mean

    print("=" * 72)
    print("CMA-ES SECTOR  (8-param: K_final + switch_dist)")
    print(
        f"  6-param seed : A={A0:.4f} B={B0:.4f} C={C0:.3f} "
        f"K={K0:.4f} T={T0:.5f} D={D0:.4f}"
    )
    print(f"  Sector seed  : K_final={Kf0:.4f}  switch_dist={sw0:.0f} m")
    print(f"  Prior best   : {prior_best_s:.3f} s")
    print(f"  Budget: 8 h  popsize={POPSIZE}")
    print()
    print("Idea: K_final reduces adaptive_speed in the final sector,")
    print(f"  forcing braking ~{3608 - sw0:.0f} m before lap end.")
    print()
    print("Launch wtorcs.exe -> RACE -> NEW RACE -> START, then press Enter.")
    print("=" * 72)
    input()

    t0 = time.time()
    _log(
        f"\n############ SECTOR START {time.strftime('%Y-%m-%d %H:%M:%S')} ############"
    )
    _log(
        f"Seed 6-param: A={A0:.4f} B={B0:.4f} C={C0:.3f} "
        f"K={K0:.4f} T={T0:.5f} D={D0:.4f}"
    )
    _log(f"Sector seed : K_final={Kf0:.4f}  switch_dist={sw0:.0f} m")
    _log(f"Prior best  : {prior_best_s:.3f} s")

    TorcsEnv = load_torcs_env()
    env = TorcsEnv(vision=False, throttle=True, gear_change=False)

    if os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        es = ckpt["es"]
        best_warm = ckpt["best_warm"]
        best_cold = ckpt["best_cold"]
        best_params = ckpt["best_params"]
        gen = ckpt["gen"]
        elapsed_carry = ckpt.get("elapsed_carry", 0.0)
        _log(
            f"=== RESUMED at gen {gen}, best_warm={best_warm}, "
            f"prior_elapsed={elapsed_carry/60:.1f} min ==="
        )
    else:
        es = cma.CMAEvolutionStrategy(
            seed_mean,
            INIT_SIGMA,
            {
                "popsize": POPSIZE,
                "maxiter": 10_000,
                "verbose": -9,
                "CMA_stds": STDS,
                "bounds": [
                    [b[0] for b in BOUNDS],
                    [b[1] for b in BOUNDS],
                ],
            },
        )
        best_warm = None
        best_cold = None
        best_params = list(seed_mean)
        gen = 0
        elapsed_carry = 0.0
        _log(f"=== FRESH START  stds={STDS} ===")

    while not es.stop():
        elapsed = elapsed_carry + (time.time() - t0)
        if elapsed >= BUDGET_S:
            _log(
                f"=== Budget reached ({elapsed/3600:.2f} h); stopping at gen {gen} ==="
            )
            break

        gen += 1
        solutions = es.ask()
        fitnesses = []
        _log(f"\n--- Gen {gen}  elapsed={elapsed/60:.1f} min ---")

        for i, params in enumerate(solutions):
            d = _to_dict(params)
            dist, lap_times, steps, crashed = _eval(env, params)
            cold, warm = _lap_stats(lap_times)
            fit = _fitness(warm, cold, dist, steps)
            fitnesses.append(fit)

            _append_lap_csv(gen, i + 1, d, lap_times)

            tag = "CRASH" if crashed else "OK   "
            warm_s = f"  warm={warm:.3f}" if warm else ""
            cold_s = f"  cold={cold:.3f}" if cold else ""
            # Show sector params to track what CMA explores
            sector_s = f"  Kf={d['K_final']:.3f} sw={d['switch_dist']:.0f}m"
            _log(
                f"  gen={gen} ind={i+1}/{POPSIZE}  "
                f"A={d['A']:5.2f} B={d['B']:5.3f} C={d['C']:6.1f} "
                f"K={d['K']:5.3f} T={d['T']:.5f} D={d['D']:.4f}"
                f"{sector_s}  dist={dist:7.1f}m  fit={fit:6.2f}  {tag}{cold_s}{warm_s}"
            )

            if warm is not None and (best_warm is None or warm < best_warm):
                best_warm = warm
                best_cold = cold
                best_params = list(params)
                payload = {
                    **d,
                    "best_lap_2plus_s": warm,
                    "lap_1_s": cold,
                    "dist": dist,
                    "gen": gen,
                    "ind": i + 1,
                    "phase": "sector",
                    "all_lap_times": [{"lap": n, "time_s": t} for n, t in lap_times],
                }
                with open(BEST_JSON, "w") as f:
                    json.dump(payload, f, indent=2)
                updated = _maybe_update_overall_best(payload)
                marker = ">>> NEW OVERALL BEST" if updated else ">>> phase best"
                kf_note = (
                    f"K_final={d['K_final']:.3f} "
                    f"({'< K = cleaner corner' if d['K_final'] < d['K'] else 'K_final>=K: sector inactive'})"
                )
                _log(
                    f"  {marker}: warm={warm:.3f}s  "
                    f"(seed={prior_best_s:.3f}s  delta={warm-prior_best_s:+.3f}s)  "
                    f"{kf_note}  sw={d['switch_dist']:.0f}m"
                )

        es.tell(solutions, fitnesses)

        elapsed_now = elapsed_carry + (time.time() - t0)
        with open(CKPT_PATH, "wb") as f:
            pickle.dump(
                {
                    "es": es,
                    "best_params": best_params,
                    "best_warm": best_warm,
                    "best_cold": best_cold,
                    "gen": gen,
                    "elapsed_carry": elapsed_now,
                },
                f,
            )

    if os.path.exists(CKPT_PATH):
        os.remove(CKPT_PATH)

    total_h = (time.time() - t0) / 3600
    _log(f"\n############ SECTOR END  total={total_h:.2f} h ############")
    if best_warm:
        d = _to_dict(best_params)
        _log(
            f"Best warm  : {best_warm:.3f} s  "
            f"(seed {prior_best_s:.3f} s, delta {best_warm-prior_best_s:+.3f} s)"
        )
        _log(
            f"6-param    : A={d['A']:.4f} B={d['B']:.4f} C={d['C']:.3f} "
            f"K={d['K']:.4f} T={d['T']:.5f} D={d['D']:.4f}"
        )
        _log(
            f"Sector     : K_final={d['K_final']:.4f}  switch_dist={d['switch_dist']:.1f} m"
        )
        ratio = d["K_final"] / d["K"]
        _log(
            f"K_final/K  : {ratio:.3f}  "
            f"({'sector reduces speed' if ratio < 0.95 else 'sector nearly inactive'})"
        )
    else:
        _log("No warm lap recorded.")
    if os.path.exists(OVERALL_BEST):
        with open(OVERALL_BEST) as f:
            ob = json.load(f)
        _log(f"Overall best now: {ob.get('best_lap_2plus_s')} s")


if __name__ == "__main__":
    main()
