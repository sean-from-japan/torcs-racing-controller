"""
run_eval.py -- reproduction runner for the final residual-NN agent.

Drives the control law in `controller.py` against a live TORCS instance and
reports the best warm lap.  With `--no-nn` it runs the CMA-ES controller alone,
which is the configuration the committed parameters in
`results/stage4_cma_8param_sector_s35.json` were measured at (108.692 s).

    python src/run_eval.py --no-nn        # CMA-ES controller only
    python src/run_eval.py                # + residual NN (needs weights)

NOTE ON THE WEIGHTS: the 106.630 s figure requires the output-layer weights
that ARS converged on in the recorded run.  Those weights were never committed,
so this script cannot replay that exact lap.  The training loop that produces
them is `train_nn_ars.py`, and it runs from the committed 108.692 s base.
See README.md, "Reproducing this".

Why two runs: the ARS training loop always evaluated a candidate on the
*second* reset of a session, and TORCS behaves differently on the first.
Run 1 drives one lap with the NN off purely to leave the environment in the
state training saw; Run 2 is the measured run, restarted via meta=True the
same way the trainer restarted it.  Skipping Run 1 changes the lap time.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import controller as ctl
from torcs_env import load_torcs_env

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DIR)
BASE_JSON = os.path.join(_REPO, "results", "stage4_cma_8param_sector_s35.json")
DEFAULT_WEIGHTS = os.path.join(_REPO, "models", "nn_ars_s35_best.pt")

DNF_PENALTY = 300.0
RECORD_LAP = 106.630  # reported best; see README on reproducibility


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--warm", type=int, default=2, help="warm laps in the recorded run")
    p.add_argument("--no-nn", action="store_true", help="rule-based only (~108.692 s)")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="residual NN weights")
    p.add_argument("--gym-torcs", default=None, help="path to the gym_torcs checkout")
    p.add_argument("--hold", type=int, default=0, help="seconds to hold after last lap")
    p.add_argument("--retries", type=int, default=5, help="max recorded-run retries")
    return p.parse_args(argv)


def build_model(weights_path):
    """Load the residual network.  Imported lazily so --no-nn needs no torch."""
    import torch
    import torch.nn as nn

    if not os.path.exists(weights_path):
        raise SystemExit(
            "Residual NN weights not found: %s\n"
            "The weights from the recorded 106.630 s run were never archived, so that\n"
            "exact lap cannot be replayed here.  Run with --no-nn to drive the CMA-ES\n"
            "controller (108.692 s), or train a new residual network with\n"
            "train_nn_ars.py and pass it via --weights." % weights_path
        )
    model = nn.Sequential(
        nn.Linear(23, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh(),
    )
    model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
    model.eval()
    return model


def nn_output(model, obs):
    import numpy as np
    import torch

    scale = np.array(ctl.observation_scale(), dtype="float32")
    x = torch.from_numpy(np.array(obs, dtype="float32") / scale).unsqueeze(0)
    with torch.no_grad():
        return float(model(x))


def run_laps(env, model, p, n_laps, hold=0):
    """Drive until `n_laps` laps are recorded or the car stops.  Returns lap times."""
    import numpy as np

    try:
        env.reset(relaunch=False)
    except Exception as exc:  # noqa: BLE001 -- socket layer raises many types
        print("  reset failed (%s)" % exc)
        return []

    lap_times = []
    lap_num = 0  # the NN stays off while this is 0, exactly as during training
    prev_lap_t = 0.0
    prev_tp = 0.0
    nn_ema = 0.0

    for _ in range(500_000):
        try:
            S = env.client.S.d

            last_lap = float(S.get("lastLapTime", 0.0))
            if last_lap > 0.0 and last_lap != prev_lap_t:
                lap_num += 1
                prev_lap_t = last_lap
                lap_times.append(last_lap)
                nn_ema = 0.0
                warm = lap_times[1:]
                marker = " *** fastest so far ***" if warm and last_lap == min(warm) else ""
                print(
                    "    Lap %d (%s): %.3f s%s"
                    % (lap_num, "cold" if lap_num == 1 else "warm", last_lap, marker)
                )
                if lap_num >= n_laps:
                    if hold > 0:
                        print("  [holding %d s for capture]" % hold)
                        time.sleep(hold)
                    break

            dist_in_lap = float(S.get("distRaced", 0.0)) % ctl.TRACK_LAP_M
            speed_x = float(S.get("speedX", 0.0))
            angle = float(S.get("angle", 0.0))
            cur_tp = float(S.get("trackPos", 0.0))
            track = S.get("track", [200.0] * 19)
            forward = float(track[9]) if len(track) > 9 else 200.0

            steer = ctl.steering(
                angle=angle,
                track_pos=cur_tp,
                d_track_pos=cur_tp - prev_tp,
                target_tp=ctl.target_trackpos(dist_in_lap, p),
                A=p.A,
                B=p.B,
                D=p.D,
            )
            prev_tp = cur_tp

            # The NN acts only on warm laps and only outside the override zones.
            # Its EMA advances only while it is active, as in training.
            nn_active = (
                model is not None
                and lap_num > 0
                and not ctl.in_sprint_zone(dist_in_lap, p)
                and not ctl.in_kfinal_zone(dist_in_lap, p)
                and not ctl.in_poststart_brake(dist_in_lap, speed_x)
            )
            if nn_active:
                raw = nn_output(
                    model, ctl.observation(track, speed_x, angle, cur_tp, dist_in_lap)
                )
                nn_ema = ctl.update_nn_ema(nn_ema, raw)

            throttle = ctl.throttle_command(
                dist_in_lap, speed_x, forward, p, nn_ema if nn_active else None
            )

            _, _, done, _ = env.step(np.array([steer, throttle], dtype=np.float32))
            if done:
                print("  [crash]")
                break

        except (OSError, AttributeError, TypeError) as exc:
            print("  socket error (%s)" % exc)
            break

    return lap_times


def main(argv=None):
    args = parse_args(argv)

    if not os.path.exists(BASE_JSON):
        raise SystemExit("controller parameters not found: %s" % BASE_JSON)
    p = ctl.Params.from_json(BASE_JSON)
    with open(BASE_JSON) as f:
        base_meta = json.load(f)

    model = None if args.no_nn else build_model(args.weights)
    n_laps = 1 + args.warm

    TorcsEnv = load_torcs_env(args.gym_torcs)

    print("=" * 70)
    print("run_eval.py")
    print("  parameters : %s (measured %.3f s)" % (BASE_JSON, base_meta["best_lap_2plus_s"]))
    print("  residual NN: %s" % (args.weights if model else "disabled"))
    print("  target     : %.3f s" % (RECORD_LAP if model else base_meta["best_lap_2plus_s"]))
    print("  run 1      : 1 lap, NN off, to match the training reset sequence")
    print("  run 2      : %d laps (1 carry-over + %d warm), measured" % (n_laps, args.warm))
    print()
    print("Launch TORCS -> RACE -> NEW RACE -> START, then press Enter.")
    print("=" * 70)
    input()

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env.default_speed = p.C

    print("\nRun 1 -- 1 lap, NN off ...")
    r1 = run_laps(env, None, p, n_laps=1)
    print("  run 1 %s" % ("complete (%.3f s)" % r1[0] if r1 else "did not finish a lap"))
    time.sleep(2)

    lap_times = []
    score = DNF_PENALTY
    t0 = time.time()
    for attempt in range(1, args.retries + 1):
        print("\nRun 2 attempt %d/%d -- %d laps, NN %s\n"
              % (attempt, args.retries, n_laps, "on" if model else "off"))
        lap_times = run_laps(env, model, p, n_laps=n_laps, hold=args.hold)
        warm = lap_times[1:]
        score = min(warm) if warm else DNF_PENALTY
        if score < DNF_PENALTY:
            break
        print("  attempt %d did not complete a warm lap; restarting" % attempt)

    warm = lap_times[1:]
    print()
    print("=" * 70)
    print("RESULTS")
    print("  all laps      : %s" % ["%.3f" % t for t in lap_times])
    print("  warm laps     : %s" % ["%.3f" % t for t in warm])
    print("  best warm lap : %.3f s" % score)
    reference = RECORD_LAP if model else base_meta["best_lap_2plus_s"]
    print("  vs reference  : %+.3f s" % (score - reference))
    print("  elapsed       : %.1f min" % ((time.time() - t0) / 60))
    print("=" * 70)
    return 0 if score < DNF_PENALTY else 1


if __name__ == "__main__":
    sys.exit(main())
