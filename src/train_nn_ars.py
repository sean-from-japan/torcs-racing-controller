"""
train_nn_ars.py -- residual neural network on top of the CMA-ES controller,
trained with Augmented Random Search (ARS).

Starting point: `results/stage4_cma_8param_sector_s35.json`, the 8-parameter sector controller
with the s35 speed cap, measured at 108.692 s.  The network does not replace
that controller; it adds a bounded correction to its throttle:

    throttle = clip(rule(obs) + 0.2 * NN(obs), -1, +1)

    obs(23) -> Linear(32) -> ReLU -> Linear(32) -> ReLU -> Linear(1) -> Tanh

Why this shape rather than a policy learned from scratch:

  * The last layer is zero-initialised, so the agent's *first* evaluation is
    byte-for-byte the CMA-ES controller.  There is no phase where the policy
    cannot complete a lap, which is precisely the cold-start failure that made
    end-to-end RL impractical inside the project's compute budget.
  * ARS optimises the last layer only -- 33 parameters.  Each evaluation costs
    a real TORCS lap, so the search space has to be small enough to make
    progress in single-digit hours, not GPU-days.
  * The three override zones (finish-line sprint, braking sector, post-start
    brake) are excluded from NN control.  They are hand-verified safety
    behaviour and there is no reason to spend samples relearning them.

Usage:
    python src/train_nn_ars.py                        # 6 h budget
    python src/train_nn_ars.py --budget 4 --pairs 2   # shorter run
    python src/train_nn_ars.py --fresh                # ignore the checkpoint

Requires a running TORCS instance; see README.md.
"""

import argparse
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import controller as ctl
from torcs_env import load_torcs_env

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DIR)

BASE_JSON = os.path.join(_REPO, "results", "stage4_cma_8param_sector_s35.json")
CKPT_PATH = os.path.join(_REPO, "nn_ars_checkpoint.pkl")
BEST_PT = os.path.join(_REPO, "models", "nn_ars_s35_best.pt")
BC_PT = os.path.join(_REPO, "models", "bc_keff.pt")
LOG_PATH = os.path.join(_REPO, "logs", "nn_ars_log.txt")

DNF_PENALTY = 300.0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="ARS fine-tuning of the residual NN")
    p.add_argument("--budget", type=float, default=6.0, help="wall-clock hours")
    p.add_argument("--sigma", type=float, default=0.02, help="perturbation scale")
    p.add_argument("--alpha", type=float, default=0.02, help="ARS step size")
    p.add_argument("--pairs", type=int, default=4, help="mirrored pairs per update")
    p.add_argument("--warm", type=int, default=2, help="warm laps per evaluation")
    p.add_argument("--fresh", action="store_true", help="ignore existing checkpoint")
    p.add_argument("--gym-torcs", default=None, help="path to the gym_torcs checkout")
    return p.parse_args(argv)


# -- model ---------------------------------------------------------------------


def make_model():
    import torch.nn as nn

    return nn.Sequential(
        nn.Linear(23, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
        nn.Tanh(),
    )


def get_last(model):
    import numpy as np

    last = model[4]
    return np.concatenate(
        [last.weight.data.cpu().numpy().flatten(), last.bias.data.cpu().numpy()]
    )


def set_last(model, params):
    import torch

    model[4].weight.data = torch.FloatTensor(params[:32].reshape(1, 32))
    model[4].bias.data = torch.FloatTensor(params[32:33])


def nn_output(model, obs):
    import numpy as np
    import torch

    scale = np.array(ctl.observation_scale(), dtype="float32")
    x = torch.from_numpy(np.array(obs, dtype="float32") / scale).unsqueeze(0)
    with torch.no_grad():
        return float(model(x))


# -- rollout -------------------------------------------------------------------


def run_laps(env, model, p, n_laps):
    """Drive `n_laps` laps with the residual controller.  Returns lap times."""
    import numpy as np

    try:
        env.reset(relaunch=False)
    except Exception as exc:  # noqa: BLE001 -- socket layer raises many types
        print("  reset failed (%s)" % exc)
        return []

    lap_times = []
    lap_num = 0
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
                print(
                    "    Lap %d (%s): %.3f s"
                    % (lap_num, "cold" if lap_num == 1 else "warm", last_lap)
                )
                if lap_num >= n_laps:
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

            nn_active = (
                lap_num > 0
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


def evaluate(env, model, p, warm_laps):
    """Fitness = best warm lap.  A run that finishes no warm lap scores DNF."""
    times = run_laps(env, model, p, n_laps=1 + warm_laps)
    warm = times[1:]
    return min(warm) if warm else DNF_PENALTY


# -- logging -------------------------------------------------------------------

_log_lines = []


def log(line):
    _log_lines.append(line)
    print(line, flush=True)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(_log_lines))


# -- main ----------------------------------------------------------------------


def main(argv=None):
    global _log_lines
    import numpy as np
    import torch

    args = parse_args(argv)
    os.makedirs(os.path.dirname(BEST_PT), exist_ok=True)
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, encoding="utf-8") as f:
            _log_lines = f.read().splitlines()

    p = ctl.Params.from_json(BASE_JSON)
    budget_s = args.budget * 3600

    model = make_model()

    # Optional warm start for the hidden layers from a behaviour-cloning run.
    # strict=False: the BC model had a 22-dim input, so only layer 0 is skipped
    # and the 32->32 hidden layer still transfers.  These weights are not
    # distributed with this repository; without them the hidden layers stay at
    # their random initialisation, which is what the published run used as a
    # fallback whenever the file was absent.
    if os.path.exists(BC_PT):
        try:
            state = torch.load(BC_PT, map_location="cpu", weights_only=True)
            missing, _ = model.load_state_dict(state, strict=False)
            print("  loaded BC hidden layers from %s (missing=%s)" % (BC_PT, missing))
        except Exception as exc:  # noqa: BLE001
            print("  WARNING: could not load %s (%s); using random init" % (BC_PT, exc))
    else:
        print("  %s not found; hidden layers stay at random init" % BC_PT)

    # Zero the output layer: ARS starts from the pure CMA-ES controller.
    model[4].weight.data.zero_()
    model[4].bias.data.zero_()
    model.eval()

    if not args.fresh and os.path.exists(CKPT_PATH):
        with open(CKPT_PATH, "rb") as f:
            ckpt = pickle.load(f)
        theta = ckpt["theta"]
        best_time = ckpt["best_time"]
        update = ckpt["update"]
        elapsed_carry = ckpt.get("elapsed_carry", 0.0)
        set_last(model, theta)
        print("=== resumed: update=%d best=%.3f s ===" % (update, best_time))
    else:
        theta = get_last(model).copy()  # all zeros
        best_time = float("inf")
        update = 0
        elapsed_carry = 0.0
        print("=== fresh start (output layer zeroed = pure CMA-ES controller) ===")

    TorcsEnv = load_torcs_env(args.gym_torcs)

    print("=" * 72)
    print("residual NN + ARS   base=%s (%.3f s)" % (BASE_JSON, p_base_time(BASE_JSON)))
    print("  dim=%d  sigma=%s  alpha=%s" % (len(theta), args.sigma, args.alpha))
    print("  pairs=%d  warm_per_eval=%d  budget=%.1f h" % (args.pairs, args.warm, args.budget))
    print("  s35 cap: C_s35=%.1f km/h from %.0f m" % (p.C_s35, p.s35_start))
    print("\nLaunch TORCS -> RACE -> NEW RACE -> START, then press Enter.")
    print("=" * 72)
    input()

    env = TorcsEnv(vision=False, throttle=True, gear_change=False)
    env.default_speed = p.C

    log("\n%s ARS START %s %s" % ("#" * 20, time.strftime("%Y-%m-%d %H:%M:%S"), "#" * 20))

    if update == 0:
        log("=== baseline evaluation (NN zeroed = pure CMA-ES controller) ===")
        set_last(model, theta)
        best_time = evaluate(env, model, p, args.warm)
        torch.save(model.state_dict(), BEST_PT)
        log("  baseline warm lap: %.3f s" % best_time)

    t0 = time.time()

    while elapsed_carry + (time.time() - t0) < budget_s:
        update += 1
        deltas = [np.random.randn(len(theta)) for _ in range(args.pairs)]
        scores_pos, scores_neg = [], []

        for i, delta in enumerate(deltas):
            set_last(model, theta + args.sigma * delta)
            t_pos = evaluate(env, model, p, args.warm)
            scores_pos.append(t_pos)

            set_last(model, theta - args.sigma * delta)
            t_neg = evaluate(env, model, p, args.warm)
            scores_neg.append(t_neg)

            elapsed_now = elapsed_carry + (time.time() - t0)
            log(
                "  u=%d pair=%d/%d  +d=%.3f s  -d=%.3f s  best=%.3f s  %.0f min"
                % (update, i + 1, args.pairs, t_pos, t_neg, best_time, elapsed_now / 60)
            )

            for score, sign in ((t_pos, +1), (t_neg, -1)):
                if score < best_time:
                    best_time = score
                    set_last(model, theta + sign * args.sigma * delta)
                    torch.save(model.state_dict(), BEST_PT)
                    log("  >>> new best %.3f s (saved %s)" % (score, BEST_PT))

        all_scores = scores_pos + scores_neg
        if all(s >= DNF_PENALTY for s in all_scores):
            log("  [every rollout DNF this round -- no gradient step]")
        else:
            # ARS V1: step along the score-weighted average of the perturbations,
            # normalised by the spread of the returns so the step size does not
            # depend on how noisy this particular round was.
            sigma_r = float(np.std(all_scores)) + 1e-6
            grad = np.zeros(len(theta))
            for i in range(args.pairs):
                grad += (scores_pos[i] - scores_neg[i]) * deltas[i]
            theta = theta - (args.alpha / (args.pairs * sigma_r)) * grad

        elapsed_now = elapsed_carry + (time.time() - t0)
        log(
            "  [update %d] round_best=%.3f s  overall_best=%.3f s  elapsed=%.0f min"
            % (update, min(all_scores), best_time, elapsed_now / 60)
        )

        with open(CKPT_PATH, "wb") as f:
            pickle.dump(
                {
                    "theta": theta,
                    "best_time": best_time,
                    "update": update,
                    "elapsed_carry": elapsed_now,
                },
                f,
            )

    log("\n=== done. best warm lap: %.3f s ===" % best_time)
    log("best model: %s" % BEST_PT)
    return 0


def p_base_time(path):
    import json

    with open(path) as f:
        return float(json.load(f)["best_lap_2plus_s"])


if __name__ == "__main__":
    sys.exit(main())
