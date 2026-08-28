"""
controller.py -- the final Corkscrew control law, as pure functions.

Extracted verbatim (behaviour-preserving) from the training script
`train_nn_ars.py` and the reproduction runner `run_eval.py`, which each
carried their own copy of the same logic.  Nothing here touches TORCS, a
socket, PyTorch or NumPy, so the whole control law is unit-testable on a
bare Python install.

The controller is two subsystems, kept separate so that the CMA-ES search
space stays interpretable:

  steering  <- angle gain A, lateral gain B, deadband D, racing-line target
  speed     <- lookahead gain K (sector-switched), speed cap C, throttle gain T

On top of the speed subsystem sits an optional residual neural-network
correction; see `residual_throttle`.  The NN is *not* part of this module --
`residual_throttle` only takes its scalar output, which keeps torch out of
the dependency set for everything except the two scripts that need it.

Parameter names match the JSON files in `results/`.
"""

import json
import math
from typing import Dict, List, Optional, Sequence

PI = math.pi

# Corkscrew lap length as reported by TORCS `distRaced` (m).
TRACK_LAP_M = 3608.0

# Damping on the rate of change of trackPos.  Fixed by hand, never optimised.
D_LAT_FIXED = 4.0

# Residual NN mixing weight and the EMA smoothing applied to its output.
RESIDUAL_SCALE = 0.2
EMA_ALPHA = 0.3

# Racing-line zone for the s20+s21 Corkscrew complex (m into the lap).
D_CORK_RAMP = 900.0
D_CORK_APP = 1420.0
D_CORK_APEX = 1516.0
D_CORK_EXIT = 1640.0

# Racing-line zone for the s35 chicane (m into the lap).
D_S35_RAMP = 2300.0
D_S35_APPROACH = 2411.0
D_S35_APEX = 2441.0
D_S35_EXIT = 2510.0

# Post-start hard brake: the grid start carries too much speed into s1.
POSTSTART_DIST_M = 500.0
POSTSTART_SPEED_KMH = 140.0

# Lower clamp on the lookahead speed target (km/h).
V_TARGET_MIN = 30.0


def _clip(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else (hi if x > hi else x)


class Params:
    """The 12 optimised numbers plus the fixed zone bounds they were found with.

    Field names are the JSON keys written by the CMA-ES scripts, so
    `Params.from_json("results/stage4_cma_8param_sector_s35.json")` round-trips exactly.
    """

    FIELDS = (
        "A",
        "B",
        "C",
        "K",
        "T",
        "D",
        "K_final",
        "switch_dist",
        "back_dist",
        "tp_approach",
        "tp_apex",
        "C_s35",
        "s35_start",
        "tp_s35_approach",
        "tp_s35_apex",
    )

    def __init__(self, **kwargs: float) -> None:
        missing = [f for f in self.FIELDS if f not in kwargs]
        if missing:
            raise KeyError("missing controller parameters: %s" % ", ".join(missing))
        for f in self.FIELDS:
            setattr(self, f, float(kwargs[f]))

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "Params":
        return cls(**{f: d[f] for f in cls.FIELDS})

    @classmethod
    def from_json(cls, path: str) -> "Params":
        with open(path) as f:
            return cls.from_dict(json.load(f))


def target_trackpos_corkscrew(dist_in_lap: float, tp_app: float, tp_apex: float) -> float:
    """Racing-line target for trackPos through the s20+s21 complex.

    Piecewise linear: ramp in to `tp_app`, cross to `tp_apex` at the apex,
    then unwind to the centre line.  Zero everywhere outside the zone.
    """
    d = dist_in_lap
    if d <= D_CORK_RAMP:
        return 0.0
    if d < D_CORK_APP:
        return (d - D_CORK_RAMP) / (D_CORK_APP - D_CORK_RAMP) * tp_app
    if d < D_CORK_APEX:
        t = (d - D_CORK_APP) / (D_CORK_APEX - D_CORK_APP)
        return tp_app + t * (tp_apex - tp_app)
    if d < D_CORK_EXIT:
        return tp_apex * (1.0 - (d - D_CORK_APEX) / (D_CORK_EXIT - D_CORK_APEX))
    return 0.0


def target_trackpos_s35(dist_in_lap: float, tp_app: float, tp_apex: float) -> float:
    """Racing-line target for trackPos through the s35 chicane.

    Same shape as the Corkscrew profile over a shorter, tighter zone.
    """
    d = dist_in_lap
    if d < D_S35_RAMP or d >= D_S35_EXIT:
        return 0.0
    if d < D_S35_APPROACH:
        return (d - D_S35_RAMP) / (D_S35_APPROACH - D_S35_RAMP) * tp_app
    if d < D_S35_APEX:
        t = (d - D_S35_APPROACH) / (D_S35_APEX - D_S35_APPROACH)
        return tp_app + t * (tp_apex - tp_app)
    return tp_apex * (1.0 - (d - D_S35_APEX) / (D_S35_EXIT - D_S35_APEX))


def target_trackpos(dist_in_lap: float, p: Params) -> float:
    """Combined racing line.  The two zones do not overlap, so they sum."""
    return target_trackpos_corkscrew(
        dist_in_lap, p.tp_approach, p.tp_apex
    ) + target_trackpos_s35(dist_in_lap, p.tp_s35_approach, p.tp_s35_apex)


def steering(
    angle: float,
    track_pos: float,
    d_track_pos: float,
    target_tp: float,
    A: float,
    B: float,
    D: float,
) -> float:
    """Steering command in [-1, +1].

    `D` is a deadband on the lateral error: inside it the lateral term is
    dropped entirely.  Without the deadband, trackPos sensor noise produced
    micro-corrections on the straights (a visible zigzag); silencing them was
    worth ~0.7 s a lap.  `d_track_pos` is the per-step change in trackPos and
    damps the resulting oscillation.
    """
    err = track_pos - target_tp
    eff = 0.0 if abs(err) <= D else err - math.copysign(D, err)
    return _clip(angle * A / PI - eff * B - d_track_pos * D_LAT_FIXED, -1.0, 1.0)


def effective_k(dist_in_lap: float, p: Params) -> float:
    """Lookahead gain for the current sector.

    Three zones per lap:
      [0, switch_dist)            main sector, K
      [switch_dist, back_dist)    braking sector into the s48 hairpin, K_final
      [back_dist, lap end]        finish straight, K again (the sprint zone,
                                  where throttle is overridden to full anyway)

    A single global K could not serve both: the value that is quick everywhere
    else arrives at the R=18-20 m final corner far too fast.
    """
    if dist_in_lap >= p.back_dist:
        return p.K
    if dist_in_lap >= p.switch_dist:
        return p.K_final
    return p.K


def target_speed(dist_in_lap: float, forward_dist: float, p: Params) -> float:
    """Speed target (km/h) from the dead-ahead range sensor, with the s35 cap.

    `forward_dist` is track[9], the distance to the track edge straight ahead.
    The s35 cap is the fix for the chicane crashes: a sector-local ceiling
    rather than a global speed reduction, so the rest of the lap is untouched.
    """
    v = _clip(effective_k(dist_in_lap, p) * forward_dist, V_TARGET_MIN, p.C)
    if p.s35_start <= dist_in_lap <= D_S35_EXIT:
        v = min(v, p.C_s35)
    return v


def rule_throttle(v_target: float, speed_x: float, T: float) -> float:
    """Proportional throttle/brake on the speed error, in [-1, +1]."""
    return _clip((v_target - speed_x) * T, -1.0, 1.0)


def in_sprint_zone(dist_in_lap: float, p: Params) -> bool:
    """Finish straight: 225 m of s50, taken at full throttle."""
    return dist_in_lap >= p.back_dist


def in_kfinal_zone(dist_in_lap: float, p: Params) -> bool:
    """Braking sector into the final hairpin."""
    return p.switch_dist <= dist_in_lap < p.back_dist


def in_poststart_brake(dist_in_lap: float, speed_x: float) -> bool:
    """Grid start carries too much speed into the first corner complex."""
    return dist_in_lap < POSTSTART_DIST_M and speed_x > POSTSTART_SPEED_KMH


def update_nn_ema(previous: float, raw: float, alpha: float = EMA_ALPHA) -> float:
    """Smooth the NN output.  Raw per-step corrections are too jittery to drive."""
    return alpha * raw + (1.0 - alpha) * previous


def residual_throttle(rule: float, nn_ema: float, scale: float = RESIDUAL_SCALE) -> float:
    """Rule-based throttle plus a bounded NN correction, clipped to [-1, +1].

    The NN is only ever allowed to nudge: at scale 0.2 a saturated output moves
    the command by 0.2.  Zero-initialising the NN's last layer therefore makes
    the residual agent start out *identical* to the CMA-ES controller, so ARS
    never has to survive a from-scratch policy that cannot finish a lap.
    """
    return _clip(rule + scale * nn_ema, -1.0, 1.0)


def throttle_command(
    dist_in_lap: float,
    speed_x: float,
    forward_dist: float,
    p: Params,
    nn_ema: Optional[float] = None,
) -> float:
    """Full speed subsystem: rule output, residual mixing, then hard overrides.

    `nn_ema` is None whenever the NN must not act -- on the cold lap, and
    inside the three override zones.  The overrides are applied last and
    unconditionally: whatever the NN asks for, the sprint zone is full throttle
    and the post-start zone is full brake.
    """
    v_target = target_speed(dist_in_lap, forward_dist, p)
    rule = rule_throttle(v_target, speed_x, p.T)

    sprint = in_sprint_zone(dist_in_lap, p)
    poststart = in_poststart_brake(dist_in_lap, speed_x)
    kfinal = in_kfinal_zone(dist_in_lap, p)

    if nn_ema is None or sprint or kfinal or poststart:
        throttle = rule
    else:
        throttle = residual_throttle(rule, nn_ema)

    if sprint:
        throttle = 1.0
    if poststart:
        throttle = -1.0
    return throttle


def observation(
    track: Sequence[float],
    speed_x: float,
    angle: float,
    track_pos: float,
    dist_in_lap: float,
) -> List[float]:
    """The 23-dim NN input, before normalisation: 19 range sensors + 4 states."""
    return list(track[:19]) + [speed_x, angle, track_pos, dist_in_lap]


def observation_scale() -> List[float]:
    """Per-element divisors that map `observation()` roughly into [-1, 1]."""
    return [200.0] * 19 + [200.0, PI, 2.0, TRACK_LAP_M]
