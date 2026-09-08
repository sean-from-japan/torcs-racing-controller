# The conditions every lap time here was measured in

A lap time means nothing without the setup that produced it. This file records the
setup, so that the numbers in `README.md` and `results/` can be read for what they are
and compared with something. It is written mainly around **106.630 s**, the fastest
figure in the repository, but the race setup applies to every stage.

Nothing here changes a published number. Some of it makes those numbers weaker than a
casual reading would suggest, which is the point of writing it down.

## What the number is

**106.630 s is the best warm lap of a single evaluation, selected as the best of 72.**

It was not produced by a separate benchmark run. It is one evaluation inside the ARS
training loop — update 7, pair 3, the negative perturbation — and the loop performed 72
evaluations over 6.6 hours. The full sequence is committed as
[`results/stage5_nn_ars_training_log.txt`](../results/stage5_nn_ars_training_log.txt).

Two consequences follow, and both cut against the number:

- **It is a maximum over noisy draws, so it is optimistically biased.** Each evaluation
  scored a candidate on the best warm lap of one episode, with no repeats. Neighbouring
  evaluations in the same log span roughly 106.6–108.3 s on candidates that differ only
  by a perturbation of scale 0.02. Some of the gap between 108.692 s and 106.630 s is
  the search finding a better controller; some of it is the search finding a lucky
  episode and keeping it.
- **The distribution around it is visible in the log, and it is wide.** Update 8, the
  round straight after, produced 106.932 / 107.264 / 107.954 / 108.274 / 106.958 /
  107.140 / 108.206 s and one DNF. A repeat measurement of the same weights was never
  taken.

The lap was also reported as reproduced by the evaluation runner and captured on video
at the time. **No artefact in this repository records that reproduction** — only the
training log, which is why the claim above is worded as it is.

Lap 1 is excluded throughout: it starts from rest, and the league did not count it.

## The race setup

Read from the development machine's TORCS configuration,
`torcs/config/raceman/quickrace.xml`:

| | |
|---|---|
| Mode | **Quick Race** (not Practice) |
| Track | `corkscrew`, category `road`, ~3,608 m |
| Race length | 10 laps |
| Damage | **off** |
| Cars on the grid | **one** — `scr_server`, index 0, and nothing else |
| Car | `car1-ow1`, the open-wheel car `scr_server 1` selects |
| Start | standing, from rest, 25 m to the line |
| Fuel | consumed normally; nothing disabled it |

The single most load-bearing line is the driver list: **there were no opponents.** The
grid is configured with two rows, which is what the project's own notes recorded, but
only one entry is filled. Every lap time in this repository is a solo run on an empty
circuit — no traffic, no slipstream, no contact, and no reason for the controller to
handle any of it. It does not, and was never asked to.

Damage off matters for the same reason: the controller drives with deliberate wall
proximity, and the bridge has upstream's out-of-track termination removed. Under damage
it would have been penalised for a driving style it was rewarded for here.

**One caveat on this evidence.** TORCS rewrites `quickrace.xml` whenever a race is
started from the menu, and the copy read above was last written on 2026-05-22, twenty
days after the record. It is therefore the setup as the machine last saved it, not a
timestamped snapshot of 2026-05-02. It agrees with the procedure the project's own
documentation described at the time — launch, RACE, NEW RACE, START — and with the
10-lap race the project notes recorded, so it is the best available evidence, but it is
evidence rather than proof.

## The host

| | |
|---|---|
| Machine | Dynabook VZ/HUL laptop |
| CPU | Intel Core i7-1195G7, 11th generation, 2.90 GHz base |
| Memory | 16 GB |
| OS | Windows 11 Home |
| Simulator | TORCS Windows build (`wtorcs.exe`), installed 2026-02-19 |
| Bridge | the project's local `gym_torcs` snapshot — the nine edits `container/prepare_bridge.py` now rebuilds |

The host is not incidental. The SCR protocol is a real-time UDP exchange on a fixed
control interval, so a machine that cannot keep up does not produce a slower lap — it
produces a different one. The evidence that this matters is already in the repository:
the same CMA-ES parameters measured 108.692 s here and 108.538 s in the container on
Apple Silicon.

## The run procedure

The evaluation is not a single drive. Both the training loop and
[`src/run_eval.py`](../src/run_eval.py) reset the environment twice before the measured
run, because `gym_torcs` restarts the race with a `meta=True` signal and TORCS does not
behave identically on the first reset of a session. The measured run is the one after
those. Skipping them changes the lap time, and the runner's docstring says so.

A session was also not reused indefinitely: repeated runs against the same `wtorcs.exe`
process degrade its internal state, and the project's practice was to restart the
simulator between measurements.

## How the container run differs

[`container/`](../container/README.md) reproduces the 108.692 s CMA-ES stage, at
108.538 s, five times. On 2026-09-08 it also ran the recovered residual network at
107.082 s and 107.416 s over two warm laps. These were measured under a different setup
from the development machine, not the same one:

| | Development machine | Container |
|---|---|---|
| Mode | Quick Race, 10 laps | Practice |
| Damage | off (race option) | off (`-nodamage`) |
| Fuel | consumed | disabled (`-nofuel`) |
| Host | x86-64 laptop, Windows | Apple Silicon, Linux in a VM |
| Simulator | Windows `wtorcs.exe` build | organiser's Linux image, pinned by digest |

So 108.538 s is a re-measurement of the *controller*, not a re-measurement under the
*same conditions*. The two agreeing to within 0.15 s is a stronger result than it would
be if the conditions were identical — but it is not a like-for-like comparison, and it
is not a tolerance.

The car was checked directly after the residual run on 2026-09-08. The generated
`practice.xml` has one driver entry, `scr_server` with `idx=0`, which selects
`car1-ow1`. That matches the development machine; indices 1–9 would select the
substantially different `car1-trb1`.

## What is not controlled anywhere

- **No repeat measurement of the original 106.630 s result.** It has one training
  observation behind it. The later pinned-container run has two warm laps — 107.082 s
  and 107.416 s — under the different conditions above, so it establishes portability
  rather than an exact reproduction.
- **One track, one car, one setup.** Every parameter is fitted to Corkscrew; `results/`
  contains no evidence of generalisation.
- **No independent verification.** Every number here was measured by this project's own
  scripts, against this project's own bridge. The league's own timing is not what these
  files record.
