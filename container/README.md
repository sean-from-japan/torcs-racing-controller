# Running the controller against real TORCS

The controller and its tests run anywhere. Actually *driving* it needs TORCS, the
SCR server patch, and a specific gym_torcs bridge — a 2013-era simulator stack
that is unpleasant to assemble by hand. This directory reduces that to one
command against a pinned container image.

```bash
bash container/build.sh     # once: bake the bridge into a derived image
bash container/run.sh --race
```

`build.sh` builds a thin image on top of the pinned base with the simulator
bridge and its one missing Python dependency already in place. `run.sh --race`
then starts it, points TORCS at Corkscrew with the SCR driver, puts the car on
the grid, drives a measured evaluation, and writes a result record to
`container/out/result.json`.

`build.sh` is optional. Skip it and `run.sh` falls back to the stock organiser
image, fetching the bridge and `gym` at run time instead.

## What this is, and what it is not

This is a **derived image on a pinned base**, not a container reproducible from
source end to end.

The base image is the competition organiser's, published on Docker Hub. Its own
build definition is not public, so it can be *identified* exactly — by digest,
never by `:latest` — but not rebuilt. Everything layered on top of it is in
[`Dockerfile`](Dockerfile) and is fully verifiable. That is the honest shape of
this: an identified base, a reproducible derivation.

The TORCS sources themselves are not the obstacle — they ship publicly in
upstream gym_torcs as `vtorcs-RL-color/`, SCR driver included, and the base image
carries them at `/opt/torcs-src`. A from-source image is a possible later
milestone. It is not this one, because a self-compiled simulator binary is not
the binary these lap times were measured against, and swapping it out would
discard the only baseline this work has established.

What is reproducible from source in this repository:

| Piece | How |
|---|---|
| The runner image | `Dockerfile` + `build.sh`, from the pinned base digest in the lock |
| The gym_torcs bridge | `prepare_bridge.py`, from a pinned upstream commit, with every edit anchored to exact upstream text |
| The race setup | `configure_race.py`, editing TORCS' own generated config in place |
| Putting TORCS on the grid | `race.sh`, deterministic menu walk with verification and retries |
| The measurement | `src/run_eval.py --no-nn` against the committed Stage 4 parameters |
| What was measured, and on what | `record_result.py` |

### Why the derived image is worth building

Without it, every run fetches `gym` from PyPI and the bridge sources from GitHub.
`prepare_bridge.py` verifies the bridge by SHA-256, so upstream drift is caught —
but the PyPI wheel is pinned only by version, and either service can change or
disappear. Building once moves both into an image layer, and the run becomes
hermetic.

Verified: the derived image races with `--network none` and produces the same
lap. The bridge baked into the image is byte-identical (SHA-256) to the one built
at run time.

It costs almost nothing on disk — the derived layers add **7.1 MB** on top of the
17.9 GB base, which is shared. The build takes about two minutes.

The image is built locally and **must not be pushed**: publishing it would
redistribute the organiser's image.

## Requirements

- **Disk:** about 6.1 GB to download on arm64, roughly 18 GB unpacked. Keep 25 GB
  free.
- **Memory:** 8 GB for the container.
- **Engine:** Docker or Podman. `run.sh` picks whichever it finds; set `ENGINE`
  to force one.

On an Apple Silicon Mac without Docker Desktop:

```bash
brew install colima docker
colima start --cpu 4 --memory 8 --disk 80
```

The image digests for arm64 and amd64 are both in
[`image-lock.json`](image-lock.json). `run.sh` selects by `uname -m`. Only the
**arm64** digest has actually been pulled and raced; the amd64 one is recorded
from the registry and is untested here.

## Observed result

| | |
|---|---|
| Host | macOS 26.6.2, Apple Silicon, Colima 0.10.3, Docker CLI 29.7.2 / engine 29.5.2 |
| Image | `johnsloe/torcs-competition@sha256:fed57137…c4b9` (arm64) |
| Controller | `src/run_eval.py --no-nn`, Stage 4 CMA-ES parameters |
| Cold lap | 114.130 s |
| **Best warm lap** | **108.538 s** |
| Historical reference | 108.692 s |

Five runs on 2026-09-02 — the original private bridge snapshot; the bridge
rebuilt by `prepare_bridge.py`; a fully automated run from a fresh container; a
clean copy of the repository in a fresh container; and the derived image with
`--network none` — all produced **the same cold lap (114.130 s) and the same best
warm lap (108.538 s)**. Only the third lap, which is not the measured one, varied
(109.454 s / 109.898 s).

That is five runs on one machine. It is not evidence that the lap time is
identical on other hosts or architectures, and no tolerance is claimed. What is
claimed is that the run is repeatable here, and that everything needed to check
it elsewhere is recorded.

## The bridge, and why it is rebuilt rather than shipped

The lap times in this repository were **not** measured against stock gym_torcs.
They were measured against a locally modified copy, and the modifications change
behaviour — stock gym_torcs will not reproduce them.

`prepare_bridge.py` rebuilds that copy from
[ugo-nama-kun/gym_torcs](https://github.com/ugo-nama-kun/gym_torcs) at commit
`da5d6dd`, verifying each source file by SHA-256 before touching it. The edits:

### `gym_torcs.py`

| Edit | Why it matters |
|---|---|
| Constructor no longer launches TORCS | The caller owns the process; the upstream `os.system('torcs &')` path fights the container's already-running desktop |
| `terminal_judge_start` 500 → 3000 | Upstream ends the episode if the car is slow ~10 s in; the car is still accelerating out of the pits then |
| Removed out-of-track termination | Corkscrew is driven with deliberate wall proximity; upstream ends the episode at `track.min() < 0` |
| Removed backward-heading termination | Momentary backward heading in the corkscrew section is recovered from, not fatal |
| Enabled the snakeoil gear thresholds | **Upstream locks the car in gear 1.** This is the single biggest behavioural difference |
| `agent_to_torcs` casts gear to `int` | TORCS rejects a float gear |
| SCR port 3101 → 3001 | Competition port; also what the image exposes |
| `end()` no longer `pkill torcs` | The caller owns the process |
| `autostart.sh` resolved next to the module | Upstream resolves it relative to the working directory |

### `snakeoil3_gym.py`

| Edit | Why it matters |
|---|---|
| A failed connection raises `ConnectionError` instead of relaunching TORCS | Upstream silently kills and restarts TORCS underneath the caller, which hides real failures and does not work when the caller owns the process |

Cosmetic reformatting present in the original working copy is deliberately **not**
reproduced, so that every edit above is a behavioural one and the diff stays
reviewable. The rebuilt bridge was checked against the working copy that produced
the recorded lap by comparing parsed syntax trees: every definition matches
except `reset_torcs`, where the two spell the same absolute `autostart.sh` path
differently, and which the measured path never calls.

## Provenance of the edits

The organiser's starter kit shipped `gym_torcs.py` **byte-identical to upstream**,
so every difference listed above is this project's own work, not the organiser's.
The organiser did change `snakeoil3_gym.py`, but only inside `drive_example()`
(a demo driver) and its `__main__` block — neither runs on this path, and neither
is reproduced here.

The organiser's starter kit, course materials, and their vendored TORCS sources
are **not** redistributed in this repository.

## Running it in pieces

```bash
bash container/build.sh               # build the derived image
bash container/run.sh                 # start the container, leave it running
bash container/run.sh --stop          # stop and remove it

# inside the container: set everything up but do not race
docker exec -it torcs bash -lc \
  'bash /home/student/workspace/controller/container/race.sh --prepare'
```

The desktop is at <http://localhost:6080/vnc.html> while the container runs —
useful for watching the car, and for seeing where TORCS stopped if the menu walk
fails.

On the stock base image, `race.sh` installs `gym==0.26.2` on first use and builds
the bridge. On the derived image both are already present and it skips straight
to the race. `gym` is the one dependency the base image does not ship; it is
unmaintained and warns about NumPy 2, but the bridge only uses `gym.spaces` to
describe shapes that nothing here reads.

## Notes on things that did not work

- **`torcs -r <raceconfig>`** looks like the clean deterministic entry point and
  is not usable: it starts the race immediately, times out waiting for a client
  that has not connected yet, and segfaults. The GUI menu walk is what works.
- **`autostart.sh` from upstream gym_torcs** navigates to *Quick Race*, not
  Practice, on this TORCS build. `race.sh` uses its own walk.
- **The XFCE screen locker** in the image takes the keyboard after a few minutes
  idle and silently breaks the menu walk. `race.sh` kills it first.
- **Colima only shares `$HOME` with its VM.** A clone outside your home directory
  mounts as an empty directory and `race.sh` is reported missing. Keep the clone
  under `$HOME`, or add the path with `colima start --mount`.

## Not covered here

The 106.630 s residual-network result cannot be reproduced: its trained weights
were never archived. See the repository README. This directory reproduces the
Stage 4 CMA-ES configuration only.
