# `nn_ars_s35_best.pt` — the residual network from the 106.630 s lap

The output-layer weights ARS converged on, and the hidden layers they were searched
on top of. This is the file [issue #1](https://github.com/sean-from-japan/torcs-racing-controller/issues/1)
was opened for: the one result in this repository that had a number in the README and
no artefact behind it. It was never lost — it was outside version control, on the
Windows machine the project was developed on, which is where it has now come from.

|  |  |
|---|---|
| Size | 10,549 bytes |
| SHA-256 | `03f489840cc039e89378dcd10a9f42e8d18fad1047e80c295fde4ab2252be5e5` |
| Format | PyTorch `state_dict`, zip serialisation |
| Loads with | `torch.load(..., weights_only=True)` — no pickled objects beyond tensors |
| Architecture | `23 → 32 → ReLU → 32 → ReLU → 1 → Tanh`, loads `strict=True` into `src/run_eval.py:build_model` |

Check it before trusting it:

```bash
python src/verify_weights.py --checksum-only   # bytes only, stdlib, no Torch
python src/verify_weights.py                   # + load and evaluate on fixed inputs
```

The second mode pins the network's output on three fixed observations. A file that
passes both checks is the file that drove the lap; a file that was re-saved, truncated
or swapped will fail one of them. CI runs the first mode on every push.

## What is in the file, and what ARS actually searched

ARS optimised **33 parameters** — the output layer, `4.weight` (1×32) and `4.bias`.
Everything before it was frozen for the whole run. But the frozen part is not the
random initialisation the training script's comment describes: layers 0 and 2 in this
file are **byte-identical** to `models/bc_keff.pt`, a behaviour-cloning pretrain that
regressed the CMA-ES controller's own throttle command over 91,329 recorded control
steps on Corkscrew.

That matters in two directions:

- **For driving, the file is self-contained.** The pretrained hidden layers are saved
  inside it. `src/run_eval.py` needs nothing but this file.
- **For retraining, it is not.** `src/train_nn_ars.py` reproduces the published run's
  starting point only if `models/bc_keff.pt` is present. Without it the hidden layers
  fall back to random initialisation, which is a *different* experiment that will
  converge somewhere else. That file is not published here; ask me for it.

The training script previously claimed the behaviour-cloning model had a 22-dimensional
input and that only its first layer was skipped. Comparing the two files disproves it:
every hidden weight transferred, first layer included. The 22-dimensional version was an
earlier, superseded attempt. The comment is corrected.

## The run this came from

Read straight out of [`results/stage5_nn_ars_s35.json`](../results/stage5_nn_ars_s35.json),
which is the record the training loop wrote:

| | |
|---|---|
| Date | 2026-05-02 |
| Base | [`stage4_cma_8param_sector_s35.json`](../results/stage4_cma_8param_sector_s35.json), 108.692 s |
| Budget | 6 h, 9 ARS updates, 4 mirrored pairs per update |
| Hyperparameters | σ = 0.02, α = 0.02, 2 warm laps per evaluation |
| Best at | update 7, pair 3, −δ |
| Best warm lap | **106.630 s** (−2.062 s) |

The run's own log is committed as
[`results/stage5_nn_ars_training_log.txt`](../results/stage5_nn_ars_training_log.txt).
It is what ties this file to that number: the loop overwrote `nn_ars_s35_best.pt` on
every improvement, 106.630 s was the last improvement it recorded, and the eight
evaluations after it were all slower. So the file the run left behind is that
individual.

## What has been verified, and what has not

Verified, here, today:

- the bytes match what the development machine holds
- the file loads `strict=True` into the committed architecture
- the output layer is non-zero, so this is a trained network and not the zero
  initialisation that would make the agent identical to the CMA-ES controller
- the network evaluates, deterministically, to the outputs pinned in
  `src/verify_weights.py`
- the observation vector and scaling in `src/controller.py` are identical to the ones
  the private runner used, so the file is being fed what it was trained on
- the published file drove the pinned arm64 container to warm laps of 107.082 s and
  107.416 s on 2026-09-08, improving on that container's 108.538 s CMA-ES-only result

**Not reproduced exactly: the lap time.** 106.630 s was measured on the original Windows
machine in May 2026. The pinned-container best is 107.082 s, 0.452 s slower. The original
number is also the best of the 72 evaluations the ARS loop made rather than a separate
benchmark run, and it was measured solo on the grid with damage off.
[`docs/RACE_CONDITIONS.md`](../docs/RACE_CONDITIONS.md) records that setup in full and
says plainly what a maximum over noisy draws is worth. Publishing the weights removes
the missing-artefact problem, and the container run makes the controller portable; it
does not turn 106.630 s into a portable exact measurement. See the committed container
record in `results/` and [`container/README.md`](../container/README.md).

The Torch version used for training was not recorded. The file carries
`.format_version` and a 64-byte storage alignment, which PyTorch began writing in 2.6,
so it was saved by 2.6 or later; it loads under 2.11.0 on CPU. `requirements.txt` asks
for `torch>=2.1`, which is the runtime floor, not evidence about the training run.
