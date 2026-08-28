# Results — provenance

Every file here was written by a training or benchmark script during the project. None
of it was reconstructed or re-measured afterwards. Files were renamed to describe their
stage; the original names are given so anything quoted here can be traced back.

`best_lap_2plus_s` is the metric throughout: the fastest lap from lap 2 onwards. Lap 1
starts from rest and did not count in the league.

| File | Original name | Best warm lap | Stage |
|---|---|---|---|
| `stage0_baseline_snakeoil.json` | `baseline_snakeoil.json` | 261.422 s | Supplied rule-based reference driver |
| `stage0_baseline_jm2.json` | `baseline_jm2.json` | — (205.168 s cold) | Second supplied driver; never completed a warm lap |
| `stage1_cma_3param.json` | `cma_best_laptime.json` | 143.100 s | CMA-ES over `A`, `B`, `C` |
| `stage2_cma_5param.json` | `cma4_best_laptime.json` | 124.148 s | `+ K`, `T` |
| `stage3_cma_6param_deadband.json` | `cma5h_best.json` | 122.060 s | `+ D`, 5 h refinement run |
| `stage3b_cma_6param_stable4lap.json` | `cma8h_phaseB_best.json` | 122.848 s | Same stage, 4 consecutive laps — the more robust individual |
| `stage3c_cma_6param_tight_refine.json` | `cma_tight_best.json` | 122.164 s | Same stage, tight local refinement, 2 laps |
| `stage4_cma_8param_sector_s35.json` | `cma_s35_best.json` | **108.692 s** | `+ K_final`, `switch_dist`, finish sprint, `C_s35` chicane cap |
| `corkscrew_segments.json` | unchanged | — | Track geometry, extracted by `src/analyze_track.py` |
| `lap_times_raw.csv` | `lap_times_all.csv` | — | Raw per-lap log, 572 records |

`stage4_cma_8param_sector_s35.json` is the parameter set the code loads: it is the input
to `src/run_eval.py`, the starting point for `src/train_nn_ars.py`, and the fixture the
test suite asserts against.

## Three files for one stage

Stage 3 has three entries because "best" is ambiguous when fitness is a single noisy
run. 122.060 s was the fastest lap recorded; 122.848 s came from the individual that held
together over four consecutive laps; 122.164 s came from a tight local refinement that
only ran two. All three are kept rather than only the fastest, because the difference
between them is roughly the size of the run-to-run variance and picking one silently
would overstate the precision.

The 122.060 s individual is the one the next stage was seeded from.

## Missing artefacts

Two figures quoted in my project report have **no file here**, because none was ever
committed:

- **116.062 s** — 8-parameter sector controller, before the s35 cap
- **112.404 s** — after adding the full-throttle finish-straight sprint

Both changes are present in `stage4_cma_8param_sector_s35.json` (`K_final`,
`switch_dist`, `back_dist`), so the effect survives in the parameters even though the
intermediate measurements do not.

The residual-NN result of **106.630 s** likewise has no artefact here: the output-layer
weights ARS converged on were never committed. The training loop that produces them is
`src/train_nn_ars.py`, and it runs from `stage4_cma_8param_sector_s35.json` — what is
missing is that run's output, not the means of producing one. See "Reproducing this" in
the top-level README.

## About `lap_times_raw.csv`

The original log, unmodified. Its column count varies between 7 and 10 across the file
because each stage's script appended rows under its own header as the parameter set
grew. It covers the CMA-ES stages up to 122.060 s only — the later stages logged to
separate files that were not committed.

It is included as-is rather than normalised, so that the raw record and the summary
JSONs can be checked against each other.
