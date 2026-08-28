# Corkscrew Track Analysis

Track length: 3608.0 m (grid offset ≈ 85 m from XML start)
Total segments: 66

## Key for CMA-ES sector control

| distRaced (m) | Segment | Type | R (m) | Length (m) | Notes |
|---|---|---|---|---|---|
| 2829 | s43 | STRAIGHT | — | 48 |  |
| 2874 | s44-1 | STRAIGHT | — | 45 |  |
| 2940 | s44-2 | STRAIGHT | — | 66 |  |
| 2980 | s45-1 | RIGHT | 58 | 41 | medium corner before straight |
| 3019 | s45-2 | RIGHT | 55 | 39 | medium corner before straight |
| 3144 | s46-1 | STRAIGHT | — | 125 | ← braking zone (switch here) |
| 3221 | s46-2 | STRAIGHT | — | 77 | ← braking zone (switch here) |
| 3247 | s47 | STRAIGHT | — | 26 |  |
| 3268 | s48-1 | LEFT | 20 | 20 | ← **BOTTLENECK** R=18-20m |
| 3286 | s48-2 | LEFT | 18 | 19 | ← **BOTTLENECK** R=18-20m |
| 3304 | s49-1 | STRAIGHT | — | 18 |  |
| 3377 | s49-2 | STRAIGHT | — | 73 |  |
| 3602 | s50 | STRAIGHT | — | 225 | finish straight |
| 3608 | s50-2 | STRAIGHT | — | 6 |  |

## All tight corners (radius < 100 m), by radius

| distRaced (m) | Segment | Dir | R (m) | Arc (°) | Length (m) |
|---|---|---|---|---|---|
| 2495 | s37 | RIGHT | 14 | 83 | 21 |
| 3286 | s48-2 | LEFT | 18 | 58 | 19 |
| 3268 | s48-1 | LEFT | 20 | 58 | 20 |
| 2461 | s35-2 | LEFT | 22 | 52 | 20 |
| 598 | s8 | LEFT | 24 | 94 | 39 |
| 2441 | s35-1 | LEFT | 32 | 52 | 29 |
| 550 | s6 | LEFT | 33 | 102 | 58 |
| 1940 | s24 | LEFT | 34 | 54 | 32 |
| 769 | s12-1 | RIGHT | 44 | 47 | 36 |
| 806 | s12-2 | RIGHT | 45 | 47 | 37 |
| 1603 | s21 | LEFT | 48 | 66 | 55 |
| 3019 | s45-2 | RIGHT | 55 | 40 | 39 |
| 2980 | s45-1 | RIGHT | 58 | 40 | 41 |
| 1084 | s15-2 | RIGHT | 62 | 27 | 29 |
| 2359 | s33 | RIGHT | 62 | 17 | 19 |
| 1548 | s20 | LEFT | 72 | 50 | 63 |
| 2718 | s42-1 | LEFT | 72 | 48 | 61 |
| 2781 | s42-2 | LEFT | 75 | 48 | 63 |
| 1054 | s15-1 | RIGHT | 77 | 45 | 60 |
| 307 | s3 | LEFT | 86 | 25 | 38 |
| 2035 | s26 | LEFT | 96 | 15 | 25 |

## Sector switch recommendation (for cma_sector.py)

```
s45-1/s45-2  RIGHT R=57m   at 2980-3019m  — medium corner
s46 straight  202m          at 3144-3246m  ← switch_dist ~3050m here
s48-1/s48-2  LEFT  R=18-20m at 3268-3286m  — BOTTLENECK (final corner)
s50          STRAIGHT 225m  at 3377-3602m  — finish straight

K_final = 0.9: at track[9]=200m → adaptive_speed=180 < 200 → braking starts on s46
K_final = 0.7: at track[9]=200m → adaptive_speed=140 → stronger braking
```

## Full segment list

| distRaced (m) | Segment | Type | R (m) | Arc (°) | Length (m) |
|---|---|---|---|---|---|
| 117 | s1 | LEFT | 154 | 12 | 32 |
| 269 | s2 | STRAIGHT | — | — | 152 |
| 307 | s3 | LEFT | 86 | 25 | 38 |
| 367 | s4-1 | STRAIGHT | — | — | 60 |
| 426 | s4-2 | STRAIGHT | — | — | 59 |
| 491 | s5 | STRAIGHT | — | — | 65 |
| 550 | s6 | LEFT | 33 | 102 | 58 |
| 558 | s7 | STRAIGHT | — | — | 9 |
| 598 | s8 | LEFT | 24 | 94 | 39 |
| 646 | s9 | STRAIGHT | — | — | 48 |
| 695 | s10 | RIGHT | 106 | 27 | 50 |
| 733 | s11 | STRAIGHT | — | — | 38 |
| 769 | s12-1 | RIGHT | 44 | 47 | 36 |
| 806 | s12-2 | RIGHT | 45 | 47 | 37 |
| 836 | s13-1 | STRAIGHT | — | — | 30 |
| 881 | s13-2 | STRAIGHT | — | — | 45 |
| 916 | s13-3 | STRAIGHT | — | — | 35 |
| 994 | s14 | STRAIGHT | — | — | 78 |
| 1054 | s15-1 | RIGHT | 77 | 45 | 60 |
| 1084 | s15-2 | RIGHT | 62 | 27 | 29 |
| 1156 | s16 | STRAIGHT | — | — | 72 |
| 1206 | s17 | STRAIGHT | — | — | 50 |
| 1299 | s18 | RIGHT | 259 | 20 | 93 |
| 1392 | s19-1 | STRAIGHT | — | — | 93 |
| 1485 | s19-2 | STRAIGHT | — | — | 93 |
| 1548 | s20 | LEFT | 72 | 50 | 63 |
| 1603 | s21 | LEFT | 48 | 66 | 55 |
| 1633 | s22-1 | STRAIGHT | — | — | 30 |
| 1723 | s22-2 | STRAIGHT | — | — | 90 |
| 1796 | s22-3 | STRAIGHT | — | — | 72 |
| 1908 | s23 | STRAIGHT | — | — | 112 |
| 1940 | s24 | LEFT | 34 | 54 | 32 |
| 2010 | s25 | STRAIGHT | — | — | 70 |
| 2035 | s26 | LEFT | 96 | 15 | 25 |
| 2064 | s27 | STRAIGHT | — | — | 29 |
| 2138 | s28 | LEFT | 384 | 11 | 74 |
| 2185 | s29 | STRAIGHT | — | — | 47 |
| 2218 | s30 | RIGHT | 480 | 4 | 34 |
| 2268 | s31 | LEFT | 480 | 6 | 50 |
| 2341 | s32 | STRAIGHT | — | — | 72 |
| 2359 | s33 | RIGHT | 62 | 17 | 19 |
| 2412 | s34 | STRAIGHT | — | — | 53 |
| 2441 | s35-1 | LEFT | 32 | 52 | 29 |
| 2461 | s35-2 | LEFT | 22 | 52 | 20 |
| 2474 | s36 | STRAIGHT | — | — | 14 |
| 2495 | s37 | RIGHT | 14 | 83 | 21 |
| 2534 | s38 | STRAIGHT | — | — | 38 |
| 2572 | s39 | RIGHT | 115 | 19 | 38 |
| 2632 | s40 | LEFT | 144 | 24 | 60 |
| 2657 | s41 | STRAIGHT | — | — | 25 |
| 2718 | s42-1 | LEFT | 72 | 48 | 61 |
| 2781 | s42-2 | LEFT | 75 | 48 | 63 |
| 2829 | s43 | STRAIGHT | — | — | 48 |
| 2874 | s44-1 | STRAIGHT | — | — | 45 |
| 2940 | s44-2 | STRAIGHT | — | — | 66 |
| 2980 | s45-1 | RIGHT | 58 | 40 | 41 |
| 3019 | s45-2 | RIGHT | 55 | 40 | 39 |
| 3144 | s46-1 | STRAIGHT | — | — | 125 |
| 3221 | s46-2 | STRAIGHT | — | — | 77 |
| 3247 | s47 | STRAIGHT | — | — | 26 |
| 3268 | s48-1 | LEFT | 20 | 58 | 20 |
| 3286 | s48-2 | LEFT | 18 | 58 | 19 |
| 3304 | s49-1 | STRAIGHT | — | — | 18 |
| 3377 | s49-2 | STRAIGHT | — | — | 73 |
| 3602 | s50 | STRAIGHT | — | — | 225 |
| 3608 | s50-2 | STRAIGHT | — | — | 6 |
