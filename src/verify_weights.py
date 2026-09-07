"""
verify_weights.py -- check that models/nn_ars_s35_best.pt is the file the
106.630 s lap was driven with, and that it still loads and computes.

The weights are the one artefact here that a reader cannot check by eye, and
the only one that would silently degrade if it were ever replaced, re-saved by
a different Torch, or truncated in transit.  So they are pinned twice: by
SHA-256 over the bytes, and by the network's output on three fixed
observations.  A file that passes both is the file that drove the lap.

    python src/verify_weights.py --checksum-only   # stdlib, no Torch needed
    python src/verify_weights.py                   # also load and evaluate

Neither mode needs TORCS.  Neither mode proves the lap time -- that needs the
simulator; see container/README.md.
"""

import argparse
import hashlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_DIR)
WEIGHTS = os.path.join(_REPO, "models", "nn_ars_s35_best.pt")

EXPECTED_SHA256 = "03f489840cc039e89378dcd10a9f42e8d18fad1047e80c295fde4ab2252be5e5"
EXPECTED_BYTES = 10549

# Three fixed raw observations -- track[0:19] then speedX, angle, trackPos,
# dist_in_lap -- chosen to sit in the middle of the lap, in the s35 approach,
# and hard against the left wall.  They are inputs only; nothing about them is
# a measurement.
PROBES = [
    [200.0] * 19 + [150.0, 0.0, 0.0, 500.0],
    [60.0, 62.0, 66.0, 72.0, 80.0, 90.0, 102.0, 116.0, 132.0, 150.0,
     132.0, 116.0, 102.0, 90.0, 80.0, 72.0, 66.0, 62.0, 60.0]
    + [95.0, -0.18, 0.16, 2400.0],
    [4.0, 4.2, 4.6, 5.4, 6.6, 8.4, 11.0, 15.0, 21.0, 30.0,
     45.0, 70.0, 110.0, 160.0, 200.0, 200.0, 200.0, 200.0, 200.0]
    + [40.0, 0.42, -0.93, 1500.0],
]

# What the published file returns for those probes, to 6 decimal places.
EXPECTED_OUTPUTS = [0.743562, 0.415774, 0.660467]

TOLERANCE = 1e-5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def check_bytes():
    if not os.path.exists(WEIGHTS):
        raise SystemExit("missing: %s" % WEIGHTS)
    size = os.path.getsize(WEIGHTS)
    digest = sha256(WEIGHTS)
    print("  size    %d bytes  (expected %d)" % (size, EXPECTED_BYTES))
    print("  sha256  %s" % digest)
    if size != EXPECTED_BYTES or digest != EXPECTED_SHA256:
        raise SystemExit("  MISMATCH -- this is not the published file")
    print("  bytes OK")


def check_outputs():
    import run_eval

    model = run_eval.build_model(WEIGHTS)
    last = list(model[4].parameters())
    n = sum(p.numel() for p in last)
    print("  output layer  %d parameters (expected 33 -- what ARS searched)" % n)
    if n != 33:
        raise SystemExit("  MISMATCH -- unexpected architecture")

    worst = 0.0
    for probe, expected in zip(PROBES, EXPECTED_OUTPUTS):
        got = run_eval.nn_output(model, probe)
        worst = max(worst, abs(got - expected))
        print("  probe  got %+.6f  expected %+.6f" % (got, expected))
    if worst > TOLERANCE:
        raise SystemExit("  MISMATCH -- worst deviation %.2e" % worst)
    print("  outputs OK (worst deviation %.2e)" % worst)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument(
        "--checksum-only",
        action="store_true",
        help="verify the bytes only; does not import torch",
    )
    args = p.parse_args(argv)

    print("models/nn_ars_s35_best.pt")
    check_bytes()
    if not args.checksum_only:
        check_outputs()
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
