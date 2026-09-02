#!/usr/bin/env python3
"""record_result.py -- write down what was measured, and on what.

A lap time on its own is not a result.  This records the lap alongside the things
that could have changed it: the container image digest, the controller commit,
the bridge the controller was driven through, the interpreter and library
versions, and the machine underneath.  Without those a number cannot be compared
to another run, and this project's whole point is that its numbers can be.

    python3 container/record_result.py --log out/run_eval.log \\
        --bridge out/bridge --out out/result.json
"""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, cwd=_REPO):
    try:
        out = subprocess.run(
            ("git",) + args, cwd=cwd, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def pkg_versions(names):
    out = {}
    for name in names:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except ImportError:
            out[name] = None
    return out


def parse_log(text):
    # Only the laps from the RESULTS block: the log also contains the warm-up
    # run and any retried attempts, which are not part of the measurement.
    laps = re.search(r"all laps      : \[([^\]]*)\]", text)
    best = re.search(r"best warm lap : ([0-9.]+) s", text)
    ref = re.search(r"measured ([0-9.]+) s\)", text)
    return {
        "laps_s": [float(x) for x in re.findall(r"[0-9.]+", laps.group(1))] if laps else [],
        "best_warm_lap_s": float(best.group(1)) if best else None,
        "reference_s": float(ref.group(1)) if ref else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--log", required=True, help="captured run_eval.py output")
    ap.add_argument("--bridge", required=True, help="prepared gym_torcs directory")
    ap.add_argument("--out", required=True, help="where to write the record")
    ap.add_argument("--note", default=None, help="free-text note about this run")
    args = ap.parse_args(argv)

    with open(args.log, encoding="utf-8", errors="replace") as f:
        log = f.read()
    result = parse_log(log)

    with open(os.path.join(_HERE, "image-lock.json")) as f:
        lock = json.load(f)

    bridge_files = {}
    for name in sorted(os.listdir(args.bridge)):
        path = os.path.join(args.bridge, name)
        if os.path.isfile(path):
            bridge_files[name] = sha256(path)

    record = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": args.note,
        "result": result,
        "controller": {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
            "parameters": "results/stage4_cma_8param_sector_s35.json",
        },
        "bridge": {
            "upstream": lock["upstream_bridge"]["repository"],
            "upstream_commit": lock["upstream_bridge"]["commit"],
            "prepared_by": "container/prepare_bridge.py",
            "files_sha256": bridge_files,
        },
        "image": {
            "repository": "%s/%s" % (lock["registry"], lock["repository"]),
            "digest": lock["images"].get(
                "arm64" if platform.machine() in ("aarch64", "arm64") else "amd64", {}
            ).get("digest"),
        },
        "environment": {
            "container": {
                "machine": platform.machine(),
                "platform": platform.platform(),
                "python": sys.version.split()[0],
                "packages": pkg_versions(["numpy", "gym", "torch"]),
            },
            # Filled in by container/run.sh, which is the only part of this that
            # can see the machine outside the container.
            "host": os.environ.get("TORCS_HOST_DESCRIPTION"),
            "engine": os.environ.get("TORCS_ENGINE"),
        },
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=False)
        f.write("\n")

    print("\nrecorded: %s" % args.out)
    if result["best_warm_lap_s"] is not None:
        print(
            "  best warm lap %.3f s (reference %.3f s, %+.3f s)"
            % (
                result["best_warm_lap_s"],
                result["reference_s"] or float("nan"),
                result["best_warm_lap_s"] - (result["reference_s"] or 0.0),
            )
        )
    else:
        print("  no lap parsed from %s -- the run did not finish a warm lap" % args.log)
    return 0


if __name__ == "__main__":
    sys.exit(main())
