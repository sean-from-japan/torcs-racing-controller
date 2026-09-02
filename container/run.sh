#!/usr/bin/env bash
# run.sh -- start the pinned TORCS competition container on this machine.
#
#   bash container/run.sh              # start it (or reuse a running one)
#   bash container/run.sh --race       # start it and run one measured lap
#   bash container/run.sh --stop       # stop and remove it
#
# The image is the organiser's, pinned by digest per architecture in
# image-lock.json.  It is large: about 6.1 GB compressed on arm64, roughly 18 GB
# unpacked, so the first run takes a while and needs the disk space free.
#
# Works with either docker or podman; set ENGINE to force one.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO/container/image-lock.json"
NAME="${CONTAINER_NAME:-torcs}"
OUT_HOST="${OUT_HOST:-$REPO/container/out}"

ENGINE="${ENGINE:-}"
if [ -z "$ENGINE" ]; then
    for candidate in docker podman; do
        command -v "$candidate" >/dev/null 2>&1 && ENGINE="$candidate" && break
    done
fi
if [ -z "$ENGINE" ]; then
    echo "No container engine found.  Install docker or podman first." >&2
    echo "On an Apple Silicon Mac without Docker Desktop:  brew install colima docker" >&2
    echo "then:  colima start --cpu 4 --memory 8 --disk 80" >&2
    exit 1
fi

case "$(uname -m)" in
    arm64|aarch64) ARCH=arm64 ;;
    x86_64|amd64)  ARCH=amd64 ;;
    *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

DIGEST="$(python3 -c "
import json,sys
lock = json.load(open('$LOCK'))
img = lock['images']['$ARCH']
if not img.get('digest'):
    sys.exit('no digest pinned for $ARCH in image-lock.json')
print('%s/%s@%s' % (lock['registry'], lock['repository'], img['digest']))
")"

if [ "${1:-}" = "--stop" ]; then
    "$ENGINE" rm -f "$NAME" >/dev/null 2>&1 || true
    echo "stopped $NAME"
    exit 0
fi

if [ "$ARCH" = "amd64" ]; then
    echo "note: the amd64 digest in image-lock.json has not been run and verified;"
    echo "      only the arm64 image has.  Report what you see."
fi

echo "engine  : $ENGINE"

# Prefer the derived image from container/build.sh when it exists: it has the
# bridge and gym already built in, so a run needs no network at all.
DERIVED_TAG="${DERIVED_TAG:-torcs-racing-controller:local}"
if "$ENGINE" image inspect "$DERIVED_TAG" >/dev/null 2>&1; then
    IMAGE="$DERIVED_TAG"
    echo "image   : $IMAGE (derived, base $DIGEST)"
else
    IMAGE="$DIGEST"
    echo "image   : $IMAGE"
    echo "          (no derived image; build one with container/build.sh to bake in"
    echo "           the bridge and drop the per-run network fetches)"
    if ! "$ENGINE" image inspect "$DIGEST" >/dev/null 2>&1; then
        echo "pulling (this is a large download; expect tens of minutes on a first run)"
        "$ENGINE" pull "$DIGEST"
    fi
fi

if "$ENGINE" container inspect "$NAME" >/dev/null 2>&1; then
    echo "reusing the running container '$NAME'"
else
    mkdir -p "$OUT_HOST"
    "$ENGINE" run -d --name "$NAME" \
        -p 5900:5900 -p 6080:6080 -p 3001:3001 -p 3001:3001/udp \
        -v "$REPO":/home/student/workspace/controller \
        -v "$OUT_HOST":/home/student/workspace/out \
        "$IMAGE" >/dev/null
    echo "started $NAME; waiting for the desktop"
    for _ in $(seq 1 40); do
        if curl -fs -o /dev/null --max-time 3 http://localhost:6080/vnc.html; then
            break
        fi
        sleep 3
    done
fi

echo
echo "desktop : http://localhost:6080/vnc.html"
echo "output  : $OUT_HOST"

if [ "${1:-}" = "--race" ]; then
    echo
    # The container cannot see the machine it is running on; pass it through so
    # the result record says what the lap was measured on.
    HOST_DESC="$(uname -s) $(uname -r) $(uname -m)"
    "$ENGINE" exec \
        -e "TORCS_HOST_DESCRIPTION=$HOST_DESC" \
        -e "TORCS_ENGINE=$ENGINE $("$ENGINE" --version 2>/dev/null | head -1)" \
        "$NAME" bash -lc \
        'bash /home/student/workspace/controller/container/race.sh'
else
    echo
    echo "Run one measured lap with:"
    echo "  bash container/run.sh --race"
    echo "Build the derived image (bakes in the bridge, no network per run):"
    echo "  bash container/build.sh"
    echo "or step through it yourself:"
    echo "  $ENGINE exec -it $NAME bash -lc 'bash /home/student/workspace/controller/container/race.sh --prepare'"
fi
