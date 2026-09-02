#!/usr/bin/env bash
# build.sh -- build the derived runner image on top of the pinned base.
#
#   bash container/build.sh
#
# The base image digest comes from image-lock.json for this architecture, so the
# lock stays the single source of truth.  The result is tagged locally and is
# never pushed: pushing it would redistribute the organiser's image.
#
# run.sh uses this image automatically once it exists.

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOCK="$REPO/container/image-lock.json"
TAG="${DERIVED_TAG:-torcs-racing-controller:local}"

ENGINE="${ENGINE:-}"
if [ -z "$ENGINE" ]; then
    for candidate in docker podman; do
        command -v "$candidate" >/dev/null 2>&1 && ENGINE="$candidate" && break
    done
fi
[ -n "$ENGINE" ] || { echo "No container engine found." >&2; exit 1; }

case "$(uname -m)" in
    arm64|aarch64) ARCH=arm64 ;;
    x86_64|amd64)  ARCH=amd64 ;;
    *) echo "Unsupported host architecture: $(uname -m)" >&2; exit 1 ;;
esac

BASE="$(python3 -c "
import json, sys
lock = json.load(open('$LOCK'))
img = lock['images']['$ARCH']
if not img.get('digest'):
    sys.exit('no digest pinned for $ARCH in image-lock.json')
print('%s/%s@%s' % (lock['registry'], lock['repository'], img['digest']))
")"

echo "engine : $ENGINE"
echo "base   : $BASE"
echo "tag    : $TAG"
echo

"$ENGINE" build \
    --build-arg "BASE_IMAGE=$BASE" \
    -f "$REPO/container/Dockerfile" \
    -t "$TAG" \
    "$REPO"

echo
echo "built $TAG"
echo "Race it with:  bash container/run.sh --race"
