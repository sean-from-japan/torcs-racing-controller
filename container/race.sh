#!/usr/bin/env bash
# race.sh -- run one measured evaluation inside the competition container.
#
# Runs from inside the container (see container/README.md).  It prepares the
# gym_torcs bridge from clean upstream, points TORCS' practice race at Corkscrew
# and scr_server, puts TORCS on the grid, and runs the controller against it.
#
#   bash container/race.sh              # measured --no-nn run
#   bash container/race.sh --prepare    # set everything up, do not race
#
# Everything it writes goes under $OUT_DIR (default /home/student/workspace/out).

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_DIR="${OUT_DIR:-/home/student/workspace/out}"
# On the derived image (container/Dockerfile) the bridge is already built and
# GYM_TORCS_DIR points at it; on the stock organiser image it is built here.
BRIDGE_DIR="${BRIDGE_DIR:-${GYM_TORCS_DIR:-$OUT_DIR/bridge}}"
UPSTREAM_DIR="$OUT_DIR/gym_torcs-upstream"
UPSTREAM_URL="https://github.com/ugo-nama-kun/gym_torcs.git"
UPSTREAM_COMMIT="da5d6ddec3a35718fea89dc1c05037743173c668"
DISPLAY="${DISPLAY:-:1}"
export DISPLAY

PREPARE_ONLY=0
[ "${1:-}" = "--prepare" ] && PREPARE_ONLY=1

mkdir -p "$OUT_DIR"

echo "== 1/5  Python dependencies"
# gym is the one thing the bridge needs that the base image does not ship.  It
# is unmaintained and warns loudly about NumPy 2; the bridge only uses
# gym.spaces to describe its action/observation shapes, which nothing here
# reads.  The derived image already has it.
python3 -c "import gym" 2>/dev/null || python3 -m pip install --quiet "gym==0.26.2"
python3 - <<'PY'
import gym, numpy, sys
print("   python %s | gym %s | numpy %s" % (sys.version.split()[0], gym.__version__, numpy.__version__))
PY

echo "== 2/5  gym_torcs bridge"
if [ "${TORCS_BRIDGE_PREBUILT:-0}" = "1" ] && [ -f "$BRIDGE_DIR/gym_torcs.py" ]; then
    # Built into the image at build time; no network needed to race.
    echo "   prebuilt in the image: $BRIDGE_DIR"
else
    if [ ! -d "$UPSTREAM_DIR" ]; then
        git clone --quiet "$UPSTREAM_URL" "$UPSTREAM_DIR"
    fi
    git -C "$UPSTREAM_DIR" fetch --quiet origin "$UPSTREAM_COMMIT" 2>/dev/null || true
    git -C "$UPSTREAM_DIR" checkout --quiet "$UPSTREAM_COMMIT"
    python3 "$REPO/container/prepare_bridge.py" \
        --src "$UPSTREAM_DIR" --out "$BRIDGE_DIR" --force
fi

echo "== 3/5  race configuration"
# TORCS writes ~/.torcs on its first run; make sure it exists before editing it.
if [ ! -f "$HOME/.torcs/config/raceman/practice.xml" ]; then
    timeout 20 /usr/local/torcs/bin/torcs -e >/dev/null 2>&1 || true
fi
python3 "$REPO/container/configure_race.py"

echo "== 4/5  starting TORCS"
# The XFCE screen locker steals the keyboard and breaks the menu walk below.
pkill -f xfce4-screensaver 2>/dev/null || true
xset s off 2>/dev/null || true
xset s noblank 2>/dev/null || true

torcs_window() {
    xwininfo -root -tree 2>/dev/null | grep -o '"[^"]*torcs-bin"' | head -1
}

on_grid() {
    netstat -lunp 2>/dev/null | grep -q ':3001 '
}

started=0
for attempt in 1 2 3; do
    pkill 'torcs-bin' 2>/dev/null || true
    sleep 2
    nohup /usr/local/torcs/bin/torcs -nofuel -nodamage -nolaptime \
        > "$OUT_DIR/torcs.log" 2>&1 &

    # Wait for the window itself rather than guessing at a sleep; a cold
    # container is much slower to get here than a warm one, and keystrokes sent
    # before TORCS has the focus are simply lost.
    for _ in $(seq 1 60); do
        [ -n "$(torcs_window)" ] && break
        sleep 1
    done
    if [ -z "$(torcs_window)" ]; then
        echo "   attempt $attempt: TORCS never opened a window"
        continue
    fi
    sleep 3

    # Click inside the TORCS window to give it the keyboard.  xdotool is not in
    # the image, and a click on blank menu background does nothing else.
    xte 'mousemove 320 400' 'mouseclick 1'
    sleep 1

    # Main menu -> Race -> (5 down) Practice -> New Race.  TORCS' menus are
    # fixed, so this walk is stable.  `torcs -r` was tried first and is not
    # usable here: it starts the race immediately and segfaults waiting for a
    # client that has not connected yet.
    for key in Return Down Down Down Down Down Return Return; do
        xte "key $key"
        sleep 0.5
    done

    for _ in $(seq 1 20); do
        on_grid && break
        sleep 1
    done
    if on_grid; then
        started=1
        break
    fi
    echo "   attempt $attempt: TORCS did not reach the grid; restarting it"
done

if [ "$started" != "1" ]; then
    echo "TORCS is not listening on UDP 3001.  Look at $OUT_DIR/torcs.log and at"
    echo "the desktop on http://localhost:6080/vnc.html to see where it stopped."
    exit 1
fi
echo "   TORCS is on the grid, waiting on UDP 3001"

if [ "$PREPARE_ONLY" = "1" ]; then
    echo
    echo "Prepared.  Race it with:"
    echo "  cd $REPO && GYM_TORCS_DIR=$BRIDGE_DIR python3 -m src.run_eval --no-nn --no-wait"
    exit 0
fi

echo "== 5/5  measured run"
cd "$REPO"
GYM_TORCS_DIR="$BRIDGE_DIR" python3 -u -m src.run_eval --no-nn --no-wait \
    2>&1 | tee "$OUT_DIR/run_eval.log"

python3 "$REPO/container/record_result.py" \
    --log "$OUT_DIR/run_eval.log" \
    --bridge "$BRIDGE_DIR" \
    --out "$OUT_DIR/result.json"
