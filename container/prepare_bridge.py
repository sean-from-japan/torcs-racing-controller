#!/usr/bin/env python3
"""prepare_bridge.py -- build the gym_torcs bridge this controller was measured against.

The lap times in this repository were not recorded against stock gym_torcs.  They
were recorded against a locally modified copy, and the modifications change how
episodes terminate and how gears are selected -- they are not cosmetic.  Stock
gym_torcs will not reproduce them.

Rather than redistribute that copy (it is a derivative of MIT-licensed upstream
code that also ships GPL-2.0 TORCS sources), this script rebuilds it: it takes a
clean upstream checkout at a pinned commit, verifies the two files it touches
byte-for-byte, and applies the edits below.  Every edit is anchored to exact
upstream text, so the script fails loudly rather than silently producing a
different bridge.

    python3 container/prepare_bridge.py --src /path/to/gym_torcs --out third_party/gym_torcs

`--src` must be a checkout of

    https://github.com/ugo-nama-kun/gym_torcs
    at commit da5d6ddec3a35718fea89dc1c05037743173c668

See container/README.md for where each edit came from and why it is here.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
LOCK = os.path.join(_HERE, "image-lock.json")

# Files copied verbatim from upstream in addition to the two patched ones.
VERBATIM = ["autostart.sh", "LICENSE"]


# --------------------------------------------------------------------------
# gym_torcs.py
# --------------------------------------------------------------------------

GYM_TORCS_EDITS = [
    (
        "do not launch TORCS from the constructor",
        """        ##print("launch torcs")
        os.system('pkill torcs')
        time.sleep(0.5)
        if self.vision is True:
            os.system('torcs -nofuel -nodamage -nolaptime  -vision &')
        else:
            os.system('torcs  -nofuel -nodamage -nolaptime &')
        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)
""",
        """        # TORCS is started and put on the grid by the caller, not here.
        # See container/README.md, "Starting the race".
""",
    ),
    (
        "hold the stall check off until the car is up to speed",
        "    terminal_judge_start = 500  # Speed limit is applied after this step",
        "    terminal_judge_start = 3000  # Speed limit is applied after this step (~60 s)",
    ),
    (
        "terminate only on a stall, not on wall contact or heading",
        """        # Termination judgement #########################
        episode_terminate = False
        if track.min() < 0:  # Episode is terminated if the car is out of track
            reward = - 1
            episode_terminate = True
            client.R.d['meta'] = True

        if self.terminal_judge_start < self.time_step: # Episode terminates if the progress of agent is small
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

        if np.cos(obs['angle']) < 0: # Episode is terminated if the agent runs backward
            episode_terminate = True
            client.R.d['meta'] = True

""",
        """        # Termination judgement #########################
        # Only a stall ends the episode: after terminal_judge_start steps, if
        # forward progress (speedX * cos(angle)) drops below the limit.  Wall
        # contact and momentary backward heading do not, because the controller
        # is expected to recover from both.
        episode_terminate = False
        if self.terminal_judge_start < self.time_step:
            if progress < self.termination_limit_progress:
                episode_terminate = True
                client.R.d['meta'] = True

""",
    ),
    (
        "enable the snakeoil gear thresholds upstream leaves commented out",
        """            #  Automatic Gear Change by Snakeoil is possible
            action_torcs['gear'] = 1
            \"\"\"
            if client.S.d['speedX'] > 50:
                action_torcs['gear'] = 2
            if client.S.d['speedX'] > 80:
                action_torcs['gear'] = 3
            if client.S.d['speedX'] > 110:
                action_torcs['gear'] = 4
            if client.S.d['speedX'] > 140:
                action_torcs['gear'] = 5
            if client.S.d['speedX'] > 170:
                action_torcs['gear'] = 6
            \"\"\"
""",
        """            #  Automatic gear change (snakeoil reference thresholds).
            #  Upstream leaves these commented out and locks the car in gear 1.
            spd = client.S.d['speedX']
            if spd > 170:
                action_torcs['gear'] = 6
            elif spd > 140:
                action_torcs['gear'] = 5
            elif spd > 110:
                action_torcs['gear'] = 4
            elif spd > 80:
                action_torcs['gear'] = 3
            elif spd > 50:
                action_torcs['gear'] = 2
            else:
                action_torcs['gear'] = 1
""",
    ),
    (
        # There are two of these; only the live one in reset() is patched, so the
        # anchor reaches back to the preceding line to stay unambiguous.  The
        # other sits inside a triple-quoted block in __init__ and never runs.
        "connect on the competition SCR port",
        """                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3101, vision=self.vision)  # Open new UDP in vtorcs""",
        """                print("### TORCS is RELAUNCHED ###")

        # Modify here if you use multiple tracks in the environment
        self.client = snakeoil3.Client(p=3001, vision=self.vision)  # Open new UDP in TORCS""",
    ),
    (
        "do not kill TORCS on env.end()",
        """    def end(self):
        os.system('pkill torcs')
""",
        """    def end(self):
        pass  # the caller owns the TORCS process
""",
    ),
    (
        "resolve autostart.sh next to this file, not next to the cwd",
        """        time.sleep(0.5)
        os.system('sh autostart.sh')
        time.sleep(0.5)

    def agent_to_torcs(self, u):""",
        """        time.sleep(0.5)
        os.system('sh %s' % os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'autostart.sh'))
        time.sleep(0.5)

    def agent_to_torcs(self, u):""",
    ),
    (
        "TORCS rejects a float gear",
        "            torcs_action.update({'gear': u[2]})",
        "            torcs_action.update({'gear': int(u[2])})",
    ),
]


# --------------------------------------------------------------------------
# snakeoil3_gym.py
# --------------------------------------------------------------------------

SNAKEOIL_EDITS = [
    (
        "fail instead of relaunching TORCS behind the caller's back",
        """                    print("relaunch torcs")
                    os.system('pkill torcs')
                    time.sleep(1.0)
                    if self.vision is False:
                        os.system('torcs -nofuel -nodamage -nolaptime &')
                    else:
                        os.system('torcs -nofuel -nodamage -nolaptime -vision &')

                    time.sleep(1.0)
                    os.system('sh autostart.sh')
                    n_fail = 5
""",
        """                    raise ConnectionError(
                        'TORCS not responding on port %d' % self.port)
""",
    ),
]

TARGETS = {
    "gym_torcs.py": GYM_TORCS_EDITS,
    "snakeoil3_gym.py": SNAKEOIL_EDITS,
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def apply_edits(name, text, edits):
    for label, old, new in edits:
        n = text.count(old)
        if n != 1:
            raise SystemExit(
                "%s: cannot apply edit %r -- the anchor text was found %d times,\n"
                "expected exactly 1.  The source is not the pinned upstream commit."
                % (name, label, n)
            )
        text = text.replace(old, new)
    return text


def main(argv=None):
    with open(LOCK) as f:
        lock = json.load(f)["upstream_bridge"]

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument(
        "--src",
        required=True,
        help="clean upstream gym_torcs checkout at commit %s" % lock["commit"][:12],
    )
    ap.add_argument(
        "--out",
        default=os.path.join(_REPO, "third_party", "gym_torcs"),
        help="where to write the prepared bridge",
    )
    ap.add_argument(
        "--force", action="store_true", help="overwrite an existing output directory"
    )
    args = ap.parse_args(argv)

    if os.path.exists(args.out):
        if not args.force:
            raise SystemExit("%s already exists; pass --force to replace it" % args.out)
        shutil.rmtree(args.out)

    # Verify the inputs before touching anything.
    for name, expect in lock["sha256"].items():
        src = os.path.join(args.src, name)
        if not os.path.isfile(src):
            raise SystemExit("missing from --src: %s" % src)
        got = sha256(src)
        if got != expect:
            raise SystemExit(
                "%s does not match the pinned upstream commit.\n"
                "  expected sha256 %s\n"
                "  got            %s\n"
                "Check out %s at %s and try again."
                % (name, expect, got, lock["repository"], lock["commit"])
            )

    os.makedirs(args.out)
    for name, edits in TARGETS.items():
        with open(os.path.join(args.src, name), encoding="utf-8") as f:
            text = f.read()
        text = apply_edits(name, text, edits)
        with open(os.path.join(args.out, name), "w", encoding="utf-8") as f:
            f.write(text)
        print("  %-18s %d edit(s) applied" % (name, len(edits)))

    for name in VERBATIM:
        src = os.path.join(args.src, name)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(args.out, name))
            print("  %-18s copied unchanged" % name)

    print("\nbridge ready: %s" % args.out)
    print("point the runner at it with GYM_TORCS_DIR=%s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
