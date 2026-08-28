"""
torcs_env.py -- locate the gym_torcs bridge and import `TorcsEnv` from it.

The original working copy vendored a pinned snapshot of gym_torcs inside the
repository.  That snapshot is upstream third-party code (MIT, Naoto Yoshida),
bundled with the GPL-2.0 TORCS simulator sources, so it is not redistributed
here; see the Attribution section of README.md.  Point this module at your own
checkout instead:

    export GYM_TORCS_DIR=/path/to/gym_torcs      # or pass --gym-torcs

It is the only file that knows where the bridge lives.  Nothing else in the
repository imports gym_torcs directly, which is what keeps `controller.py`
and the test suite runnable with no simulator installed.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DIR = os.path.join(_REPO_ROOT, "third_party", "gym_torcs")

_HELP = """
gym_torcs was not found at:
    {path}

Set GYM_TORCS_DIR, pass --gym-torcs <dir>, or clone the bridge into
third_party/gym_torcs:

    git clone https://github.com/ugo-nama-kun/gym_torcs third_party/gym_torcs

The directory must contain gym_torcs.py and snakeoil3_gym.py.  Two local
edits to gym_torcs.py were in force when these lap times were recorded --
they are listed in README.md and must be reapplied to reproduce them.
"""


def resolve_dir(explicit=None):
    """Return the gym_torcs directory: explicit arg, then env var, then default."""
    return explicit or os.environ.get("GYM_TORCS_DIR") or DEFAULT_DIR


def load_torcs_env(explicit=None):
    """Import and return the `TorcsEnv` class, adding its directory to sys.path.

    Raises SystemExit with setup instructions rather than an ImportError
    traceback, because a missing simulator is a setup problem, not a bug.
    """
    path = resolve_dir(explicit)
    if not os.path.isdir(path):
        raise SystemExit(_HELP.format(path=path))
    if path not in sys.path:
        sys.path.insert(0, path)
    try:
        from gym_torcs import TorcsEnv
    except ImportError as exc:
        raise SystemExit("%s\nImport failed: %s" % (_HELP.format(path=path), exc))
    return TorcsEnv
