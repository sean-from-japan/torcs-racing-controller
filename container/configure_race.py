#!/usr/bin/env python3
"""configure_race.py -- point TORCS' practice race at Corkscrew and scr_server.

TORCS reads the practice race setup from $HOME/.torcs/config/raceman/practice.xml,
which it writes itself the first time it runs.  Leaving that file to whatever the
GUI last saved is how a lap time becomes unreproducible, so this script edits the
two things the measurement depends on -- the track and the driver -- and leaves
everything else as TORCS installed it.

It edits TORCS' own file in place rather than shipping a replacement, so no TORCS
data files are redistributed here.

    python3 container/configure_race.py            # edit, then report
    python3 container/configure_race.py --check    # report only, exit 1 if wrong

Nothing here is container-specific; it works against any TORCS install.
"""

import argparse
import os
import re
import sys

DEFAULT_XML = os.path.expanduser("~/.torcs/config/raceman/practice.xml")

TRACK = "corkscrew"
CATEGORY = "road"
DRIVER = "scr_server"


def _section(text, name):
    """Return (start, end) of the body of <section name="...">, or None."""
    m = re.search(r'<section\s+name="%s"\s*>' % re.escape(name), text)
    if not m:
        return None
    depth, i = 1, m.end()
    for tag in re.finditer(r"<section\b[^>]*>|</section>", text[m.end():]):
        depth += 1 if tag.group(0).startswith("<section") else -1
        if depth == 0:
            return m.end(), m.end() + tag.start()
    return None


def _set_attstr(body, name, value):
    pat = re.compile(r'(<attstr\s+name="%s"\s+val=")([^"]*)(")' % re.escape(name))
    m = pat.search(body)
    if not m:
        raise SystemExit('practice.xml has no <attstr name="%s">' % name)
    return pat.sub(lambda _m: _m.group(1) + value + _m.group(3), body, count=1), m.group(2)


def read_setup(path):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    out = {}
    for section, keys in (("Tracks", ("name", "category")), ("Drivers", ("module",))):
        span = _section(text, section)
        if span is None:
            raise SystemExit('practice.xml has no "%s" section: %s' % (section, path))
        body = text[span[0]:span[1]]
        inner = _section(body, "1")
        if inner is None:
            raise SystemExit('practice.xml "%s" section has no entry 1' % section)
        entry = body[inner[0]:inner[1]]
        for k in keys:
            m = re.search(r'<attstr\s+name="%s"\s+val="([^"]*)"' % k, entry)
            out["%s.%s" % (section, k)] = m.group(1) if m else None
    return text, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--xml", default=DEFAULT_XML, help="TORCS practice race config")
    ap.add_argument("--check", action="store_true", help="report only, do not edit")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.xml):
        raise SystemExit(
            "%s does not exist.\nRun TORCS once so it creates ~/.torcs, then try again."
            % args.xml
        )

    text, before = read_setup(args.xml)
    want = {
        "Tracks.name": TRACK,
        "Tracks.category": CATEGORY,
        "Drivers.module": DRIVER,
    }
    wrong = {k: v for k, v in before.items() if v != want[k]}

    if args.check:
        for k in sorted(want):
            print("  %-18s %-12s (want %s)" % (k, before[k], want[k]))
        return 1 if wrong else 0

    if not wrong:
        print("already configured: %s on %s" % (DRIVER, TRACK))
        return 0

    # Edit only the Tracks entry and the Drivers focus; the driver slot itself
    # already names scr_server in a stock install.
    tstart, tend = _section(text, "Tracks")
    tbody = text[tstart:tend]
    tbody, old_track = _set_attstr(tbody, "name", TRACK)
    tbody, old_cat = _set_attstr(tbody, "category", CATEGORY)
    text = text[:tstart] + tbody + text[tend:]

    dstart, dend = _section(text, "Drivers")
    dbody = text[dstart:dend]
    if re.search(r'<attstr\s+name="focused module"', dbody):
        dbody, _ = _set_attstr(dbody, "focused module", DRIVER)
    inner = _section(dbody, "1")
    entry = dbody[inner[0]:inner[1]]
    entry, old_driver = _set_attstr(entry, "module", DRIVER)
    dbody = dbody[:inner[0]] + entry + dbody[inner[1]:]
    text = text[:dstart] + dbody + text[dend:]

    with open(args.xml, "w", encoding="utf-8") as f:
        f.write(text)

    print("configured %s" % args.xml)
    print("  track  : %s/%s -> %s/%s" % (old_cat, old_track, CATEGORY, TRACK))
    print("  driver : %s -> %s" % (old_driver, DRIVER))
    return 0


if __name__ == "__main__":
    sys.exit(main())
