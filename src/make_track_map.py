"""
make_track_map.py -- render the Corkscrew track maps into figures/.

    figures/corkscrew_corner_map.svg       geometry, the tight corners, the crash band
    figures/corkscrew_zones_steering.svg   where the steering law is overridden
    figures/corkscrew_zones_speed.svg      where the speed law is overridden

Three maps rather than one, so every label can be set large enough to read.  The
tables and the explanations that go with them belong in the README, not in the
images.

All three are reconstructed from `results/corkscrew_segments.json`, which
`analyze_track.py` produces from the TORCS track definition, and the zone maps
read their distances from `results/stage4_cma_8param_sector_s35.json`.  Nothing
is drawn by hand: if a parameter changes, the map moves.

Positions are in the race-distance frame (TORCS `distRaced`), the same frame the
controller switches sectors in and the same one used by
`docs/corkscrew_analysis.md`.  That is the XML frame plus the grid offset of
85.2 m recorded in the segment file.

The centreline is integrated from segment lengths, radii and arc angles, so it is
a plan view of the track's own geometry rather than a trace of a driven lap.  It
does not close perfectly -- the two ends land 89 m apart on the finish straight,
2.5% of the lap -- because the XML stores each segment to one decimal place and
the map ignores elevation.  Corner positions are unaffected.

    python src/make_track_map.py

Stdlib only -- no matplotlib, so the figures regenerate anywhere.
"""

import json
import math
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO, "results")
FIGURES = os.path.join(_REPO, "figures")

SEGMENTS = os.path.join(RESULTS, "corkscrew_segments.json")
PARAMS = os.path.join(RESULTS, "stage4_cma_8param_sector_s35.json")

# Zone bounds fixed by hand rather than searched.  These are the constants in
# controller.py, repeated here so the figure script does not depend on the
# control law's module layout.
D_CORK_RAMP, D_CORK_EXIT = 900.0, 1640.0
D_S35_RAMP, D_S35_EXIT = 2300.0, 2510.0
POSTSTART_DIST_M = 500.0

W, H = 820, 780
PLOT = (60, 96, 700, 600)  # x, y, width, height of the map area
ARC_STEPS = 24

TITLE_SIZE = 21
LABEL_SIZE = 17
SOURCE_SIZE = 12.5
LINE_H = 21

INK = "#11161d"
MUTED = "#4b5563"
FAINT = "#8b96a3"
BLUE = "#2f6fb5"
PALE = "#c2ccd9"
RED = "#c0392b"
AMBER = "#a2560d"
GREEN = "#2e7d5b"
PURPLE = "#6b4fa0"


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------


def centreline():
    """Integrate the segment table into (x, y, race distance) points."""
    with open(SEGMENTS) as f:
        data = json.load(f)

    x = y = 0.0
    heading = math.pi / 2  # arbitrary: only the shape matters
    dist = data["grid_offset_m"]
    points = [(x, y, dist)]

    for seg in data["segments"]:
        length = seg["length_m"]
        radius = seg["radius_m"]
        arc = math.radians(seg["arc_deg"] or 0.0)

        if seg["type"] == "str":
            for j in range(1, ARC_STEPS + 1):
                f = j / ARC_STEPS
                points.append(
                    (x + length * f * math.cos(heading), y + length * f * math.sin(heading),
                     dist + length * f)
                )
            x += length * math.cos(heading)
            y += length * math.sin(heading)
        else:
            turn = 1.0 if seg["type"] == "lft" else -1.0
            cx = x + radius * math.cos(heading + turn * math.pi / 2)
            cy = y + radius * math.sin(heading + turn * math.pi / 2)
            a0 = math.atan2(y - cy, x - cx)
            for j in range(1, ARC_STEPS + 1):
                a = a0 + turn * arc * j / ARC_STEPS
                points.append(
                    (cx + radius * math.cos(a), cy + radius * math.sin(a),
                     dist + length * j / ARC_STEPS)
                )
            heading += turn * arc
            x = cx + radius * math.cos(a0 + turn * arc)
            y = cy + radius * math.sin(a0 + turn * arc)

        dist += length

    return points


def projector(points):
    """Fit the track into the plot rectangle; return world -> pixel."""
    px, py, pw, ph = PLOT
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    scale = min(pw / (max(xs) - min(xs)), ph / (max(ys) - min(ys)))
    ox = px + (pw - (max(xs) - min(xs)) * scale) / 2 - min(xs) * scale
    oy = py + (ph + (max(ys) - min(ys)) * scale) / 2 + min(ys) * scale

    def to_px(x, y):
        return ox + x * scale, oy - y * scale  # SVG y grows downwards

    return to_px


def band(points, to_px, lo, hi):
    """Pixel path for the stretch of track between two race distances."""
    return [to_px(p[0], p[1]) for p in points if lo <= p[2] <= hi]


def at(points, to_px, dist):
    """Pixel anchor for one race distance."""
    p = min(points, key=lambda q: abs(q[2] - dist))
    return to_px(p[0], p[1])


def path(coords):
    return "M " + " L ".join("%.1f %.1f" % c for c in coords)


# --------------------------------------------------------------------------
# drawing
# --------------------------------------------------------------------------


def callout(out, anchor, dx, dy, lines, colour, side):
    """A dot on the track, a leader line, and a short label."""
    ax, ay = anchor
    tx, ty = ax + dx, ay + dy
    out.append('<circle cx="%.1f" cy="%.1f" r="5" fill="%s"/>' % (ax, ay, colour))
    out.append(
        '<path d="M %.1f %.1f L %.1f %.1f" stroke="%s" stroke-width="1.2" fill="none"/>'
        % (ax, ay, tx, ty, colour)
    )
    for i, line in enumerate(lines):
        out.append(
            '<text x="%.1f" y="%.1f" font-size="%s" font-weight="%s" fill="%s" '
            'text-anchor="%s">%s</text>'
            % (
                tx + (7 if side == "start" else -7),
                ty + 6 + i * LINE_H,
                LABEL_SIZE,
                "600" if i == 0 else "400",
                colour if i == 0 else MUTED,
                side,
                esc(line),
            )
        )


def build_map(points, to_px, title, bands, callouts, source, base=BLUE):
    """One map: a title, the centreline, coloured bands, callouts, a source line."""
    out = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" viewBox="0 0 %d %d" '
        'font-family="Helvetica Neue, Helvetica, Arial, sans-serif">' % (W, H, W, H),
        '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
        '<text x="30" y="36" font-size="%s" font-weight="600" fill="%s">%s</text>'
        % (TITLE_SIZE, INK, esc(title)),
    ]

    track = [to_px(p[0], p[1]) for p in points]
    out.append(
        '<path d="%s" stroke="%s" stroke-width="5" fill="none" stroke-linecap="round" '
        'stroke-linejoin="round"/>' % (path(track), base)
    )

    for lo, hi, colour in bands:
        coords = band(points, to_px, lo, hi)
        if len(coords) > 1:
            out.append(
                '<path d="%s" stroke="%s" stroke-width="9" fill="none" '
                'stroke-linecap="round" stroke-linejoin="round"/>' % (path(coords), colour)
            )

    sx, sy = at(points, to_px, 3607.9)
    out.append('<circle cx="%.1f" cy="%.1f" r="6.5" fill="%s"/>' % (sx, sy, INK))
    out.append(
        '<text x="%.1f" y="%.1f" font-size="%s" font-weight="600" fill="%s" '
        'text-anchor="end">start / finish</text>' % (sx - 13, sy + 6, LABEL_SIZE, INK)
    )

    for dist, dx, dy, lines, colour, side in callouts:
        callout(out, at(points, to_px, dist), dx, dy, lines, colour, side)

    out.append(
        '<text x="30" y="%d" font-size="%s" fill="%s">%s</text>'
        % (H - 20, SOURCE_SIZE, FAINT, esc(source))
    )
    out.append("</svg>")
    return "\n".join(out)


SEGMENT_SOURCE = (
    "Plan view integrated from results/corkscrew_segments.json. It ignores elevation, "
    "so the lap's two ends land 89 m apart on the finish straight."
)
PARAM_SOURCE = (
    "Zone boundaries read from results/stage4_cma_8param_sector_s35.json and the fixed "
    "constants in src/controller.py."
)


def corner_map(points, to_px):
    return build_map(
        points,
        to_px,
        "Corkscrew — the corners that mattered",
        [(2338.9, 2510.0, RED)],
        [
            (598, -34, -14, ["s6 + s8", "R 33 m, then 24 m"], INK, "end"),
            (1603, 12, -54, ["s20 + s21 — the Corkscrew", "R 72 m, then 48 m"], INK, "start"),
            (1940, 28, -22, ["s24", "R 34 m"], INK, "start"),
            (2461, -185, 44, ["crash band, 2,339 – 2,510 m",
                              "s35-1 R 32 m, s35-2 R 22 m",
                              "s37 R 14 m — track minimum"], RED, "start"),
            (3277, -34, 26, ["s48 — final hairpin", "R 20 m, then 18 m"], INK, "end"),
        ],
        SEGMENT_SOURCE,
    )


def steering_map(points, to_px):
    return build_map(
        points,
        to_px,
        "Where the steering target leaves the centre line",
        [(D_CORK_RAMP, D_CORK_EXIT, PURPLE), (D_S35_RAMP, D_S35_EXIT, PURPLE)],
        [
            (1270, -30, 10, ["racing line, s20 + s21", "900 – 1,640 m"], PURPLE, "end"),
            (2300, -169, 100, ["racing line, s35 chicane", "2,300 – 2,510 m"], PURPLE,
             "start"),
        ],
        PARAM_SOURCE,
        base=PALE,
    )


def speed_map(points, to_px, p):
    return build_map(
        points,
        to_px,
        "Where the speed law is overridden",
        [
            (points[0][2], POSTSTART_DIST_M, AMBER),
            (p["s35_start"], D_S35_EXIT, RED),
            (p["switch_dist"], p["back_dist"], BLUE),
            (p["back_dist"], 3607.9, GREEN),
        ],
        [
            (300, -40, -30, ["post-start brake, 0 – 500 m",
                             "full brake above 140 km/h"], AMBER, "end"),
            (2430, -175, 44, ["s35 speed cap, 2,339 – 2,510 m",
                              "ceiling of 71.4 km/h"], RED, "start"),
            (3032, -26, -46, ["K_final braking, 3,032 – 3,305 m",
                              "lookahead gain 2.171 → 0.784"], BLUE, "end"),
            (3480, -34, -30, ["sprint, 3,305 m to the line",
                              "throttle forced to 1.0"], GREEN, "end"),
        ],
        PARAM_SOURCE,
        base=PALE,
    )


def main():
    os.makedirs(FIGURES, exist_ok=True)
    points = centreline()
    to_px = projector(points)

    with open(PARAMS) as f:
        params = json.load(f)

    for name, svg in (
        ("corkscrew_corner_map.svg", corner_map(points, to_px)),
        ("corkscrew_zones_steering.svg", steering_map(points, to_px)),
        ("corkscrew_zones_speed.svg", speed_map(points, to_px, params)),
    ):
        out_path = os.path.join(FIGURES, name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg)
        print("wrote %s" % out_path)


if __name__ == "__main__":
    main()
