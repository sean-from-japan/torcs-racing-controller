"""
make_figure.py -- render figures/lap_time_progression.svg from results/.

Reads the stage JSON files so the chart cannot drift away from the committed
evidence: every bar length comes from a `best_lap_2plus_s` field on disk.  The
one stage with no artefact is drawn hatched and labelled as such rather than
silently omitted or silently included.

    python src/make_figure.py

Stdlib only -- no matplotlib, so the figure regenerates anywhere.
"""

import json
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO, "results")
OUT = os.path.join(_REPO, "figures", "lap_time_progression.svg")

# (label, method, results file or None, note)
STAGES = [
    ("Rule-based baseline", "supplied snakeoil driver", "stage0_baseline_snakeoil.json", ""),
    ("CMA-ES, 3 params", "A, B, C", "stage1_cma_3param.json", ""),
    ("CMA-ES, 5 params", "+ K, T", "stage2_cma_5param.json", ""),
    ("CMA-ES, 6 params", "+ D (steering deadband)", "stage3_cma_6param_deadband.json", ""),
    ("CMA-ES, 8 params + s35 cap", "+ K_final, switch_dist, C_s35", "stage4_cma_8param_sector_s35.json", ""),
    ("Residual NN + ARS", "throttle + 0.2 x NN(obs)", None, "trained weights not archived"),
]

# Reported in the project log for the stage with no committed artefact.
REPORTED_NN_LAP = 106.630

W, ROW_H, TOP, LEFT, BAR_X = 900, 62, 74, 24, 330
BAR_W = 500


def load_time(filename):
    with open(os.path.join(RESULTS, filename)) as f:
        return float(json.load(f)["best_lap_2plus_s"])


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build():
    rows = []
    for label, method, filename, note in STAGES:
        t = load_time(filename) if filename else REPORTED_NN_LAP
        rows.append((label, method, t, filename is not None, note))

    slowest = max(r[2] for r in rows)
    baseline = rows[0][2]
    height = TOP + ROW_H * len(rows) + 54

    out = []
    out.append(
        '<svg xmlns="http://www.w3.org/2000/svg" width="%d" height="%d" '
        'viewBox="0 0 %d %d" font-family="Helvetica Neue, Helvetica, Arial, sans-serif">'
        % (W, height, W, height)
    )
    out.append('<rect width="%d" height="%d" fill="#ffffff"/>' % (W, height))
    out.append(
        '<pattern id="unverified" width="7" height="7" patternUnits="userSpaceOnUse" '
        'patternTransform="rotate(45)">'
        '<rect width="7" height="7" fill="#c9d4e3"/>'
        '<rect width="3" height="7" fill="#eef2f7"/></pattern>'
    )

    out.append(
        '<text x="%d" y="30" font-size="19" font-weight="600" fill="#11161d">'
        "Corkscrew best warm lap, by optimisation stage</text>" % LEFT
    )
    out.append(
        '<text x="%d" y="52" font-size="13" fill="#5a6673">'
        "TORCS, ~3,608 m lap. Lower is better; lap 1 starts from rest and is not counted."
        "</text>" % LEFT
    )

    for i, (label, method, t, verified, note) in enumerate(rows):
        y = TOP + i * ROW_H
        bar = BAR_W * t / slowest
        fill = "#2f6fb5" if verified else "url(#unverified)"

        out.append(
            '<text x="%d" y="%d" font-size="13.5" font-weight="600" fill="#11161d">%s</text>'
            % (LEFT, y + 14, esc(label))
        )
        out.append(
            '<text x="%d" y="%d" font-size="11.5" fill="#6b7684">%s</text>'
            % (LEFT, y + 31, esc(method))
        )
        out.append(
            '<rect x="%d" y="%d" width="%.1f" height="22" rx="3" fill="%s"/>'
            % (BAR_X, y + 2, bar, fill)
        )
        if not verified:
            out.append(
                '<rect x="%d" y="%d" width="%.1f" height="22" rx="3" fill="none" '
                'stroke="#8fa3bb" stroke-width="1" stroke-dasharray="4 3"/>'
                % (BAR_X, y + 2, bar)
            )
        out.append(
            '<text x="%.1f" y="%d" font-size="13" font-weight="600" fill="#11161d">'
            "%.3f s</text>" % (BAR_X + bar + 9, y + 18, t)
        )
        if i > 0:
            out.append(
                '<text x="%.1f" y="%d" font-size="11.5" fill="#6b7684">'
                "-%.1f%% vs baseline</text>"
                % (BAR_X + bar + 9, y + 33, 100.0 * (baseline - t) / baseline)
            )
        if note:
            out.append(
                '<text x="%d" y="%d" font-size="11" fill="#a2560d">%s</text>'
                % (LEFT, y + 45, esc(note))
            )

    foot_y = TOP + ROW_H * len(rows) + 22
    out.append(
        '<rect x="%d" y="%d" width="13" height="11" rx="2" fill="#2f6fb5"/>' % (LEFT, foot_y - 10)
    )
    out.append(
        '<text x="%d" y="%d" font-size="11.5" fill="#5a6673">'
        "parameters committed in results/</text>" % (LEFT + 19, foot_y)
    )
    out.append(
        '<rect x="%d" y="%d" width="13" height="11" rx="2" fill="url(#unverified)" '
        'stroke="#8fa3bb" stroke-width="1"/>' % (LEFT + 240, foot_y - 10)
    )
    out.append(
        '<text x="%d" y="%d" font-size="11.5" fill="#5a6673">'
        "measured in the project; trained weights not archived</text>" % (LEFT + 259, foot_y)
    )
    out.append("</svg>")
    return "\n".join(out)


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(build())
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
