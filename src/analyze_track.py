"""
analyze_track.py  --  Parse TORCS track XML and produce segment analysis.

Outputs:
  results/corkscrew_segments.json  -- machine-readable segment list
  docs/corkscrew_analysis.md       -- human-readable corner map

Usage:
  python scripts/analyze_track.py
  python scripts/analyze_track.py --xml C:/torcs/torcs/tracks/road/corkscrew/corkscrew.xml
"""

import argparse
import json
import math
import os
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

BASE = Path(__file__).parent.parent

# Path this was originally run with.  Override with --xml on any other install;
# the file ships with TORCS as tracks/road/corkscrew/corkscrew.xml.
DEFAULT_XML = "C:/torcs/torcs/tracks/road/corkscrew/corkscrew.xml"
# distRaced offset: TORCS counts from grid start, XML from track start
GRID_OFFSET = 85.2  # 3608.0 - 3522.8
TRACK_LAP_M = 3608.0


def parse_segments(xml_path: str) -> list[dict]:
    with open(xml_path, encoding="utf-8", errors="replace") as f:
        content = f.read()
    content = re.sub(r"<!DOCTYPE[^>]*\[.*?\]>", "", content, flags=re.DOTALL)
    content = content.replace("&default-surfaces;", "")

    root = ET.fromstring(content)
    segments = []
    dist = 0.0

    for sec in root.iter("section"):
        if sec.get("name") != "Track Segments":
            continue
        for seg in sec:
            nums, strings = {}, {}
            for child in seg:
                k = child.get("name", "")
                v = child.get("val", "")
                if child.tag == "attnum":
                    try:
                        nums[k] = float(v)
                    except ValueError:
                        pass
                elif child.tag == "attstr":
                    strings[k] = v

            seg_type = strings.get("type", "")
            if seg_type == "str":
                lg = nums.get("lg", 0.0)
                radius = 0.0
                arc = 0.0
            elif seg_type in ("lft", "rgt"):
                arc = nums.get("arc", 0.0)
                radius = nums.get("radius", 0.0)
                lg = arc * math.pi / 180.0 * radius
            else:
                continue

            if lg <= 0:
                continue

            dist += lg
            segments.append(
                {
                    "name": seg.get("name", ""),
                    "type": seg_type,
                    "length_m": round(lg, 2),
                    "radius_m": round(radius, 1) if radius else None,
                    "arc_deg": round(arc, 1) if arc else None,
                    "dist_end_xml": round(dist, 1),
                    "dist_end_race": round(dist + GRID_OFFSET, 1),
                }
            )

    return segments


def write_json(segments: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(exist_ok=True)
    data = {
        "track": "Corkscrew (Laguna Seca style)",
        "xml_length_m": segments[-1]["dist_end_xml"],
        "race_length_m": TRACK_LAP_M,
        "grid_offset_m": GRID_OFFSET,
        "segments": segments,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved: {out_path}")


def write_markdown(segments: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(exist_ok=True)

    tight = [
        s
        for s in segments
        if s["type"] != "str" and s["radius_m"] and s["radius_m"] < 100
    ]
    tight_sorted = sorted(tight, key=lambda x: x["radius_m"])

    lines = [
        "# Corkscrew Track Analysis",
        "",
        f"Track length: {TRACK_LAP_M} m (grid offset ≈ {GRID_OFFSET:.0f} m from XML start)",
        f"Total segments: {len(segments)}",
        "",
        "## Key for CMA-ES sector control",
        "",
        "| distRaced (m) | Segment | Type | R (m) | Length (m) | Notes |",
        "|---|---|---|---|---|---|",
    ]

    # Final sector: 2800m onwards
    for s in segments:
        dr = s["dist_end_race"]
        if dr < 2800:
            continue
        t = {"str": "STRAIGHT", "lft": "LEFT", "rgt": "RIGHT"}[s["type"]]
        r = f"{s['radius_m']:.0f}" if s["radius_m"] else "—"
        note = ""
        if s["name"] == "s46-1" or s["name"] == "s46-2":
            note = "← braking zone (switch here)"
        elif s["name"] in ("s48-1", "s48-2"):
            note = "← **BOTTLENECK** R=18-20m"
        elif s["name"] in ("s45-1", "s45-2"):
            note = "medium corner before straight"
        elif s["name"] == "s50":
            note = "finish straight"
        lines.append(
            f"| {dr:.0f} | {s['name']} | {t} | {r} | {s['length_m']:.0f} | {note} |"
        )

    lines += [
        "",
        "## All tight corners (radius < 100 m), by radius",
        "",
        "| distRaced (m) | Segment | Dir | R (m) | Arc (°) | Length (m) |",
        "|---|---|---|---|---|---|",
    ]
    for s in tight_sorted:
        dr = s["dist_end_race"]
        d = {"lft": "LEFT", "rgt": "RIGHT"}[s["type"]]
        lines.append(
            f"| {dr:.0f} | {s['name']} | {d} "
            f"| {s['radius_m']:.0f} | {s['arc_deg']:.0f} | {s['length_m']:.0f} |"
        )

    lines += [
        "",
        "## Sector switch recommendation (for cma_sector.py)",
        "",
        "```",
        "s45-1/s45-2  RIGHT R=57m   at 2980-3019m  — medium corner",
        "s46 straight  202m          at 3144-3246m  ← switch_dist ~3050m here",
        "s48-1/s48-2  LEFT  R=18-20m at 3268-3286m  — BOTTLENECK (final corner)",
        "s50          STRAIGHT 225m  at 3377-3602m  — finish straight",
        "",
        "K_final = 0.9: at track[9]=200m → adaptive_speed=180 < 200 → braking starts on s46",
        "K_final = 0.7: at track[9]=200m → adaptive_speed=140 → stronger braking",
        "```",
        "",
        "## Full segment list",
        "",
        "| distRaced (m) | Segment | Type | R (m) | Arc (°) | Length (m) |",
        "|---|---|---|---|---|---|",
    ]

    for s in segments:
        dr = s["dist_end_race"]
        t = {"str": "STRAIGHT", "lft": "LEFT", "rgt": "RIGHT"}[s["type"]]
        r = f"{s['radius_m']:.0f}" if s["radius_m"] else "—"
        a = f"{s['arc_deg']:.0f}" if s["arc_deg"] else "—"
        lines.append(
            f"| {dr:.0f} | {s['name']} | {t} | {r} | {a} | {s['length_m']:.0f} |"
        )

    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", default=DEFAULT_XML, help="Path to corkscrew.xml")
    args = parser.parse_args()

    if not os.path.exists(args.xml):
        print(f"Track XML not found: {args.xml}")
        print("Pass --xml <path to corkscrew.xml> from your own TORCS install.")
        sys.exit(1)

    print(f"Parsing: {args.xml}")
    segments = parse_segments(args.xml)
    print(
        f"  Found {len(segments)} segments, length {segments[-1]['dist_end_xml']:.1f} m"
    )

    write_json(segments, BASE / "results" / "corkscrew_segments.json")
    write_markdown(segments, BASE / "docs" / "corkscrew_analysis.md")

    # Quick summary
    tight = [
        s
        for s in segments
        if s["type"] != "str" and s["radius_m"] and s["radius_m"] < 60
    ]
    print(f"\nTightest corners (R < 60m):")
    for s in sorted(tight, key=lambda x: x["radius_m"]):
        d = {"lft": "L", "rgt": "R"}[s["type"]]
        print(
            f"  {s['dist_end_race']:6.0f}m  {s['name']:8s}  {d}  R={s['radius_m']:5.1f}m"
        )


if __name__ == "__main__":
    main()
