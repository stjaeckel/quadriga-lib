#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
"""Validate the material rig's angle and frequency grid once against a fine grid (not run in the suite).

The rig (tests/catch2_tests/test_acoustic_materials.cpp) forms random-incidence values from a coarse grid: N midpoint
angles over 0-90 deg (weight cos sin) x n tones per third (1: the nominal center), octaves 31.5 Hz - 8 kHz, a third the
mean of its tones, an octave the mean of its three thirds (absorption arithmetic, transmission in energy). This script
builds the same stack from rows of a material CSV (sheets in the plane x = 0 in the order given, the room side first;
traced solids behind them), runs every angle through band_probe (one ray through ray_progress, R and T summed
coherently over the paths, thin_slab_threshold 0, the library's sound speed) and prints the octave absorption and
transmission loss on each grid with its deviation from the first (the reference).

  tools/acoustic/build_band_probe.sh /tmp/band_probe
  python3 -I tools/acoustic/grid_probe.py --probe /tmp/band_probe --csv data/materials_acoustic.csv \\
      --layer concrete_cast_152mm:sheet --grids 192x6 50x1 192x3
  python3 -I tools/acoustic/grid_probe.py --probe /tmp/band_probe --csv rows.csv --layer wool:sheet:0.05 \\
      --layer concrete_cast_152mm:solid --grids 3072x12 50x1

--layer NAME:sheet|solid[:THK]: a row of the CSV (THK in m; the row's thk where omitted). The first grid is the
reference; 3072x12 converges coincidence ridges of light panels (about 13 s per row on one thread in C++).
"""
import argparse, csv, math, subprocess, sys

COLS = ["a", "b", "c", "d", "e", "f", "g", "h", "thk", "tf", "tfB", "m", "fRef", "resF", "resQ", "resS", "att", "coiF", "coiQ",
        "alpha", "tfP", "tfF", "tfQ", "side", "attB", "alphaB"]  # band_probe's order
DFLT = {"fRef": 1.0, "a": 1.0, "e": 1.0, "side": 3.29}
THIRDS = [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150,
          4000, 5000, 6300, 8000, 10000]
OCT = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]


def rows(fn):
    out = {}
    with open(fn, newline="") as fh:
        for r in csv.DictReader(fh):
            out[r["name"].strip()] = {k.strip(): float(v) if v.strip() else 0.0 for k, v in r.items() if k != "name"}
    return out


def material_line(row, kind, thk):
    vals = []
    for k in COLS:
        v = row.get(k, DFLT.get(k, 0.0))
        if k == "thk":
            v = thk if kind == "sheet" else 0.0
        vals.append(repr(float(v)))
    return " ".join(vals)


def stack(table, specs):
    layers = []
    for s in specs:
        p = s.split(":")
        r = table[p[0]]
        layers.append((r, p[1], float(p[2]) if len(p) > 2 else r.get("thk", 0.0)))
    faces, x = [], 0.0
    for i, (r, kind, t) in enumerate(layers):
        if kind == "sheet":
            faces.append((0.0, i + 1, 0))
    for i, (r, kind, t) in enumerate(layers):
        if kind == "solid":
            faces.append((x, i + 1, 0))
            x += t
            faces.append((x, i + 1, 1))
    mats = " ".join(material_line(r, kind, t) for r, kind, t in layers)
    fstr = " ".join("%r %d %d" % f for f in faces)
    return len(layers), mats, len(faces), fstr


def grid(spec):
    na, ns = (int(v) for v in spec.lower().split("x"))
    th = [(i + 0.5) * 90.0 / na for i in range(na)]
    w = [math.cos(math.radians(t)) * math.sin(math.radians(t)) for t in th]
    sw = sum(w)
    f, third = [], []
    for k, fc in enumerate(THIRDS):
        for j in range(ns):
            f.append(fc * 2.0 ** (((j + 0.5) / ns - 0.5) / 3.0) if ns > 1 else fc)
            third.append(k)
    return th, [v / sw for v in w], f, third


def run(probe, st, spec, chunk=120):
    th, w, f, third = grid(spec)
    nm, mats, nf, fstr = st
    order = sorted(range(len(f)), key=lambda q: abs(math.log(f[q] / 125.0)))  # 125 Hz first in every chunk
    base = order[0]
    rest = [q for q in range(len(f)) if q != base]
    chunks = []
    for c in range(0, len(rest), chunk - 1):
        chunks.append([base] + rest[c:c + chunk - 1])
    lines, keys = [], []
    for ia, a in enumerate(th):
        for ch in chunks:
            lines.append("g 4 %r 0 0 0 %d %s %d %s 0 60 -120 %d %s" % (a, nm, mats, nf, fstr, len(ch), " ".join(repr(f[q]) for q in ch)))
            keys += [(ia, q) for q in ch]
    p = subprocess.run([probe], input="\n".join(lines) + "\n", capture_output=True, text=True, check=True)
    out = [l.split(",") for l in p.stdout.splitlines() if l.startswith("P,g,")]
    if len(out) != len(keys):
        raise RuntimeError("band_probe returned %d rows for %d" % (len(out), len(keys)))
    R, T = {}, {}
    for k, v in zip(keys, out):
        T[k] = float(v[10]) ** 2 + float(v[11]) ** 2
        R[k] = float(v[12]) ** 2 + float(v[13]) ** 2
    a3, t3, n3 = [0.0] * 27, [0.0] * 27, [0] * 27
    for q, fq in enumerate(f):
        al = ta = 0.0
        for ia, wa in enumerate(w):
            al += wa * (1.0 - R[(ia, q)])
            ta += wa * T[(ia, q)]
        a3[third[q]] += al
        t3[third[q]] += ta
        n3[third[q]] += 1
    a8 = [sum(a3[3 * o + k] / n3[3 * o + k] for k in range(3)) / 3.0 for o in range(9)]
    t8 = [-10.0 * math.log10(max(sum(t3[3 * o + k] / n3[3 * o + k] for k in range(3)) / 3.0, 1e-300)) for o in range(9)]
    return a8, t8


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", required=True, help="the band_probe binary (tools/acoustic/build_band_probe.sh)")
    ap.add_argument("--csv", default="data/materials_acoustic.csv")
    ap.add_argument("--layer", action="append", required=True, help="NAME:sheet|solid[:THK], room side first")
    ap.add_argument("--grids", nargs="+", default=["192x6", "50x1"], help="NAxNS; the first is the reference")
    a = ap.parse_args()
    st = stack(rows(a.csv), a.layer)
    res = [(g, run(a.probe, st, g)) for g in a.grids]
    ref = res[0][1]
    print("grid," + ",".join("a%g" % o for o in OCT) + "," + ",".join("tl%g" % o for o in OCT))
    for g, (a8, t8) in res:
        print("%s,%s,%s" % (g, ",".join("%.4f" % v for v in a8), ",".join("%.3f" % v for v in t8)))
    for g, (a8, t8) in res[1:]:
        print("%s-%s,%s,%s" % (g, res[0][0], ",".join("%+.4f" % (x - y) for x, y in zip(a8, ref[0])),
                               ",".join("%+.3f" % (x - y) for x, y in zip(t8, ref[1]))))


if __name__ == "__main__":
    sys.exit(main())
