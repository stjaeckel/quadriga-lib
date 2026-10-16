#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
"""Random-incidence absorption of a locally reacting face (md/acoustic.md 2.6, 9; M6).

A face of normalized impedance z = r + j x reflects (z c - 1)/(z c + 1) at the cosine c in air. Its Paris integral
(random incidence, 0 to 90 deg, weight sin 2 theta) has the closed form

    P = (8 r/|z|^2) (1 - (r/|z|^2) ln(1 + 2 r + |z|^2) + ((r^2 - x^2)/(x |z|^2)) atan(x/(1 + r)))

(the x -> 0 limit is taken where x = 0). A mass-law face with a peak is the series impedance of a resonator per Z0,
z = R' + j (1 + R') |tfQ| (f/tfF - tfF/f) with tfP = 4 R'/(1 + R')^2 (over-damped R' >= 1, or 1/R' where tfQ < 0; tfP
above one is R' = 1), behind the pad of its power-law part tf: alpha = tf + (1 - tf) P. The field value (0 to 78 deg)
is integrated numerically.

  python3 tools/acoustic/paris_cf.py face --tfP 0.25 --tfF 150 --tfQ 0.75 --tf 0.05 --f 63 125 150 250 500
  python3 tools/acoustic/paris_cf.py check
"""
import argparse, cmath, math, sys


def paris(z):
    """Paris integral 0-90 deg of 1 - |(z c - 1)/(z c + 1)|^2 for a normalized impedance z (closed form)"""
    r, x = z.real, z.imag
    a2 = r * r + x * x
    last = (r * r - x * x) / (x * a2) * math.atan(x / (1.0 + r)) if abs(x) > 1e-12 else (r / a2) * (r / (1.0 + r))
    return (8.0 * r / a2) * (1.0 - (r / a2) * math.log(1.0 + 2.0 * r + a2) + last)


def alpha(z, c, tm=False):
    w = z / c if tm else z * c
    return 1.0 - abs((w - 1.0) / (w + 1.0)) ** 2


def integral(fun, tmax=90.0, n=4000):
    """the same weight numerically, 0 to tmax deg, normalized by the integral of sin 2 theta over that range"""
    s = w = 0.0
    for i in range(n):
        th = math.radians((i + 0.5) * tmax / n)
        k = math.sin(2.0 * th)
        s += fun(th) * k
        w += k
    return s / w


def zeta(f, tfP, tfF, tfQ):
    a = min(max(tfP, 1e-300), 1.0)
    hi = ((2.0 - a) + 2.0 * math.sqrt(1.0 - a)) / a
    Rp = 1.0 / hi if tfQ < 0.0 else hi
    return complex(Rp, (1.0 + Rp) * abs(tfQ) * (f / tfF - tfF / f))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    fa = sub.add_parser('face', help='a peaked mass-law face: normal-incidence, Paris and field values per frequency')
    fa.add_argument('--tfP', type=float, required=True)
    fa.add_argument('--tfF', type=float, required=True, help='Hz')
    fa.add_argument('--tfQ', type=float, required=True, help='Q_alpha; negative for the under-damped root')
    fa.add_argument('--tf', type=float, default=0.0, help='the power-law part at these frequencies (one value)')
    fa.add_argument('--f', type=float, nargs='+', required=True, help='Hz')
    sub.add_parser('check', help='the closed form against the numerical integral and Jeong\'s values')
    a = ap.parse_args()
    if a.cmd == 'face':
        print('f_Hz      Re(zeta)    Im(zeta)   alpha_0   Paris_0-90  field_0-78')
        for f in a.f:
            z = zeta(f, a.tfP, a.tfF, a.tfQ)
            t = min(1.0, max(0.0, a.tf))
            a0 = t + (1 - t) * alpha(z, 1.0)
            pa = t + (1 - t) * paris(z)
            fi = t + (1 - t) * integral(lambda th: alpha(z, math.cos(th)), 78.0)
            print('%-8g %10.4f %11.4f %9.4f %11.4f %11.4f' % (f, z.real, z.imag, a0, pa, fi))
        return 0
    worst = 0.0
    for z in (complex(13.9, 0), complex(13.9, 30), complex(13.9, -60), complex(0.07, 0.5), complex(0.07, -3), complex(1.567, 0),
              complex(3.4, 10), complex(40, 0), complex(10, 0), complex(4, 0)):
        num = integral(lambda th: alpha(z, math.cos(th)), 90.0, 20000)
        worst = max(worst, abs(num - paris(z)))
    print('closed form against the numerical integral, 10 impedances: max difference %.1e' % worst)
    print('Jeong: maximum %.6f at 1.567 (0.951); impedances 40 / 10 / 4: %.3f / %.3f / %.3f (0.17 / 0.49 / 0.79)'
          % (paris(complex(1.567, 0)), paris(complex(40, 0)), paris(complex(10, 0)), paris(complex(4, 0))))
    ok = worst < 1e-8 and abs(paris(complex(1.567, 0)) - 0.951) < 5e-4
    print('check passed' if ok else 'CHECK FAILED')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
