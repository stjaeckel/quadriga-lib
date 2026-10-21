#!/usr/bin/env python3
"""
Sabine (reverberation-chamber, ISO 354) absorption to random-incidence absorption, by Jeong's method.

A one-parameter porous layer (Miki, flow resistivity sigma, known thickness, rigid backing) is fitted so that a finite
sample of the test size reproduces the chamber values; Thomasson's field impedance of the rectangular patch carries
the finite size [Thomasson_1980_AbsorptionCoefficient, Eqs. 17, 23, 26]. The output is the Paris integral (0-90 deg)
of the same layer as an infinite surface: the random-incidence coefficient a tf reference takes (md/acoustic.md 3.1,
A13). Extended reaction (the layer refracts) by default; local reaction on request.

Use only where the model fits the product and the sample is a flush patch: where sigma is measured, compare the fitted
sigma with it (md/acoustic.md 6, A13 and A14: PU foam 10.3 against 10.6 kPa s/m2, glass wool 19.0 against 13.1;
polyester and a framed stone wool did not convert). Source keys refer to the literature review in
/sjc/accoustic_research (references/index.csv).

  sabine_to_random.py convert --thk 0.05 --sample 3 4 --alpha 0.153 0.587 0.993 1.073 1.017 0.987
  sabine_to_random.py check            # reproduces the validation of md/acoustic.md 6 (A13, A14)

Octaves 125 Hz to 4 kHz, each the mean of its three third-octave centres (as the references in the review).
The field-impedance table of a sample size takes minutes on one thread; it is cached in $QD_ACOUSTIC_CACHE
(default ~/.cache/quadriga-lib/acoustic).
"""
import argparse, math, os, sys
import numpy as np

C0 = 343.0
OCT = [125, 250, 500, 1000, 2000, 4000]
THIRDS = [fc * 2 ** (k / 3) for fc in OCT for k in (-1, 0, 1)]
_GX, _GW = np.polynomial.legendre.leggauss(32)
TH = 0.25 * np.pi * (_GX + 1); TW = 0.25 * np.pi * _GW          # theta nodes on [0, pi/2]
PHI = (0.0, np.pi / 4, np.pi / 2)
CACHE = os.environ.get('QD_ACOUSTIC_CACHE', os.path.join(os.path.expanduser('~'), '.cache', 'quadriga-lib', 'acoustic'))

# ---- Thomasson's field impedance of an a x b patch in a rigid baffle, trace-matched velocity, normalized to rho0 c0 ----
# zeta_F = (j k / (2 pi S)) * integral of R(u,v) e^{-jkr}/r e^{j k s (u cos phi + v sin phi)} du dv, R the self-overlap;
# an infinite patch gives 1/cos(theta).
_GL = {}
def _gl(n):
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]
def zeta_F(f, a, b, theta, phi, nal=160):
    k = 2 * np.pi * f / C0
    s = np.sin(theta)
    ac = np.arctan2(b, a)                         # the corner direction: rmax switches there
    out = 0j
    for lo, hi, lim in ((0.0, ac, 'a'), (ac, np.pi / 2, 'b')):
        xa, wa = _gl(nal)
        al = 0.5 * (hi - lo) * xa + 0.5 * (hi + lo); wa = 0.5 * (hi - lo) * wa
        rmax = a / np.cos(al) if lim == 'a' else b / np.sin(al)
        nr = int(max(48, 2.5 * k * rmax.max() * (1 + s)))
        nr = int(2 ** np.ceil(np.log2(nr)))
        xr, wr = _gl(nr)
        r = 0.5 * rmax[:, None] * (xr[None, :] + 1); w = 0.5 * rmax[:, None] * wr[None, :]
        A = al[:, None]
        g = (a - r * np.cos(A)) * (b - r * np.sin(A)) * np.exp(-1j * k * r) \
            * np.cos(k * s * r * np.cos(A) * np.cos(phi)) * np.cos(k * s * r * np.sin(A) * np.sin(phi))
        out += np.sum(wa * np.sum(w * g, axis=1))
    return 4j * k / (2 * np.pi * a * b) * out
def zf_table(a, b):
    fn = os.path.join(CACHE, 'zf_%gx%g.npz' % (a, b))
    if os.path.exists(fn):
        return np.load(fn)['Z']
    Z = np.zeros((len(THIRDS), len(TH), len(PHI)), complex)
    for i, f in enumerate(THIRDS):
        for j, th in enumerate(TH):
            for q, ph in enumerate(PHI):
                Z[i, j, q] = zeta_F(f, a, b, th, ph)
        print('field impedance %g x %g m: %.0f Hz' % (a, b, f), file=sys.stderr)
    os.makedirs(CACHE, exist_ok=True)
    np.savez(fn, Z=Z)
    return Z

# ---- the layer ----
def miki(f, sig):                                                 # [Miki_1990_ModificationsDelanyBazley]
    X = f / sig
    zc = 1 + 0.0699 * X ** -0.632 - 1j * 0.107 * X ** -0.632
    kc = (2 * np.pi * f / C0) * (1 + 0.109 * X ** -0.618 - 1j * 0.160 * X ** -0.618)
    return zc, kc
def zeta_A(f, sig, d, th, mode):                                  # normalized surface impedance, rigid backing
    zc, kc = miki(f, sig)
    if mode == 'LR':
        return -1j * zc / np.tan(kc * d) + 0 * th
    k0s = 2 * np.pi * f / C0 * np.sin(th)
    kz = np.sqrt(kc * kc - k0s * k0s + 0j); kz = np.where(kz.imag > 0, -kz, kz)
    return -1j * zc * (kc / kz) / np.tan(kz * d)
def paris_inf(f, sig, d, mode):
    zA = zeta_A(f, sig, d, TH, mode)
    a = 4 * zA.real * np.cos(TH) / np.abs(zA * np.cos(TH) + 1) ** 2
    return float(np.sum(TW * a * np.sin(2 * TH)))
def sabine_finite(i, f, sig, d, mode, Z):                         # Thomasson Eq. 26, averaged over phi
    zA = zeta_A(f, sig, d, TH, mode)[:, None]
    g = 8 * zA.real * np.sin(TH)[:, None] / np.abs(Z[i] + zA) ** 2
    return float(np.sum(TW * g.mean(axis=1)))
def octv(vals):
    return [sum(vals[3 * k:3 * k + 3]) / 3 for k in range(len(OCT))]
def fit(meas, d, mode, Z, finite=True):
    """sigma (Pa s/m2, 1e3..3e5 on a log grid) minimizing the squared octave residual against the chamber values"""
    best = None
    for ls in np.linspace(np.log(1e3), np.log(3e5), 241):
        sig = math.exp(ls)
        mod = octv([sabine_finite(i, f, sig, d, mode, Z) if finite else paris_inf(f, sig, d, mode) for i, f in enumerate(THIRDS)])
        c = sum((m - v) ** 2 for m, v in zip(mod, meas) if v is not None)
        if best is None or c < best[0]:
            best = (c, sig, mod)
    return best
def convert(alpha_s, thk, a, b, mode='ER'):
    """chamber octaves 125..4k -> (fitted sigma, random-incidence octaves, the fitted finite-sample octaves)"""
    c, sig, mod = fit(alpha_s, thk, mode, zf_table(a, b), True)
    return sig, octv([paris_inf(f, sig, thk, mode) for f in THIRDS]), mod
def random_at(sig, thk, mode='ER'):
    return octv([paris_inf(f, sig, thk, mode) for f in THIRDS])

# ---- the edge correction beta*E [Kuhl_1960_EdgeEffect; review 7.2]: mineral wool only ----
CORR_ISO = [0.042, 0.187, 0.22, 0.14, 0.07, 0.04]                 # beta*E at E = 1.17 m^-1 (a 3 x 4 m sample), octaves
def edge_corrected(alpha_s, E=14.0 / 12.0):
    return [min(1.0, x - c * E / (14.0 / 12.0)) for x, c in zip(alpha_s, CORR_ISO)]

# ---- the validation of md/acoustic.md 6 (A13, A14), 2026-10-15 ----
CASES = [  # name, thickness, sample, chamber 125..4k, independent random incidence (None: none), measured sigma
    ('Dupont glass wool 50 mm', 0.05, (3, 4), [0.153, 0.587, 0.993, 1.073, 1.017, 0.987], [0.130, 0.407, 0.720, 0.887, 0.937, 0.940], 13112),
    ('Dupont PU foam 50 mm', 0.05, (3, 4), [0.107, 0.437, 0.817, 0.980, 0.963, 0.973], [0.177, 0.317, 0.620, 0.883, 0.880, 0.913], 10605),
    ('Kosten rock wool 50 mm, 12 m2', 0.05, (3, 4), [0.20, 0.60, 0.95, 1.01, 1.03, 1.01], [None, 0.40, 0.69, 0.90, 0.95, 0.95], None),
    ('Kosten rock wool 50 mm, 8 m2', 0.05, (2, 4), [0.20, 0.68, 1.02, 1.06, 1.03, 1.04], [None, 0.40, 0.69, 0.90, 0.95, 0.95], None),
    ('Kosten rock wool 50 mm, 4 m2', 0.05, (2, 2), [0.20, 0.70, 1.15, 1.13, 1.08, 1.11], [None, 0.40, 0.69, 0.90, 0.95, 0.95], None),
    ('Scrosati polyester 17.8 mm', 0.0178, (3, 3.6), [0.064, 0.166, 0.419, 0.680, 0.824, 0.902], None, 15000),
    ('Scrosati stone wool 105 mm, framed', 0.105, (3, 3.6), [0.711, 0.983, 1.067, 1.022, 0.967, 0.938], None, 30300),
    ('Knauf TP 120 A 20 mm (mounting not stated)', 0.02, (3, 4), [0.11, 0.26, 0.52, 0.70, 0.84, 0.95], None, 11000),
]
def _row(lab, v, ref=None):
    s = '  %-32s' % lab + ' '.join('%6.3f' % x for x in v)
    if ref:
        r = [x - y for x, y in zip(v, ref) if y is not None]
        s += '   max|d| %.3f  rms %.3f' % (max(abs(x) for x in r), math.sqrt(sum(x * x for x in r) / len(r)))
    print(s)
def check():
    for name, d, (a, b), ch, ref, sig_m in CASES:
        E = 2 * (a + b) / (a * b)
        print('%s, %g x %g m, measured sigma %s' % (name, a, b, sig_m))
        print('  %-32s' % 'octave' + ' '.join('%6d' % f for f in OCT))
        _row('chamber (Sabine)', ch)
        if ref:
            _row('independent random incidence', [x if x is not None else float('nan') for x in ref])
        sig, rnd, mod = convert(ch, d, a, b)
        _row('converted, sigma %6.0f' % sig, rnd, ref)
        _row('  its finite-sample fit', mod, ch)
        _row('edge correction beta*E', edge_corrected(ch, E), ref)
        if sig_m:
            _row('Miki at the measured sigma', random_at(sig_m, d), ref)
        print()
if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest='cmd', required=True)
    c = sp.add_parser('convert', help='chamber octaves 125 Hz..4 kHz to random incidence')
    c.add_argument('--thk', type=float, required=True, help='layer thickness, m')
    c.add_argument('--sample', type=float, nargs=2, default=(3.0, 4.0), help='sample size a b, m (default 3 4)')
    c.add_argument('--alpha', type=float, nargs=6, required=True, help='chamber values at 125 250 500 1k 2k 4k Hz')
    c.add_argument('--local', action='store_true', help='local reaction instead of extended reaction')
    c.add_argument('--sigma', type=float, help='a measured flow resistivity, Pa s/m2, to compare with the fitted one')
    sp.add_parser('check', help='reproduce the validation of md/acoustic.md 6 (A13, A14)')
    args = ap.parse_args()
    if args.cmd == 'check':
        check()
    else:
        sig, rnd, mod = convert(args.alpha, args.thk, *args.sample, mode='LR' if args.local else 'ER')
        print('fitted sigma %.0f Pa s/m2' % sig + ('' if args.sigma is None else ' (measured %.0f, ratio %.2f%s)' % (
            args.sigma, sig / args.sigma, '' if 1 / 1.5 <= sig / args.sigma <= 1.5 else ': outside 1.5, do not use')))
        _row('octave', OCT)
        _row('chamber (Sabine)', args.alpha)
        _row('random incidence', rnd)
        _row('finite-sample fit', mod, args.alpha)
