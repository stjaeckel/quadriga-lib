# SPDX-License-Identifier: Apache-2.0
# Part of quadriga-lib — see LICENSE for terms.
#
# The transmission fit of md/acoustic.md 9 on the formula copy txlib.py, vectorized: the knobs of a mass-law material
# fitted per construction to a transmission reference, minimax over octaves (md/m10_m15_report.md 0.2, 1.0).
#   Knobs p = [X1k, m, att, attB, cF, cQ, rF, rQ, rS, C]: X1k the level (X0 at 1 kHz, i.e. fRef with thk), m the mass
#   column (a switch: fitted only with V['mfit']), att (dB at 1 kHz) / attB, cF / cQ the bending stiffness, rF / rQ / rS
#   the resonance (+ the mass-air-mass dip, - the first mode), C the column of candidate M13.
#   Rule R1h; tf fixed at the class's absorption law; references, absorption laws and physical constants from
#   tl_refs.json (the 13 constructions of report 3.6). Bounds of report 3.6 on cF, cQ, rF, rQ.
#   Variant V (dict): amax (upper angle of the average; 90 = random incidence, the library's own average and the fit
#   convention of md/acoustic.md 9; 78 = field incidence), mfit, noatt, m13 ('col' | 'davy'), m14 (deg), m15, pcf (coiF
#   pinned at the physical coincidence), attmax (the att bound, 12 dB by default).
# Procedure: differential evolution on 96 x 3, Nelder-Mead on 192 x 6, then every fitted vector of every seed evaluated at
# 768 x 12 and the best kept (a polish or a selection on a coarser grid picks knobs that alias it).
#
# usage: python3 tools/acoustic/txfit.py [--variant JSON] [--seeds N] [--procs P] [--out FILE] [construction ...]
#   e.g. --variant '{"amax": 90}' fits the library as built at random incidence (md/m10_m15_report.md 1.0: 2.97 dB)
import argparse, json, math, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from txlib import FMAP, RHO0C

_R = json.load(open(os.path.join(HERE, 'tl_refs.json')))
OCT = _R['octaves_hz']
REF = [(c['name'], c['cls'], c['thk'], [tuple(t) if t else None for t in c['targets']]) for c in _R['constructions']]
ABS = {k: tuple(v) for k, v in _R['absorption_law'].items()}
NAMES = [r[0] for r in REF]
PHYS_FC = {n: v['fc_hz'] for n, v in _R['physical'].items() if v['fc_hz']}


class Grid:
    def __init__(self, na, ns, amax=78.0):
        self.na, self.ns, self.amax = na, ns, amax
        self.F = np.array([fc * 2 ** (k / 3 + ((j + 0.5) / ns - 0.5) / 3) for fc in OCT for k in (-1, 0, 1) for j in range(ns)])
        th = np.radians((np.arange(na) + 0.5) * amax / na)
        self.C = np.cos(th)
        w = self.C * np.sin(th)
        self.W = w / w.sum()


_G = {}


def grid(na, ns, amax=78.0):
    k = (na, ns, amax)
    if k not in _G:
        _G[k] = Grid(na, ns, amax)
    return _G[k]


def TL_oct(p, tfl, V, g):
    """octave TL (dB) of the plate rule over the average of g"""
    X1k, m, att, attB, cF, cQ, rF, rQ, rS, C = p
    if not V.get('mfit'):
        m = 20.0
    if V.get('noatt'):
        att = 0.0
    f = g.F[:, None]
    c = g.C[None, :]
    if V.get('m14') is not None:
        c = np.maximum(c, math.cos(math.radians(V['m14'])))
    Xm = X1k * (f / 1000.0) ** (m / 20.0) * np.ones_like(c)
    if rF > 0 and rQ > 0 and rS != 0:
        if rS > 0:
            x = f * c / rF
            Xm = Xm * np.minimum(1.0, np.abs((1 - x * x) + 1j * x / rQ))
        else:
            y = rF / f
            Xm = Xm * np.abs((1 - y * y) - 1j * y / rQ)
    cross = 0.0
    if cF > 0 and cQ > 0:
        eta = 1.0 / cQ + 0 * f
        if V.get('m13') == 'col':
            eta = eta + C * (f / cF) ** -0.5
        elif V.get('m13') == 'davy':
            eta = eta + X1k * RHO0C / (1000.0 * math.pi) / (485.0 * np.sqrt(f))   # m'' from the level (m = 20)
        s2 = 1 - c * c
        w = (f / cF) ** 2 * s2 * s2
        X = Xm * np.abs((1 - w) - 1j * w * eta)
        if V.get('m15'):
            cross = 2.0 * c * Xm * w * eta
    else:
        X = Xm
    Xc = np.maximum(X, 1.0)
    A = 20 * np.log10(Xc)
    with np.errstate(divide='ignore', invalid='ignore'):
        TL = np.where(X > 1.0, A * np.log10(1 + (Xc * c) ** 2 + cross) / np.log10(1 + Xc * Xc), 0.0)
    tau = 10 ** (-0.1 * TL)
    a = np.clip(tfl[0] * (g.F / 1000.0) ** tfl[1], 0.0, 1.0)[:, None]
    ig = (10 ** (-0.1 * np.maximum(att * (g.F / 1000.0) ** attB, 0.0)))[:, None]
    T = a * tau / (a + tau - a * tau)
    return -10 * np.log10((T * ig * g.W[None, :]).sum(axis=1).reshape(9, 3 * g.ns).mean(axis=1))


def resid(v, t):
    if t is None:
        return 0.0
    lo, hi = t
    return v - hi if v > hi else (v - lo if v < lo else 0.0)


BOUNDS = dict(cF=(30.0, 20000.0), cQ=(1.0, 300.0), rF=(10.0, 1000.0), rQ=(0.3, 50.0), C=(0.0, 10.0))


def valid(p, V):
    X1k, m, att, attB, cF, cQ, rF, rQ, rS, C = p
    if not (0 < X1k < 10 ** 4.5 and m > 0 and att >= 0 and -3 <= attB <= 3):
        return False
    if not (BOUNDS['cF'][0] <= cF <= BOUNDS['cF'][1] and BOUNDS['cQ'][0] <= cQ <= BOUNDS['cQ'][1]):
        return False
    if not (BOUNDS['rF'][0] <= rF <= BOUNDS['rF'][1] and BOUNDS['rQ'][0] <= rQ <= BOUNDS['rQ'][1]):
        return False
    return not (V.get('m13') == 'col' and not (BOUNDS['C'][0] <= C <= BOUNDS['C'][1]))


def cost_of(p, tfl, tg, V, g):
    """worst octave residual + 0.01 rms (dB); 1e9 outside the bounds"""
    if not valid(p, V):
        return 1e9
    with np.errstate(divide='ignore', invalid='ignore'):
        v = TL_oct(p, tfl, V, g)
    if not np.all(np.isfinite(v)):
        return 1e9
    r = [resid(x, t) for x, t in zip(v, tg)]
    return max(abs(x) for x in r) + 0.01 * math.sqrt(sum(x * x for x in r))


def nm(fun, x0, step, it=800, tol=1e-9):
    n = len(x0); S = [list(x0)] + [[x0[j] + (step[j] if j == i else 0) for j in range(n)] for i in range(n)]
    F = [fun(s) for s in S]
    for _ in range(it):
        o = sorted(range(n + 1), key=lambda i: F[i]); S = [S[i] for i in o]; F = [F[i] for i in o]
        if abs(F[-1] - F[0]) < tol and F[0] < 1e8: break
        c = [sum(S[i][j] for i in range(n)) / n for j in range(n)]
        xr = [c[j] + (c[j] - S[-1][j]) for j in range(n)]; fr = fun(xr)
        if fr < F[0]:
            xe = [c[j] + 2 * (c[j] - S[-1][j]) for j in range(n)]; fe = fun(xe)
            S[-1], F[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < F[-2]: S[-1], F[-1] = xr, fr
        else:
            xc = [c[j] + 0.5 * (S[-1][j] - c[j]) for j in range(n)]; fc = fun(xc)
            if fc < F[-1]: S[-1], F[-1] = xc, fc
            else:
                S = [S[0]] + [[S[0][j] + 0.5 * (s[j] - S[0][j]) for j in range(n)] for s in S[1:]]; F = [F[0]] + [fun(s) for s in S[1:]]
    i = min(range(n + 1), key=lambda i: F[i]); return S[i], F[i]


def free_of(V):
    fr = [0, 2, 3, 4, 5, 6, 7]
    if V.get('mfit'):
        fr.insert(1, 1)
    if V.get('noatt'):
        fr.remove(2)
        fr.remove(3)
    if V.get('m13') == 'col':
        fr.append(9)
    return fr


# knob ranges of the global search (log for X1k, cF, cQ, rF, rQ)
RANGE = {0: (math.log(10 ** 0.3), math.log(10 ** 4.5)), 1: (0.5, 60.0), 2: (0.0, 12.0), 3: (-3.0, 3.0),
         4: (math.log(30.0), math.log(20000.0)), 5: (0.0, math.log(300.0)), 6: (math.log(10.0), math.log(1000.0)),
         7: (math.log(0.3), math.log(50.0)), 9: (0.0, 10.0)}
LOGK = {0, 4, 5, 6, 7}


def rng_of(V):
    R = dict(RANGE)
    R[2] = (0.0, V.get('attmax', 12.0))
    return R


def de(fun, lo, hi, pop=48, gens=350, rnd=None, inits=()):
    """self-adaptive differential evolution (jDE, rand/1/bin with a best/1 share), box-bounded; returns (x, f) of the best"""
    n = len(lo); lo = np.array(lo); hi = np.array(hi)
    P = lo + (hi - lo) * np.array([[rnd.random() for _ in range(n)] for _ in range(pop)])
    for i, x in enumerate(inits[:pop // 4]): P[i] = np.clip(np.array(x), lo, hi)
    Fv = np.array([fun(list(x)) for x in P]); Fs = np.full(pop, 0.6); CRs = np.full(pop, 0.9)
    for g in range(gens):
        ib = int(np.argmin(Fv))
        for i in range(pop):
            F_ = rnd.uniform(0.2, 1.0) if rnd.random() < 0.1 else Fs[i]; CR = rnd.random() if rnd.random() < 0.1 else CRs[i]
            a, b, c = rnd.sample([j for j in range(pop) if j != i], 3)
            base = P[ib] if rnd.random() < 0.3 else P[a]
            v = base + F_ * (P[b] - P[c])
            jr = rnd.randrange(n)
            u = np.array([v[j] if (rnd.random() < CR or j == jr) else P[i][j] for j in range(n)])
            u = np.where(u < lo, lo + (P[i] - lo) * rnd.random(), u); u = np.where(u > hi, hi - (hi - P[i]) * rnd.random(), u)
            fu = fun(list(u))
            if fu <= Fv[i]: P[i], Fv[i], Fs[i], CRs[i] = u, fu, F_, CR
    ib = int(np.argmin(Fv)); order = np.argsort(Fv)
    return [list(P[j]) for j in order[:6]], [float(Fv[j]) for j in order[:6]]

def fit_de(name, V, seed=0, x0s=(), pop=48, gens=350, mid=(96, 3), fine=(192, 6)):
    """global: jDE on the mid grid for each resonance sign, then Nelder-Mead polish of the six best of each on the fine grid"""
    k = NAMES.index(name); _, acl, thk, tg = REF[k]; tfl = ABS[acl]
    amax = V.get('amax', 78.0)
    gm, gf = grid(mid[0], mid[1], amax), grid(fine[0], fine[1], amax)
    fr = free_of(V)
    base0 = [1.0, 20.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    if V.get('pcf') and name in PHYS_FC: fr = [j for j in fr if j != 4]; base0[4] = PHYS_FC[name]
    R = rng_of(V)
    lo = [R[j][0] for j in fr]; hi = [R[j][1] for j in fr]
    def unpack(q, rS):
        p = list(base0); p[8] = rS
        for i, j in enumerate(fr): p[j] = math.exp(q[i]) if j in LOGK else q[i]
        return p
    def pack(p): return [math.log(p[j]) if j in LOGK else p[j] for j in fr]
    rnd = random.Random(7919 * seed + 31 * k + 1)
    best = None
    for rS in (1.0, -1.0):
        inits = []
        for x in x0s:
            if (x[8] > 0) != (rS > 0): continue
            x = list(x)
            if V.get('pcf') and name in PHYS_FC: x[4] = PHYS_FC[name]
            if not V.get('mfit'): x[1] = 20.0
            if V.get('noatt'): x[2] = 0.0; x[3] = 0.0
            if V.get('m13') == 'col' and not (x[9] > 0): x[9] = 0.05
            inits.append([min(max(v, R[j][0]), R[j][1]) for v, j in zip(pack(x), fr)])
        fun = lambda q: cost_of(unpack(q, rS), tfl, tg, V, gm)
        Q, Fq = de(fun, lo, hi, pop=pop, gens=gens, rnd=rnd, inits=inits)
        funf = lambda z: cost_of(unpack(z, rS), tfl, tg, V, gf)
        for q in Q[:3]:
            q, fq = nm(funf, q, [0.03 * (h - l) for l, h in zip(lo, hi)], it=400)
            if best is None or fq < best[0]: best = (fq, unpack(q, rS), q, rS)
    fq, p, q, rS = best
    funf = lambda z: cost_of(unpack(z, rS), tfl, tg, V, gf)
    for _ in range(2):
        q, fq = nm(funf, q, [0.01 * (h - l) for l, h in zip(lo, hi)], it=400)
    return unpack(q, rS)



def evaluate(name, p, V, na=768, ns=12):
    """octave TL, residuals and worst residual at the evaluation grid"""
    k = NAMES.index(name)
    _, acl, thk, tg = REF[k]
    v = TL_oct(p, ABS[acl], V, grid(na, ns, V.get('amax', 78.0)))
    r = [resid(x, t) for x, t in zip(v, tg)]
    return v, r, max(abs(x) for x, t in zip(r, tg) if t is not None)


def project(p, V, name=None):
    """a knob vector onto variant V: its fixed knobs set, the rest clipped to the search ranges"""
    p = list(p)
    if V.get('pcf') and name in PHYS_FC:
        p[4] = PHYS_FC[name]
    if not V.get('mfit'):
        p[1] = 20.0
    if V.get('noatt'):
        p[2] = p[3] = 0.0
    if V.get('m13') != 'col':
        p[9] = 0.0
    for j, (lo, hi) in rng_of(V).items():
        if j == 1 and not V.get('mfit'):
            continue
        v = math.log(p[j]) if j in LOGK else p[j]
        v = min(max(v, lo), hi)
        p[j] = math.exp(v) if j in LOGK else v
    return p


def select(name, V, vecs):
    """the best of the vectors (projected onto V) at the evaluation grid"""
    k = NAMES.index(name)
    _, acl, thk, tg = REF[k]
    ge = grid(768, 12, V.get('amax', 78.0))
    return min(((cost_of(q, ABS[acl], tg, V, ge), q) for q in (project(p, V, name) for p in vecs)), key=lambda x: x[0])[1]


def columns(name, p, V={}):
    """library columns (GHz) of a fit"""
    k = NAMES.index(name)
    _, acl, thk, tg = REF[k]
    tfl = ABS[acl]
    X1k, m, att, attB, cF, cQ, rF, rQ, rS, C = p
    m = m if V.get('mfit') else 20.0
    if V.get('noatt'):
        att = 0.0
    F1K = 1000 * FMAP
    fRef = F1K * thk / X1k ** (20.0 / m)
    return dict(thk=thk, tf=tfl[0] * (fRef / F1K) ** tfl[1], tfB=tfl[1], m=m, fRef=fRef, resF=rF * FMAP, resQ=rQ, resS=rS,
                att=att * (fRef / F1K) ** attB, attB=attB, coiF=cF * FMAP, coiQ=cQ)


def _task(a):
    name, V, seed = a
    return name, fit_de(name, V, seed=seed)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='transmission fit on the formula copy (md/acoustic.md 9)')
    ap.add_argument('names', nargs='*', help='constructions (default: all 13)')
    ap.add_argument('--variant', default='{"amax": 90}', help='variant dict as JSON (default: the library at random incidence)')
    ap.add_argument('--seeds', type=int, default=6)
    ap.add_argument('--procs', type=int, default=os.cpu_count() or 1)
    ap.add_argument('--out', default='', help='write the fits as JSON')
    A = ap.parse_args()
    V = json.loads(A.variant)
    names = A.names or NAMES
    from multiprocessing import Pool
    vecs = {n: [] for n in names}
    with Pool(A.procs) as P:
        for n, p in P.imap_unordered(_task, [(n, V, s) for n in names for s in range(A.seeds)]):
            vecs[n].append(p)
    out = {}
    print('%-30s worst | residual per octave %s' % ('construction', ' '.join('%g' % f for f in OCT)))
    for n in names:
        p = select(n, V, vecs[n])
        v, r, w = evaluate(n, p, V)
        out[n] = dict(p=p, oct=list(map(float, v)), resid=r, worst=w, columns=columns(n, p, V))
        print('%-30s %5.2f | %s' % (n, w, ' '.join('%+.2f' % x for x in r)), flush=True)
    print('worst %.2f' % max(d['worst'] for d in out.values()))
    if A.out:
        json.dump(dict(V=V, fits=out), open(A.out, 'w'), indent=1)
