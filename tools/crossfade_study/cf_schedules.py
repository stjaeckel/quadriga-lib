"""Cross-fade schedules between two channel segments, and the metrics along the track (numpy only).

A schedule is a power weight per ray of the old segment (A) and of the new segment (B) at every snapshot, plus the scale of
the LOS path. schedule(o, name) knows these names (rho in percent, X and cap in dB, beta in percent):
  A_lib            present "taps" rule of src/crossfade_segments.cpp (port)
  B_ng             channel merger of QuaDRiGa-NG (port)
  C_union          all old paths with 1 - w, all new ones with w (present "rays" rule)
  D<rho>_<order>   staggered exchange at own levels; orders: rand, ng, delay, power, pw2, optds, optall, optda, optmix, optuni,
                   optuds; suffixes _sin, _tap, _coef
  E25_free, E25_norm   independent birth and death points
  F<rho>g_<order>, F<rho>l_<order>   level and structure separated, level dB-linear (g) or power-linear (l)
  Gc<cap>, Gb<beta>, Hx<X>, Hold0    blends of F and D at the dB-linear level (rho = 0.25, order optuni)
  Gx<X>, Gz<X>, Gy<X>, Gm<X>         exchange where the level is within X dB of the weaker segment; Gm10 is the chosen rule
"""
import numpy as np

# ---------------------------------------------------------------- units and moments
def units(seg, kind):
    """Unit index of each NLOS path: kind 'tap' (output taps) or 'cluster' (slot = tap // 3). Returns (idx[P], n_unit); LOS: -1"""
    key = seg['tap'] if kind == 'tap' else seg['tap'] // 3
    nl = seg['ptype'] != 0
    u = np.full(key.size, -1)
    vals, inv = np.unique(key[nl], return_inverse=True)
    u[nl] = inv
    return u, vals.size

def moments(seg, sl, w=None):
    """Additive moments per path, mean over the snapshots sl: power, p*tau, p*tau^2, p*exp(j angle) for the 4 angles."""
    pg = seg['pg'][:, sl]
    m0 = pg.mean(1)
    m1 = (pg * seg['delay'][:, sl]).mean(1)
    m2 = (pg * seg['delay'][:, sl] ** 2).mean(1)
    z = (pg[:, None, :] * np.exp(1j * seg['ang'][:, :, sl])).mean(2)          # [P, 4]
    return m0, m1, m2, z

def unit_moments(seg, sl, kind):
    u, n = units(seg, kind)
    m0, m1, m2, z = moments(seg, sl)
    nl = u >= 0
    U = lambda x: np.bincount(u[nl], weights=x[nl], minlength=n)
    Z = np.stack([U(z[:, k].real) + 1j * U(z[:, k].imag) for k in range(4)], 1)
    los = ~nl
    det = (m0[los].sum(), m1[los].sum(), m2[los].sum(), z[los].sum(0))
    return dict(n=n, u=u, m0=U(m0), m1=U(m1), m2=U(m2), z=Z, det=det)

def tap_moments_coeff(seg, sl):
    """Moments of the output taps as the library and NG see them: power of the tap coefficients (rays summed), mean over the
    snapshots and the element pairs; one delay per tap."""
    u, n = units(seg, 'tap')
    st = np.flatnonzero(np.r_[True, np.diff(seg['tap']) != 0])
    t = np.add.reduceat(seg['coeff'][:, :, :, sl], st, axis=2)                 # [R, T, n_tap(+LOS), n]
    p = (np.abs(t) ** 2).mean((0, 1, 3)).astype(np.float64)
    pg = np.add.reduceat(seg['pg'][:, sl], st, axis=0); pd = np.add.reduceat(seg['pg'][:, sl] * seg['delay'][:, sl], st, axis=0)
    dfirst = seg['delay'][st][:, sl]
    dl = np.where(pg > 0, pd / np.maximum(pg, 1e-300), dfirst).mean(1)
    is_los = seg['ptype'][st] == 0
    m0 = p[~is_los]; dn = dl[~is_los]
    um = unit_moments(seg, sl, 'tap')
    det = (p[is_los].sum(), (p[is_los] * dl[is_los]).sum(), (p[is_los] * dl[is_los] ** 2).sum(), um['det'][3])
    return dict(n=n, u=u, m0=m0, m1=m0 * dn, m2=m0 * dn * dn, z=um['z'], det=det)

def spreads(m0, m1, m2, z):
    """DS and the four circular angular spreads (TR 38.901 Annex A.1) from summed moments; inputs [..., ] and z [..., 4]"""
    s = np.maximum(m0, 1e-300)
    ds = np.sqrt(np.maximum(m2 / s - (m1 / s) ** 2, 0.0))
    r = np.clip(np.abs(z) / s[..., None], 1e-300, 1.0)
    asp = np.sqrt(-2.0 * np.log(r))
    return ds, asp

# ---------------------------------------------------------------- ramps and weight schedules
def ramp(n_r):
    x = (np.arange(n_r) + 1.0) / (n_r + 1.0)
    return 0.5 * (1.0 + np.sin((x - 0.5) * np.pi)), x

def w_seq(p, order, f, fade_out):
    """Library rule (qd_xf_weights): the units fade one after another, each over the part of the ramp that matches its power share."""
    n, P = p.size, p.sum()
    W = np.zeros((n, f.size))
    before = np.zeros(n)
    acc = 0.0
    rng = range(n - 1, -1, -1) if fade_out else range(n)
    for m in rng:
        before[m] = acc; acc += p[order[m]]
    for m in range(n):
        q = p[order[m]]; x = f * P - before[m]
        W[order[m]] = np.clip(x / q, 0, 1) if q > 0 else (x > 0).astype(float)
    return W

def w_vr(p, order, w, rho, fade_out, grid=1025, shape='lin'):
    """Visibility-region family: unit m of the fade order has the window [a, b] = [(1-rho) F(m-1), (1-rho) F(m) + rho] of a
    parameter v in [0, 1] (F: cumulative power share in fade order), its faded part g rises from 0 to 1 over the window; v(x)
    is chosen so that the summed power follows the ramp w exactly. rho = 0: the library rule; rho = 1: all at once (union)."""
    n, P = p.size, p.sum()
    if n == 0 or P <= 0:
        return np.tile((1 - w) if fade_out else w, (n, 1))
    sh = p[order] / P
    F1 = np.cumsum(sh); F0 = F1 - sh
    a = (1 - rho) * F0; b = (1 - rho) * F1 + rho
    v = np.unique(np.concatenate([[0.0, 1.0], a, b])) if shape == 'lin' else np.unique(np.concatenate([np.linspace(0, 1, grid), a, b]))
    def G(vv):
        g = np.clip((vv[None, :] - a[:, None]) / np.maximum(b - a, 1e-12)[:, None], 0, 1)
        if shape == 'sin':
            g = 0.5 * (1 - np.cos(np.pi * g))
        return g
    S = (sh[:, None] * G(v)).sum(0)                       # faded power share, 0 -> 1, monotone
    S[0], S[-1] = 0.0, 1.0
    vx = np.interp(w, S, v)
    g = G(vx)
    W = np.empty((n, w.size))
    W[order] = (1 - g) if fade_out else g
    return W

def w_ng(na, nb, oa, ob, n_r):
    """NG literal (merge_coeff.m): n_sub = min(na, nb, n_r) sub-intervals of floor(n_r / n_sub) snapshots, pair l: the old
    unit oa[l] ramps down while the new unit ob[l] ramps up (sine power ramp of the sub-interval); units without partner
    use the ramp of the whole interval."""
    Wa = np.ones((na, n_r)); Wb = np.zeros((nb, n_r))
    n_sub = min(na, nb, n_r)
    w_full, _ = ramp(n_r)
    if n_sub > 0:
        L = n_r // n_sub
        r, _ = ramp(L)
        for l in range(n_sub):
            i2 = slice(l * L, (l + 1) * L); i3 = slice((l + 1) * L, n_r)
            Wa[oa[l], i2] = 1 - r; Wa[oa[l], i3] = 0
            Wb[ob[l], i2] = r; Wb[ob[l], i3] = 1
    for l in range(n_sub, na):
        Wa[oa[l]] = 1 - w_full
    for l in range(n_sub, nb):
        Wb[ob[l]] = w_full
    return Wa, Wb

# ---------------------------------------------------------------- fade orders
def order_ng(ua, ub, n_r):
    """Port of qd_xf_order / NG merging_cost_fcn: DS of n_sub sub-intervals close to a linear change, plus the power penalty."""
    na, nb = ua['n'], ub['n']
    n_sub = min(na, nb, n_r)
    oa, ob = list(range(na)), list(range(nb))
    if n_sub == 0:
        return np.array(oa, int), np.array(ob, int)
    da = ua['m1'] / np.maximum(ua['m0'], 1e-300); db = ub['m1'] / np.maximum(ub['m0'], 1e-300)
    pa, pb = ua['m0'], ub['m0']
    pad, dad = ua['det'][0], (ua['det'][1] / ua['det'][0] if ua['det'][0] > 0 else 0.0)
    pbd, dbd = ub['det'][0], (ub['det'][1] / ub['det'][0] if ub['det'][0] > 0 else 0.0)
    def sp(p, dl):
        s = p.sum(); m = (p * dl).sum() / s
        return np.sqrt(max((p * dl * dl).sum() / s - m * m, 0.0))
    ds1 = sp(np.r_[pad, pa], np.r_[dad, da]); ds2 = sp(np.r_[pbd, pb], np.r_[dbd, db])
    P1, P2 = pad + pa.sum(), pbd + pb.sum()
    x = (2.0 * np.arange(n_sub) + 1.0) / (2.0 * n_sub)
    ro = 0.5 * (1 + np.sin((x - 0.5) * np.pi)); ds_t = ds1 + (ds2 - ds1) * x
    tri = np.triu(np.ones((n_sub, n_sub))) - 0.5 * np.eye(n_sub)          # [i, j]: j > i -> 1, j == i -> 0.5
    def cost(A, B):
        WA = np.empty((n_sub, na)); WA[:, :n_sub] = tri; WA[:, n_sub:] = (1 - ro)[:, None]
        WB = np.empty((n_sub, nb)); WB[:, :n_sub] = tri.T; WB[:, n_sub:] = ro[:, None]
        p = np.concatenate([(1 - ro)[:, None] * pad, ro[:, None] * pbd, WA * pa[A][None, :], WB * pb[B][None, :]], 1)
        dl = np.concatenate([[dad], [dbd], da[A], db[B]])
        s = p.sum(1); m = (p * dl).sum(1) / s
        ds = np.sqrt(np.maximum((p * dl * dl).sum(1) / s - m * m, 0.0))
        num = ((ds_t - ds) ** 2).sum(); den = (ds_t ** 2).sum()
        tgt = P1 - (P1 - P2) * ro
        pw = np.abs(10 * np.log10(tgt / s)).max()
        return (10 * np.log10(num / den) if den > 0 and num > 0 else (-300.0 if den > 0 else 0.0)) + pw
    def improve(O, first):
        n = len(O)
        for c in range(n):
            rest = [x for x in O if x != c]
            best, at = np.inf, 0
            for pos in range(n):
                T = rest[:pos] + [c] + rest[pos:]
                v = cost(np.array(T), np.array(ob)) if first else cost(np.array(oa), np.array(T))
                if v < best:
                    best, at = v, pos
            O[:] = rest[:at] + [c] + rest[at:]
    for _ in range(2):
        improve(oa, True); improve(ob, False)
    return np.array(oa, int), np.array(ob, int)

def order_opt(ua, ub, n_r, rho, metrics=(0, 1, 2, 3, 4), n_eval=16, init=None, passes=2, shape='lin', target='lin', ratio=None):
    """Generalised NG idea: fade order of the old and new units that keeps the chosen spreads (0 DS, 1 ASD, 2 ESD, 3 ASA,
    4 ESA) close to a linear change between the two segments, evaluated with the actual weights of w_vr at n_eval points."""
    na, nb = ua['n'], ub['n']
    xs = (np.arange(n_eval) + 0.5) / n_eval
    w = 0.5 * (1 + np.sin((xs - 0.5) * np.pi))
    if ratio is not None:                                                  # structure share of the new segment instead of the ramp
        w = ratio(w) if callable(ratio) else w * ratio / ((1 - w) + w * ratio)
    def full(u):
        return spreads(u['m0'].sum() + u['det'][0], u['m1'].sum() + u['det'][1], u['m2'].sum() + u['det'][2], u['z'].sum(0) + u['det'][3])
    dsa, asa = full(ua); dsb, asb = full(ub)
    ta = np.r_[dsa, asa]; tb = np.r_[dsb, asb]
    tgt = ta[None, :] + (tb - ta)[None, :] * xs[:, None]                   # [n_eval, 5]
    if target == 'union':                                                  # spreads of the superposition of both segments (weights 1 - w and w)
        ds_u, as_u = spreads((1 - w) * (ua['m0'].sum() + ua['det'][0]) + w * (ub['m0'].sum() + ub['det'][0]),
                             (1 - w) * (ua['m1'].sum() + ua['det'][1]) + w * (ub['m1'].sum() + ub['det'][1]),
                             (1 - w) * (ua['m2'].sum() + ua['det'][2]) + w * (ub['m2'].sum() + ub['det'][2]),
                             np.outer(1 - w, ua['z'].sum(0) + ua['det'][3]) + np.outer(w, ub['z'].sum(0) + ub['det'][3]))
        tgt = np.concatenate([ds_u[:, None], as_u], 1)
    if target == 'mix':                                                    # spreads of the power-weighted mixture without the term of the mean offset
        PA, PB = ua['m0'].sum() + ua['det'][0], ub['m0'].sum() + ub['det'][0]
        al = (1 - w) * PA / ((1 - w) * PA + w * PB)
        tgt = np.sqrt(al[:, None] * ta[None, :] ** 2 + (1 - al)[:, None] * tb[None, :] ** 2)
    sel = np.array(metrics)
    def cost(A, B):
        Wa = w_vr(ua['m0'], A, w, rho, True, grid=257, shape=shape); Wb = w_vr(ub['m0'], B, w, rho, False, grid=257, shape=shape)
        m0 = Wa.T @ ua['m0'] + Wb.T @ ub['m0'] + (1 - w) * ua['det'][0] + w * ub['det'][0]
        m1 = Wa.T @ ua['m1'] + Wb.T @ ub['m1'] + (1 - w) * ua['det'][1] + w * ub['det'][1]
        m2 = Wa.T @ ua['m2'] + Wb.T @ ub['m2'] + (1 - w) * ua['det'][2] + w * ub['det'][2]
        z = Wa.T @ ua['z'] + Wb.T @ ub['z'] + np.outer(1 - w, ua['det'][3]) + np.outer(w, ub['det'][3])
        ds, asp = spreads(m0, m1, m2, z)
        X = np.concatenate([ds[:, None], asp], 1)
        e = ((X - tgt) ** 2).mean(0) / np.maximum((tgt ** 2).mean(0), 1e-300)
        return e[sel].sum()
    oa = list(range(na)) if init is None else list(init[0]); ob = list(range(nb)) if init is None else list(init[1])
    def improve(O, first):
        n = len(O)
        for c in range(n):
            rest = [x for x in O if x != c]
            best, at = np.inf, 0
            for pos in range(n):
                T = rest[:pos] + [c] + rest[pos:]
                v = cost(np.array(T), np.array(ob)) if first else cost(np.array(oa), np.array(T))
                if v < best - 1e-15:
                    best, at = v, pos
            O[:] = rest[:at] + [c] + rest[at:]
    for _ in range(passes):
        improve(oa, True); improve(ob, False)
    return np.array(oa, int), np.array(ob, int)

def order_key(ua, ub, key, seed=0):
    if key == 'rand':
        rng = np.random.default_rng(seed)
        return rng.permutation(ua['n']), rng.permutation(ub['n'])
    if key == 'delay':
        return np.argsort(ua['m1'] / np.maximum(ua['m0'], 1e-300), kind='stable'), np.argsort(ub['m1'] / np.maximum(ub['m0'], 1e-300), kind='stable')
    if key == 'power':                                                       # weakest first
        return np.argsort(ua['m0'], kind='stable'), np.argsort(ub['m0'], kind='stable')
    if key == 'pw2':                                                         # old: weakest dies first; new: strongest is born first
        return np.argsort(ua['m0'], kind='stable'), np.argsort(-ub['m0'], kind='stable')
    raise ValueError(key)

# ---------------------------------------------------------------- the candidate methods
def schedule(o, name, overlap=0.5, seed=0):
    """Weights of all paths of both segments at all snapshots for the method 'name'.
    Returns dict: wA [PA, N], wB [PB, N] (power weights of the NLOS paths), los: complex scale of the LOS coefficient of A and B."""
    A, B, nA = o['A'], o['B'], o['nA']
    N = B['pg'].shape[1]
    n_r = o.get('n_r', int(np.floor(overlap * nA + 0.5))); r0 = nA - n_r
    sl = slice(r0, nA)
    w, x = ramp(n_r)
    unit = 'tap' if name in ('A_lib', 'B_ng') or name.endswith('_tap') else 'cluster'
    if name.endswith('_coef'):                                                # powers from the tap coefficients (with antennas and fading), as the library has them for "taps"
        name = name[:-5]; coef = True
    else:
        coef = False
    if name in ('A_lib', 'B_ng') or coef:
        ua, ub = tap_moments_coeff(A, sl), tap_moments_coeff(B, sl)
        if coef:
            unit = 'tap'
    else:
        ua, ub = unit_moments(A, sl, unit), unit_moments(B, sl, unit)
    los_mode = 'lib'
    base = name[:-4] if name.endswith('_tap') else name
    if base == 'C_union':
        Wa = np.tile(1 - w, (ua['n'], 1)); Wb = np.tile(w, (ub['n'], 1))
    elif base == 'A_lib':
        oa, ob = order_ng(ua, ub, n_r)
        Wa = w_seq(ua['m0'], oa, 1 - w, True); Wb = w_seq(ub['m0'], ob, w, False)
    elif base == 'B_ng':
        oa, ob = order_ng(ua, ub, n_r)
        Wa, Wb = w_ng(ua['n'], ub['n'], oa, ob, n_r); los_mode = 'ng'
    elif base.startswith('D'):                                                # D<rho%>_<order>[_sin]
        parts = base.split('_')
        rho = int(parts[0][1:]) / 100.0; key = parts[1]; shape = 'sin' if 'sin' in parts[2:] else 'lin'
        if key in ('rand', 'delay', 'power', 'pw2'):
            oa, ob = order_key(ua, ub, key, seed)
        elif key == 'ng':
            oa, ob = order_ng(ua, ub, n_r)
        elif key == 'optds':
            oa, ob = order_opt(ua, ub, n_r, rho, metrics=(0,), shape=shape)
        elif key == 'optall':
            oa, ob = order_opt(ua, ub, n_r, rho, shape=shape)
        elif key == 'optuni':
            oa, ob = order_opt(ua, ub, n_r, rho, shape=shape, target='union')
        elif key == 'optuds':
            oa, ob = order_opt(ua, ub, n_r, rho, metrics=(0,), shape=shape, target='union')
        elif key == 'optmix':
            oa, ob = order_opt(ua, ub, n_r, rho, shape=shape, target='mix')
        elif key == 'optda':                                                  # delay and the two azimuth spreads
            oa, ob = order_opt(ua, ub, n_r, rho, metrics=(0, 1, 3), shape=shape)
        else:
            raise ValueError(name)
        Wa = w_vr(ua['m0'], oa, w, rho, True, shape=shape); Wb = w_vr(ub['m0'], ob, w, rho, False, shape=shape)
    elif base.startswith('F'):                                                # F<rho%><g|l>[_order]: level and structure separated
        parts = base.split('_')
        rho = int(parts[0][1:-1]) / 100.0; level = parts[0][-1]; key = parts[1] if len(parts) > 1 else 'rand'
        PA_ = A['pg'][:, sl].sum(0); PB_ = B['pg'][:, sl].sum(0)               # local-mean power of each segment incl. LOS, per ramp snapshot
        def norm(u, Pm):
            return dict(n=u['n'], u=u['u'], m0=u['m0'] / Pm, m1=u['m1'] / Pm, m2=u['m2'] / Pm, z=u['z'] / Pm, det=tuple(x / Pm for x in u['det']))
        na_, nb_ = norm(ua, PA_.mean()), norm(ub, PB_.mean())                  # both segments at unit power: the order sees the structure only
        if key == 'rand':
            oa, ob = order_key(ua, ub, 'rand', seed)
        elif key == 'optuni':
            oa, ob = order_opt(na_, nb_, n_r, rho, target='union')
        else:
            raise ValueError(name)
        G = PA_ ** (1 - w) * PB_ ** w if level == 'g' else (1 - w) * PA_ + w * PB_
        Wa = w_vr(ua['m0'], oa, w, rho, True) * (G / PA_)[None, :]; Wb = w_vr(ub['m0'], ob, w, rho, False) * (G / PB_)[None, :]
        los_mode = ('F', G, PA_, PB_)
    elif base[0] in 'GH':                                                     # blends of F and D at the dB-linear level G, rho = 0.25, order optuni
        # Gc<cap dB|inf>: shares as if the level difference were at most cap; Gb<beta%>: shares by P^beta;
        # Hx<X>: F with each cluster's weight limited to X dB above its own power, shortfall to the other segment; Hold0: old clusters only, X = 0
        rho = 0.25
        PA_ = A['pg'][:, sl].sum(0); PB_ = B['pg'][:, sl].sum(0)
        def norm(u, Pm):
            return dict(n=u['n'], u=u['u'], m0=u['m0'] / Pm, m1=u['m1'] / Pm, m2=u['m2'] / Pm, z=u['z'] / Pm, det=tuple(q / Pm for q in u['det']))
        na_, nb_ = norm(ua, PA_.mean()), norm(ub, PB_.mean())
        dl = 10 * np.log10(PB_ / PA_)
        if base.startswith('Gc'):
            cap = np.inf if base[2:] == 'inf' else float(base[2:])
            rat = 10 ** (np.clip(dl, -cap, cap) / 10)
        elif base.startswith('Gb'):
            rat = 10 ** (float(base[2:]) / 100.0 * dl / 10)
        else:
            rat = np.ones(n_r)
        rat_m = 10 ** (np.log10(rat).mean())
        qb = w * rat / ((1 - w) + w * rat)                                    # structure share of the new segment
        if base[:2] in ('Gx', 'Gy', 'Gz', 'Gm'):                                    # exchange only where the level G is within X dB of the weaker segment
            X = float(base[2:])
            def share(ww, d_):
                f = np.minimum(X / np.maximum(np.abs(d_), 1e-9), 1.0)         # part of the ramp (in w) that is used
                return np.where(d_ < 0, np.clip((ww - (1 - f)) / f, 0, 1), np.clip(ww / f, 0, 1))
            qb = share(w, dl); rat_m = lambda ww: share(ww, dl.mean())
            if base.startswith('Gy'):                                         # one level difference per transition (mean over the ramp)
                qb = share(w, np.full(n_r, dl.mean()))
            if base.startswith('Gz'):                                         # per snapshot, made monotone
                qb = np.maximum.accumulate(qb)
            if base.startswith('Gm'):                                         # one value per transition: the largest level difference in the ramp, side from the mean
                dm = np.full(n_r, np.sign(dl.mean()) * np.abs(dl).max())
                qb = share(w, dm); rat_m = lambda ww: share(ww, dm[0])
        oa, ob = order_opt(na_, nb_, n_r, rho, target='union', ratio=rat_m)
        G = PA_ ** (1 - w) * PB_ ** w
        Wa = w_vr(ua['m0'], oa, qb, rho, True) * (G / PA_)[None, :]; Wb = w_vr(ub['m0'], ob, qb, rho, False) * (G / PB_)[None, :]
        if base[0] == 'H':
            X = 10 ** (float(base[2:] if base.startswith('Hx') else 0) / 10)
            Wa_c = np.minimum(Wa, X); Wb_c = Wb if base == 'Hold0' else np.minimum(Wb, X)
            sa_, sb_ = (Wa - Wa_c).T @ ua['m0'], (Wb - Wb_c).T @ ub['m0']     # power cut from each segment
            ta_, tb_ = Wa_c.T @ ua['m0'], Wb_c.T @ ub['m0']
            Wa = Wa_c * (1 + sb_ / np.maximum(ta_, 1e-300))[None, :]; Wb = Wb_c * (1 + sa_ / np.maximum(tb_, 1e-300))[None, :]
        los_mode = ('F', G, PA_, PB_, qb)
    elif base.startswith('E'):                                                # E<rho%>_free | _norm: independent birth and death points
        parts = base.split('_')
        rho = int(parts[0][1:]) / 100.0
        rng = np.random.default_rng(seed + 77)
        def gains(n, out):
            c = rng.uniform(rho / 2, 1 - rho / 2, n)                          # centre of each unit's transition in x
            g = np.clip((x[None, :] - c[:, None]) / rho + 0.5, 0, 1)
            g = 0.5 * (1 - np.cos(np.pi * g))
            return (1 - g) if out else g
        Wa, Wb = gains(ua['n'], True), gains(ub['n'], False)
        if parts[1] == 'norm':                                                # one scale per segment so that its power follows the ramp
            Wa = Wa * ((1 - w) * ua['m0'].sum() / np.maximum(Wa.T @ ua['m0'], 1e-300))[None, :]
            Wb = Wb * (w * ub['m0'].sum() / np.maximum(Wb.T @ ub['m0'], 1e-300))[None, :]
    else:
        raise ValueError(name)
    wA = np.zeros((A['tap'].size, N)); wB = np.zeros((B['tap'].size, N))
    nla, nlb = ua['u'] >= 0, ub['u'] >= 0
    wA[nla, :r0] = 1.0; wB[nlb, nA:] = 1.0
    wA[np.ix_(nla, np.arange(r0, nA))] = Wa[ua['u'][nla]]
    wB[np.ix_(nlb, np.arange(r0, nA))] = Wb[ub['u'][nlb]]
    # LOS: complex scales sA, sB of the two LOS coefficients (one LOS path in the output)
    ia, ib = np.flatnonzero(~nla), np.flatnonzero(~nlb)
    sA = np.zeros(N); sB = np.zeros(N)
    sA[:r0] = 1.0; sB[nA:] = 1.0
    if ia.size and ib.size:
        ga = A['pg'][ia[0], sl]; gb = B['pg'][ib[0], sl]
        if los_mode == 'ng':
            sA[sl] = 1 - w; sB[sl] = w                                        # coherent mix of the two coefficients
        else:
            tgt = (1 - w) * ga + w * gb
            if isinstance(los_mode, tuple):                                   # LOS share of each segment, mixed like the structure, at the level G
                qb_ = los_mode[4] if len(los_mode) > 4 else w
                tgt = los_mode[1] * ((1 - qb_) * ga / los_mode[2] + qb_ * gb / los_mode[3])
            fb = gb > 0
            sB[sl] = np.where(fb, np.sqrt(tgt / np.maximum(gb, 1e-300)), 0.0)
            sA[sl] = np.where(~fb & (ga > 0), np.sqrt(tgt / np.maximum(ga, 1e-300)), 0.0)
    dB = lambda v: 10 * np.log10(max(float(v), 1e-30))
    ex = dict(old=dB(Wa.max()) if Wa.size else -300.0, new=dB(Wb.max()) if Wb.size else -300.0, los=dB(max((sA[sl] ** 2).max(), (sB[sl] ** 2).max())))
    # share of the units that exceed their own power by more than 0.1 dB at some point, and the power-weighted mean of the largest excess
    return dict(wA=wA, wB=wB, sA=sA, sB=sB, ia=ia, ib=ib, r0=r0, n_r=n_r, w=w, x=x, unit=unit, ex=ex,
                ex_units=np.concatenate([10 * np.log10(np.maximum(Wa.max(1), 1e-30)), 10 * np.log10(np.maximum(Wb.max(1), 1e-30))]) if Wa.size and Wb.size else np.zeros(0))

# ---------------------------------------------------------------- metrics along the track
def _pad(a, N):
    out = np.zeros(a.shape[:-1] + (N,), a.dtype); out[..., :a.shape[-1]] = a
    return out

def prepare(o):
    """Pads segment A to all snapshots and precomputes per-path moments per snapshot."""
    N = o['B']['pg'].shape[1]
    for k in ('A', 'B'):
        s = o[k]
        for f in ('coeff', 'delay', 'pg', 'ang'):
            if s[f].shape[-1] != N:
                s[f] = _pad(s[f], N)
        s['e'] = np.exp(1j * s['ang'])                                         # [P, 4, N]
        s['pc'] = (np.abs(s['coeff']) ** 2).astype(np.float64)                  # [R, T, P, N]
    return o

def curves(o, sch, sub=8, fast=True):
    """All metrics per snapshot for one schedule. Path-gain based ('local mean') except the fading part."""
    A, B = o['A'], o['B']
    N = B['pg'].shape[1]
    out = {}
    # effective path gains: NLOS by weights, LOS by |scale|^2 (coherent mix: from the coefficients)
    gA = sch['wA'] * A['pg']; gB = sch['wB'] * B['pg']
    ia, ib = sch['ia'], sch['ib']
    hl = None
    if ia.size and ib.size:
        hl = sch['sA'][None, None, :] * A['coeff'][:, :, ia[0], :] + sch['sB'][None, None, :] * B['coeff'][:, :, ib[0], :]   # [R, T, N]
        pa = (np.abs(A['coeff'][:, :, ia[0], :]) ** 2).mean((0, 1)); pb = (np.abs(B['coeff'][:, :, ib[0], :]) ** 2).mean((0, 1))
        ph = (np.abs(hl) ** 2).mean((0, 1))
        use_b = pb > 0
        with np.errstate(all='ignore'):
          glos = np.where(use_b, ph / np.maximum(pb, 1e-300) * B['pg'][ib[0]], np.where(pa > 0, ph / np.maximum(pa, 1e-300) * A['pg'][ia[0]], 0.0))
        gA[ia[0]] = 0; gB[ib[0]] = 0
        los_src_b = use_b
    else:
        glos = np.zeros(N); los_src_b = np.ones(N, bool)
    pn = gA.sum(0) + gB.sum(0)
    out['P_dB'] = 10 * np.log10(pn + glos)
    out['KF_dB'] = 10 * np.log10(np.maximum(glos, 1e-30) / pn)
    # spreads; the LOS takes delay and angles of its source path
    def mom(g, s):
        return g.sum(0), (g * s['delay']).sum(0), (g * s['delay'] ** 2).sum(0), np.einsum('pn,pkn->nk', g, s['e'])
    a0, a1, a2, az = mom(gA, A); b0, b1, b2, bz = mom(gB, B)
    if ia.size and ib.size:
        dl = np.where(los_src_b, B['delay'][ib[0]], A['delay'][ia[0]])
        el = np.where(los_src_b[None, :], B['e'][ib[0]], A['e'][ia[0]])          # [4, N]
        l0, l1, l2, lz = glos, glos * dl, glos * dl * dl, (glos[None, :] * el).T
    else:
        l0 = l1 = l2 = 0.0; lz = 0.0
    ds, asp = spreads(a0 + b0 + l0, a1 + b1 + l1, a2 + b2 + l2, az + bz + lz)
    out['DS_ns'] = ds * 1e9
    for k, nm in enumerate(('ASD', 'ESD', 'ASA', 'ESA')):
        out[nm + '_deg'] = np.rad2deg(asp[:, k])
    # counts
    onA, onB = sch['wA'] > 0, sch['wB'] > 0
    has_los = glos > 0
    out['n_path'] = onA.sum(0) + onB.sum(0) + 1                                 # the LOS path is kept in the output
    def per_key(key, on, g):
        vals, inv = np.unique(key, return_inverse=True)
        M = np.zeros((vals.size, key.size)); M[inv, np.arange(key.size)] = 1
        return (M @ on) > 0, M @ g
    tA_on, tA_p = per_key(A['tap'], onA, gA); tB_on, tB_p = per_key(B['tap'], onB, gB)
    cA_on, _ = per_key(A['tap'] // 3, onA, gA); cB_on, _ = per_key(B['tap'] // 3, onB, gB)
    out['n_tap'] = tA_on.sum(0) + tB_on.sum(0) + 1
    out['n_cluster'] = cA_on.sum(0) + cB_on.sum(0)
    tp = np.concatenate([tA_p, tB_p, glos[None, :]], 0)
    out['n_eff'] = tp.sum(0) ** 2 / np.maximum((tp ** 2).sum(0), 1e-300)        # participation number of the tap powers
    out['n_tap20'] = (tp > tp.sum(0)[None, :] * 10 ** (-2.5)).sum(0)             # taps above -25 dB of the total power
    # coherence bandwidth (0.5) from the power delay profile, on a subset of the snapshots
    idx = np.arange(0, N, sub)
    df = np.concatenate([[0.0], np.logspace(4, 8, 161)])
    bc = np.empty(idx.size)
    for j, n in enumerate(idx):
        p = np.concatenate([gA[:, n], gB[:, n], [glos[n]]]); tau = np.concatenate([A['delay'][:, n], B['delay'][:, n], [0.0]])
        k = p > 0
        R = np.abs(np.exp(-2j * np.pi * np.outer(df, tau[k])) @ p[k]) / p[k].sum()
        below = np.flatnonzero(R < 0.5)
        if below.size == 0:
            bc[j] = df[-1]
        else:
            i = below[0]; bc[j] = df[i - 1] + (R[i - 1] - 0.5) / (R[i - 1] - R[i]) * (df[i] - df[i - 1])
    out['Bc_MHz'] = np.interp(np.arange(N), idx, bc / 1e6)
    # local-mean Rx and Tx covariance: effective rank and correlation of adjacent elements
    rr = np.empty(idx.size); rt = np.empty(idx.size); cr = np.empty(idx.size); ct = np.empty(idx.size)
    for j, n in enumerate(idx):
        ca = A['coeff'][:, :, :, n] * np.sqrt(sch['wA'][:, n])[None, None, :]
        cb = B['coeff'][:, :, :, n] * np.sqrt(sch['wB'][:, n])[None, None, :]
        Rr = np.einsum('rtp,stp->rs', ca, ca.conj()) + np.einsum('rtp,stp->rs', cb, cb.conj())
        Rt = np.einsum('rtp,rup->tu', ca, ca.conj()) + np.einsum('rtp,rup->tu', cb, cb.conj())
        if hl is not None:
            Rr = Rr + hl[:, :, n] @ hl[:, :, n].conj().T; Rt = Rt + hl[:, :, n].T @ hl[:, :, n].conj()
        for Rm, er, cc in ((Rr, rr, cr), (Rt, rt, ct)):
            lam = np.maximum(np.linalg.eigvalsh(Rm), 0); lam = lam / lam.sum(); lam = lam[lam > 0]
            er[j] = np.exp(-(lam * np.log(lam)).sum())
            cc[j] = np.abs(Rm[0, 1]) / np.sqrt(Rm[0, 0].real * Rm[1, 1].real)
    out['rank_rx'] = np.interp(np.arange(N), idx, rr); out['rank_tx'] = np.interp(np.arange(N), idx, rt)
    out['corr_rx'] = np.interp(np.arange(N), idx, cr); out['corr_tx'] = np.interp(np.arange(N), idx, ct)
    if fast:
        # narrowband coefficients and wideband power (sum of the tap powers), with their incoherent means
        sa, sb = np.sqrt(sch['wA']).astype(np.float32), np.sqrt(sch['wB']).astype(np.float32)
        ca = A['coeff'] * sa[None, None]; cb = B['coeff'] * sb[None, None]
        if hl is not None:
            ca[:, :, ia[0], :] = 0; cb[:, :, ib[0], :] = 0
        def taps(c, tap):
            st = np.flatnonzero(np.r_[True, np.diff(tap) != 0])
            return np.add.reduceat(c, st, axis=2)
        tA, tB = taps(ca, A['tap']), taps(cb, B['tap'])
        h = tA.sum(2) + tB.sum(2)
        pw = (np.abs(tA) ** 2).sum(2) + (np.abs(tB) ** 2).sum(2)
        pm = np.einsum('rtpn,pn->rtn', A['pc'], sch['wA'] * (A['ptype'] != 0)[:, None] if hl is not None else sch['wA']) + \
             np.einsum('rtpn,pn->rtn', B['pc'], sch['wB'] * (B['ptype'] != 0)[:, None] if hl is not None else sch['wB'])
        if hl is not None:
            h = h + hl; pw = pw + np.abs(hl) ** 2; pm = pm + np.abs(hl) ** 2
        out['h'] = h; out['pw'] = pw; out['pm'] = pm
        sv = np.linalg.svd(np.moveaxis(h, 2, 0), compute_uv=False)              # [N, 4]
        out['sv_spread_dB'] = 20 * np.log10(sv[:, 0] / np.maximum(sv[:, -1], 1e-30))
        out['cap'] = np.log2(1 + 10.0 * sv ** 2 / np.maximum(pm.mean((0, 1)), 1e-300)[:, None] / 4).sum(1)   # 10 dB SNR, equal power
    return out

def fading_stats(c, sl, dx, lam):
    """Fast-fading statistics of the narrowband coefficients over the snapshots sl, normalised by the local incoherent mean."""
    g = np.abs(c['h'][:, :, sl]) ** 2 / c['pm'][:, :, sl]                        # [R, T, n]
    hn = c['h'][:, :, sl] / np.sqrt(c['pm'][:, :, sl])
    gw = c['pw'][:, :, sl] / c['pm'][:, :, sl]
    st = {}
    st['mean_dB'] = 10 * np.log10(g.mean())
    st['AF'] = g.var() / g.mean() ** 2                                          # amount of fading: 1 for Rayleigh, (2K+1)/(K+1)^2 for Rice
    st['p_m10'] = (g < 0.1 * g.mean()).mean(); st['p_m20'] = (g < 0.01 * g.mean()).mean()
    thr = 0.1 * g.mean()
    up = ((g[:, :, :-1] < thr) & (g[:, :, 1:] >= thr)).sum() / (g.shape[0] * g.shape[1])
    st['lcr_m10_per_lam'] = up / ((g.shape[2] - 1) * dx) * lam
    n = hn.shape[2]; lags = np.arange(0, min(n // 2, int(round(3 * lam / dx)) + 1))
    acf = np.array([np.mean(hn[:, :, k:] * hn[:, :, :n - k].conj()) for k in lags])
    acf = np.abs(acf) / np.abs(acf[0])
    below = np.flatnonzero(acf < 0.5)
    st['dcoh_lam'] = (lags[below[0]] * dx / lam) if below.size else np.nan
    st['AF_wb'] = gw.var() / gw.mean() ** 2                                      # fading of the wideband power
    st['std_wb_dB'] = np.std(10 * np.log10(gw))
    st['sv_spread_dB'] = np.median(c['sv_spread_dB'][sl]); st['cap'] = np.mean(c['cap'][sl])
    return st

SMOOTH = ('DS_ns', 'ASD_deg', 'ESD_deg', 'ASA_deg', 'ESA_deg', 'Bc_MHz', 'rank_rx', 'rank_tx', 'corr_rx', 'corr_tx', 'n_eff')

def smooth_scores(c, cA, cB, sch, dx):
    """Scores of one metric curve X over the ramp against the pure segments XA, XB at the same snapshots (relative to their mean):
    over: excursion outside [min(XA, XB), max(XA, XB)]; dev: rms deviation from (1 - w) XA + w XB; tv: excess total variation;
    step: largest change within 0.25 m."""
    r0, n_r, w = sch['r0'], sch['n_r'], sch['w']
    sl = slice(r0, r0 + n_r)
    k = max(1, int(round(0.25 / dx)))
    s = {}
    for m in SMOOTH:
        X, XA, XB = c[m][sl], cA[m][sl], cB[m][sl]
        ref = 0.5 * (XA + XB).mean()
        hi, lo = np.maximum(XA, XB), np.minimum(XA, XB)
        s[m + ':over'] = max(0.0, (X - hi).max(), (lo - X).max()) / ref
        s[m + ':dev'] = np.sqrt(((X - ((1 - w) * XA + w * XB)) ** 2).mean()) / ref
        full = np.r_[XA[0], X, XB[-1]]
        s[m + ':tv'] = (np.abs(np.diff(full)).sum() - abs(full[-1] - full[0])) / ref
        s[m + ':step'] = np.abs(full[k:] - full[:-k]).max() / ref
    tgt = 10 * np.log10((1 - w) * 10 ** (cA['P_dB'][sl] / 10) + w * 10 ** (cB['P_dB'][sl] / 10))
    s['P:maxdev_dB'] = np.abs(c['P_dB'][sl] - tgt).max()
    s['P:step_dB'] = np.abs(np.r_[cA['P_dB'][r0 - 1], c['P_dB'][sl]][k:] - np.r_[cA['P_dB'][r0 - 1], c['P_dB'][sl]][:-k]).max()
    for m in ('n_path', 'n_tap', 'n_cluster', 'n_tap20', 'n_eff'):
        s[m + ':max'] = c[m][sl].max(); s[m + ':mean'] = c[m][sl].mean()
        s[m + ':A'] = cA[m][sl].mean(); s[m + ':B'] = cB[m][sl].mean()
    return s

def pure(o, which, overlap=0.5):
    """Schedule of a pure segment over the snapshots of zone A (reference curves)."""
    A, B, nA = o['A'], o['B'], o['nA']
    N = B['pg'].shape[1]
    n_r = o.get('n_r', int(np.floor(overlap * nA + 0.5))); r0 = nA - n_r
    w, x = ramp(n_r)
    nla, nlb = A['ptype'] != 0, B['ptype'] != 0
    wA = np.zeros((A['tap'].size, N)); wB = np.zeros((B['tap'].size, N))
    sA = np.zeros(N); sB = np.zeros(N)
    if which == 'A':
        wA[nla, :nA] = 1; sA[:nA] = 1
    else:
        wB[nlb, :] = 1; sB[:] = 1
    return dict(wA=wA, wB=wB, sA=sA, sB=sB, ia=np.flatnonzero(~nla), ib=np.flatnonzero(~nlb), r0=r0, n_r=n_r, w=w, x=x, unit='cluster')
