"""Tables of the scores of all schedules in a results folder of cf_run.py (Markdown, written to the output folder).

Example:  python3 cf_analyse.py results_end tables_end
"""
import argparse
import glob
import os
import pickle
import numpy as np

SPREADS = ('DS_ns', 'ASD_deg', 'ESD_deg', 'ASA_deg', 'ESA_deg')
LOS_TO_NLOS, NLOS_TO_LOS, NO_LOS = ('UMa_LOS_NLOS', 'InF_LOS_NLOS'), ('UMi_NLOS_LOS',), ('UMa_NLOS_O2I',)


def load(folder):
    res = {}
    for f in sorted(glob.glob(os.path.join(folder, 'res_*.pkl'))):
        with open(f, 'rb') as fh:
            r = pickle.load(fh)
        res[(r[0]['case'], r[0]['gm'])] = r
    return res


def reference(r, m):
    """The full superposition a schedule is compared with: at own levels (C_union) or at the dB-linear level (F100g)."""
    sep = m[0] in 'GH' or (m[0] == 'F' and m.split('_')[0].endswith('g'))
    name = 'F100g' if sep else 'C_union'
    return name if name in r['curves'] else None


def score(r, m):
    """Scores of schedule m in one transition, from the stored curves over the ramp."""
    st, dx = r['stride'], r['dx']
    sl = slice(r['r0'] // st, r['nA'] // st)
    w = r['w'][::st]
    c, cA, cB = r['curves'][m], r['curves']['A'], r['curves']['B']
    k = max(1, int(round(0.25 / (st * dx))))                 # 0.25 m
    ref = reference(r, m)
    s = {}
    for q in SPREADS + ('Bc_MHz', 'rank_rx', 'rank_tx'):
        X, XA, XB = c[q][sl].astype(float), cA[q][sl].astype(float), cB[q][sl].astype(float)
        norm = 0.5 * (XA + XB).mean()
        hi, lo = np.maximum(XA, XB), np.minimum(XA, XB)
        s[q + ':out'] = max(0.0, (X - hi).max(), (lo - X).max()) / norm
        full = np.r_[XA[0], X, XB[-1]]
        s[q + ':tv'] = (np.abs(np.diff(full)).sum() - abs(full[-1] - full[0])) / norm
        s[q + ':dev'] = np.sqrt(((X - r['curves'][ref][q][sl].astype(float)) ** 2).mean()) / norm if ref else np.nan
    PA, PB = cA['P_dB'][sl].astype(float), cB['P_dB'][sl].astype(float)
    P = c['P_dB'][sl].astype(float)
    s['P:devlin'] = np.abs(P - 10 * np.log10((1 - w) * 10 ** (PA / 10) + w * 10 ** (PB / 10))).max()
    s['P:devG'] = np.abs(P - ((1 - w) * PA + w * PB)).max()
    pf = np.r_[PA[0], P, PB[-1]]
    s['P:step'] = np.abs(pf[k:] - pf[:-k]).max()
    clip = lambda x: np.maximum(x.astype(float), -30.0)      # the K-factor is clipped at -30 dB
    kf = np.r_[clip(c['KF_dB'][sl]), clip(cB['KF_dB'][sl])[-1]]
    s['KF:step'] = np.abs(kf[k:] - kf[:-k]).max()
    for q in ('n_eff', 'n_tap20'):
        s[q + ':minr'] = (c[q][sl] / np.minimum(cA[q][sl], cB[q][sl])).min()
    s['clusters'] = c['n_cluster'][sl].max() / max(cA['n_cluster'][sl].mean(), cB['n_cluster'][sl].mean())
    s['taps'] = c['n_tap'][sl].max()
    s['paths'] = c['n_path'][sl].mean() / (cA['n_path'][sl].mean() + cB['n_path'][sl].mean() - 1)   # against all paths of both segments, one LOS
    e = r['ex'][m]
    s['ex'] = max(e['old'], e['new'], 0.0)
    s['ex_los'] = max(e['los'], 0.0)
    u = r['exu'][m]
    s['ex_share'] = float(np.mean(u > 3.0)) if u.size else 0.0
    s['dlev'] = float(np.mean(PB - PA))
    off = (r['nA'] - r['change']) * dx                        # ramp end behind the change row [m]

    def half(X, XA, XB):                                      # position of the half-way point relative to the change row [m]
        a, b = XA[0], XB[-1]
        if abs(b - a) < 1e-9:
            return np.nan
        i = np.flatnonzero((X - a) / (b - a) >= 0.5)
        return ((i[0] - X.size) * st * dx if i.size else 0.0) + off
    for q, f in (('DS_ns', lambda x: x.astype(float)), ('ASA_deg', lambda x: x.astype(float)), ('KF_dB', clip)):
        s['half:' + q] = half(f(c[q][sl]), f(cA[q][sl]), f(cB[q][sl]))
    i = min(r['change'] // st, c['KF_dB'].size - 1)           # the change row
    ia = min(i, r['nA'] // st - 1)
    s['KF:change'] = float(c['KF_dB'][i]) - (float(cA['KF_dB'][ia]) if r['case'] in LOS_TO_NLOS else float(cB['KF_dB'][i]))
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results', help='results folder of cf_run.py')
    ap.add_argument('out', help='output folder (created); the file tables.md is written there')
    a = ap.parse_args()
    res = load(a.results)
    if not res:
        raise SystemExit('no res_*.pkl in ' + a.results)
    allr = [r for k in res for r in res[k]]
    methods = [m for m in allr[0]['curves'] if m not in ('A', 'B')]
    for r in allr:
        r['score'] = {m: score(r, m) for m in methods}
    out = []
    P = lambda *x: out.append(' '.join(str(v) for v in x))
    val = lambda recs, m, key: np.array([r['score'][m][key] for r in recs])
    AS = lambda recs, m, q: np.array([np.mean([r['score'][m][s + ':' + q] for s in SPREADS[1:]]) for r in recs])

    def table(recs, title):
        P('\n### ' + title + f' ({len(recs)} transitions)\n')
        cols = ['Method', 'clusters max / larger segment', 'taps max', 'paths in ramp / all paths of both segments', 'excess median [dB]', 'excess p90 [dB]',
                'excess max [dB]', 'clusters > 3 dB above own power [%]', 'largest LOS excess [dB]', 'level against power-linear, median / max [dB]',
                'level against dB-linear, max [dB]', 'DS out [%]', 'DS out p90 [%]', 'DS tv [%]', 'DS dev [%]', 'AS out [%]', 'AS out p90 [%]', 'AS tv [%]',
                'AS dev [%]', 'Bc out [%]', 'rank out [%]', 'KF step [dB]', 'KF step, LOS changes [dB]', 'level step [dB]',
                'significant taps min ratio (median / p10)', 'N_eff min ratio (median / p10)']
        P('| ' + ' | '.join(cols) + ' |')
        P('|' + '---|' * len(cols))
        los = [r for r in recs if r['case'] not in NO_LOS]
        for m in methods:
            v = lambda key: val(recs, m, key)
            row = [m, '%.2f' % v('clusters').mean(), '%.1f' % v('taps').mean(), '%.2f' % v('paths').mean(), '%.1f' % np.median(v('ex')),
                   '%.1f' % np.percentile(v('ex'), 90), '%.1f' % v('ex').max(), '%.0f' % (100 * v('ex_share').mean()), '%.1f' % v('ex_los').max(),
                   '%.2f / %.1f' % (np.median(v('P:devlin')), v('P:devlin').max()), '%.1f' % v('P:devG').max(),
                   '%.1f' % (100 * np.median(v('DS_ns:out'))), '%.1f' % (100 * np.percentile(v('DS_ns:out'), 90)), '%.1f' % (100 * np.median(v('DS_ns:tv'))),
                   '%.1f' % (100 * np.nanmedian(v('DS_ns:dev'))), '%.1f' % (100 * np.median(AS(recs, m, 'out'))), '%.1f' % (100 * np.percentile(AS(recs, m, 'out'), 90)),
                   '%.1f' % (100 * np.median(AS(recs, m, 'tv'))), '%.1f' % (100 * np.nanmedian(AS(recs, m, 'dev'))), '%.1f' % (100 * np.median(v('Bc_MHz:out'))),
                   '%.1f' % (100 * np.median(0.5 * (v('rank_rx:out') + v('rank_tx:out')))), '%.1f' % np.median(v('KF:step')),
                   ('%.1f' % np.median(val(los, m, 'KF:step'))) if los else '-', '%.1f' % np.median(v('P:step')),
                   '%.2f / %.2f' % (np.median(v('n_tap20:minr')), np.percentile(v('n_tap20:minr'), 10)),
                   '%.2f / %.2f' % (np.median(v('n_eff:minr')), np.percentile(v('n_eff:minr'), 10))]
            P('| ' + ' | '.join(row) + ' |')

    dl = np.abs(val(allr, methods[0], 'dlev'))
    P('Level difference of the two segments in the ramp [dB]: median %.1f, p90 %.1f, max %.1f.' % (np.median(dl), np.percentile(dl, 90), dl.max()))
    P('Ramp end behind the change row: %.1f m.' % ((allr[0]['nA'] - allr[0]['change']) * allr[0]['dx']))
    table(allr, 'All cases')
    for k in res:
        table(res[k], '%s, method %s' % k)
    P('\n### Half-way positions relative to the change row, median (10th to 90th percentile) [m]\n')
    P('| Method | K-factor, LOS to NLOS | K-factor, NLOS to LOS | DS, LOS to NLOS | DS, NLOS to LOS | DS, NLOS to O2I | ASA, LOS to NLOS | ASA, NLOS to LOS | ASA, NLOS to O2I |')
    P('|' + '---|' * 9)
    sel = lambda cases: [r for r in allr if r['case'] in cases]
    for m in methods:
        f = lambda key, cases: ('%.1f (%.1f to %.1f)' % tuple(np.nanpercentile(val(sel(cases), m, key), q) for q in (50, 10, 90))) if sel(cases) else '-'
        P('| ' + ' | '.join([m, f('half:KF_dB', LOS_TO_NLOS), f('half:KF_dB', NLOS_TO_LOS)] + [f('half:' + q, cs) for q in ('DS_ns', 'ASA_deg') for cs in (LOS_TO_NLOS, NLOS_TO_LOS, NO_LOS)]) + ' |')
    P('\n### K-factor at the change row against the segment with the LOS path alone, median [dB]\n')
    P('| Method | LOS to NLOS | NLOS to LOS |')
    P('|---|---|---|')
    for m in methods:
        f = lambda cases: ('%.1f' % np.median(val(sel(cases), m, 'KF:change'))) if sel(cases) else '-'
        P('| %s | %s | %s |' % (m, f(LOS_TO_NLOS), f(NLOS_TO_LOS)))
    ks = ('AF', 'p_m10', 'p_m20', 'lcr_m10_per_lam', 'dcoh_lam', 'AF_wb', 'std_wb_dB', 'sv_spread_dB', 'cap')
    for gm in sorted({k[1] for k in res}):
        recs = [r for k in res if k[1] == gm for r in res[k]]
        P('\n### Fading over the ramp, method %s, means over %d transitions\n' % (gm, len(recs)))
        P('| Channel | AF | P(< -10 dB) | P(< -20 dB) | LCR(-10 dB) per wavelength | d_coh [wavelengths] | AF wideband | std wideband [dB] | SV spread [dB] | capacity [bit/s/Hz] |')
        P('|' + '---|' * 10)
        for m in ['A', 'B'] + methods:
            vv = [np.nanmean([r['fade'][m][q] for r in recs]) for q in ks]
            P('| %s | %.3f | %.4f | %.5f | %.3f | %.3f | %.3f | %.2f | %.1f | %.2f |' % (({'A': 'old segment alone', 'B': 'new segment alone'}.get(m, m),) + tuple(vv)))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'tables.md'), 'w') as f:
        f.write('\n'.join(out) + '\n')
    print('written:', os.path.join(a.out, 'tables.md'))


if __name__ == '__main__':
    main()
