"""Figures of the report from the results of cf_run.py.

Example:  python3 cf_figures.py --end results_end --centred results_centred --out figures
The figure of the weights of the chosen rule is computed with the library (one transition); --no-weights skips it.
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from cf_analyse import load, score, SPREADS

PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6', '#000000', '#eda100']
INK = '#52514e'
plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'axes.edgecolor': INK, 'axes.labelcolor': 'black',
                     'xtick.color': INK, 'ytick.color': INK, 'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.spines.top': False,
                     'axes.spines.right': False, 'font.size': 8, 'lines.linewidth': 1.4, 'legend.frameon': False, 'legend.fontsize': 7})
DPI = 80


def save(fig, path):
    """Small palette PNG."""
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    try:
        from PIL import Image
        Image.open(path).convert('RGB').quantize(colors=48, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE).save(path, optimize=True)
    except ImportError:
        pass
    print('written:', path, os.path.getsize(path), 'bytes')


def transition(res, key, seed, methods, labels, title, path):
    r = [x for x in res[key] if x['seed'] == seed][0]
    st, dx = r['stride'], r['dx']
    panels = (('P_dB', 'sum of the path gains [dB]'), ('KF_dB', 'K-factor [dB]'), ('DS_ns', 'delay spread [ns]'), ('ASA_deg', 'azimuth spread of arrival [deg]'),
              ('n_tap20', 'taps above -25 dB of the total'), ('rank_rx', 'effective rank, Rx covariance'))
    fig, ax = plt.subplots(2, 3, figsize=(10, 5.2), sharex=True)
    x = (np.arange(r['curves']['B']['P_dB'].size) * st - r['change']) * dx
    for a, (k, lab) in zip(ax.ravel(), panels):
        a.axvspan((r['r0'] - r['change']) * dx, (r['nA'] - r['change']) * dx, color='#f0efec', zorder=0)
        a.axvline(0, color=INK, lw=0.6)
        clip = (lambda y: np.maximum(y, -40)) if k == 'KF_dB' else (lambda y: y)
        ya = clip(r['curves']['A'][k].astype(float))
        ya[r['nA'] // st:] = np.nan
        a.plot(x, ya, color=INK, lw=0.9, ls='--', label='old segment alone')
        a.plot(x, clip(r['curves']['B'][k].astype(float)), color=INK, lw=0.9, ls=':', label='new segment alone')
        for i, m in enumerate(methods):
            a.plot(x, clip(r['curves'][m][k].astype(float)), color=PAL[i], label=labels[i])
        a.set_ylabel(lab)
        a.set_xlim((r['r0'] - r['change']) * dx - 2, (r['nA'] - r['change']) * dx + 2)
    for a in ax[-1]:
        a.set_xlabel('position relative to the change row [m]')
    ax[0, 0].legend(loc='lower left')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    save(fig, path)


def counts(res, methods, labels, path):
    allr = [r for k in res for r in res[k]]
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.0), sharex=True)
    for a, (k, lab) in zip(ax, (('n_cluster', 'clusters with power / larger segment'), ('n_tap20', 'taps above -25 dB of the total / larger segment'))):
        for i, m in enumerate(methods):
            Y = []
            for r in allr:
                sl = slice(r['r0'] // r['stride'], r['nA'] // r['stride'])
                Y.append(r['curves'][m][k][sl] / max(r['curves']['A'][k][sl].mean(), r['curves']['B'][k][sl].mean()))
            Y = np.array(Y)
            a.plot(np.linspace(0, 1, Y.shape[1]), Y.mean(0), color=PAL[i], label=labels[i])
        a.set_ylabel(lab)
        a.set_xlabel('position in the ramp')
    ax[0].legend()
    fig.tight_layout()
    save(fig, path)


def tradeoff(res, path):
    allr = [r for k in res for r in res[k]]
    names = set(allr[0]['curves'])
    fams = (('Gx%s', ('3', '6', '10', '15', '20'), 'Gx: exchange within X dB of the weaker segment'), ('Gc%s', ('3', '6', '10', '15', '20', 'inf'), 'Gc: capped level difference'),
            ('Gb%s', ('25', '50', '75'), 'Gb: shares by P^beta'), ('Hx%s', ('6', '10'), 'Hx: weight limit'))
    single = [m for m in ('F25g_optuni', 'D25_optuni', 'Gm10') if m in names]
    need = [p % v for p, vs, _ in fams for v in vs if p % v in names] + single
    sc = {m: [score(r, m) for r in allr] for m in need}
    ys = (('delay spread: jaggedness (tv) [%]', lambda s: 100 * s['DS_ns:tv']), ('angular spreads: jaggedness (tv) [%]', lambda s: 100 * np.mean([s[q + ':tv'] for q in SPREADS[1:]])),
          ('K-factor step within 0.25 m [dB]', lambda s: s['KF:step']))
    fig, ax = plt.subplots(1, 3, figsize=(10, 3.6))
    xv = lambda m: np.percentile([s['ex'] for s in sc[m]], 90)
    for a, (lab, fun) in zip(ax, ys):
        val = lambda m: np.median([fun(s) for s in sc[m]])
        for i, (pat, vals, nm) in enumerate(fams):
            ms = [pat % v for v in vals if pat % v in names]
            a.plot([xv(m) for m in ms], [val(m) for m in ms], '-o', color=PAL[i], ms=4, label=nm)
            for m in ms:
                a.annotate(m[2:], (xv(m), val(m)), textcoords='offset points', xytext=(3, 4), fontsize=6, color=INK)
        for j, m in enumerate(single):
            a.plot([xv(m)], [val(m)], marker='sD*'[j], ls='', ms=7 if j < 2 else 10, color=('black' if m == 'Gm10' else (PAL[3] if j == 0 else INK)),
                   label={'F25g_optuni': 'F: no limit', 'D25_optuni': 'D: own levels', 'Gm10': 'chosen rule (Gm10)'}[m])
        a.set_xlabel('largest cluster excess, 90th percentile [dB]')
        a.set_ylabel(lab)
        a.set_ylim(bottom=0)
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.legend(h, l, loc='lower center', ncol=4)
    save(fig, path)


def weights(path, work_dir):
    import cf_segments as gen
    import cf_schedules as xf
    np.seterr(divide='ignore', invalid='ignore')
    o = gen.segments('UMa_LOS_NLOS', 3, '38.901', 'phase', work_dir, tag='fig', shift=200)
    xf.prepare(o)
    s = xf.schedule(o, 'Gm10', seed=3)
    sl = slice(s['r0'], s['r0'] + s['n_r'])
    A, B = o['A'], o['B']
    pos = (np.arange(s['n_r']) + s['r0'] - o['change']) * gen.DX
    PA, PB, w = A['pg'][:, sl].sum(0), B['pg'][:, sl].sum(0), s['w']
    G = PA ** (1 - w) * PB ** w
    fig, ax = plt.subplots(1, 4, figsize=(11, 2.9))
    ax[0].plot(pos, 10 * np.log10(PA), color=INK, ls='--', lw=0.9, label='old segment alone')
    ax[0].plot(pos, 10 * np.log10(PB), color=INK, ls=':', lw=0.9, label='new segment alone')
    ax[0].plot(pos, 10 * np.log10(G), color=PAL[2], label='level G')
    ax[0].set_ylabel('sum of the path gains [dB]')

    def first(seg):                                           # one ray per NLOS cluster
        key, nl = seg['tap'] // 3, seg['ptype'] != 0
        return np.flatnonzero(nl)[np.unique(key[nl], return_index=True)[1]]
    for j, (wgt, seg, P_, nm) in enumerate(((s['wA'], A, PA, 'old clusters'), (s['wB'], B, PB, 'new clusters'))):
        for n, q in enumerate(first(seg)):
            ax[1].plot(pos, wgt[q, sl] * P_ / G, color=PAL[j], lw=0.7, label=nm if n == 0 else None)
            ax[2].plot(pos, 10 * np.log10(np.maximum(wgt[q, sl], 1e-4)), color=PAL[j], lw=0.7)
    ax[1].plot(pos, w, color=INK, lw=0.9, ls='--', label='ramp w')
    ax[1].set_ylabel('part of each cluster that is present')
    ax[2].axhline(10, color=INK, lw=0.8, ls='--')
    ax[2].set_ylim(-40, 12)
    ax[2].set_ylabel('cluster weight against own power [dB]')
    ia, ib = s['ia'][0], s['ib'][0]
    los = s['sA'][sl] ** 2 * A['pg'][ia, sl] + s['sB'][sl] ** 2 * B['pg'][ib, sl]
    nlos = (s['wA'][:, sl] * A['pg'][:, sl]).sum(0) + (s['wB'][:, sl] * B['pg'][:, sl]).sum(0) - s['wA'][ia, sl] * A['pg'][ia, sl] - s['wB'][ib, sl] * B['pg'][ib, sl]
    ax[3].plot(pos, 10 * np.log10(np.maximum(los, 1e-30) / nlos), color=PAL[2], label='chosen rule')
    ax[3].plot(pos, 10 * np.log10(A['pg'][ia, sl] / (PA - A['pg'][ia, sl])), color=INK, ls='--', lw=0.9, label='old segment alone')
    ax[3].set_ylim(-30, 10)
    ax[3].set_ylabel('K-factor [dB]')
    for a in ax:
        a.axvline(0, color=INK, lw=0.6)
        a.set_xlabel('position relative to the change row [m]')
    for a in (ax[0], ax[1], ax[3]):
        a.legend()
    fig.tight_layout()
    save(fig, path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--end', required=True, help='results with the ramp ending at the change row (cf_run.py --methods all)')
    ap.add_argument('--centred', required=True, help='results with the centred ramp (cf_run.py --methods centred --shift 200)')
    ap.add_argument('--out', required=True, help='output folder (created)')
    ap.add_argument('--no-weights', action='store_true', help='skip the figure that needs the library')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    end, cen = load(a.end), load(a.centred)
    lab = ('present "taps" rule', 'present "rays" rule (union)', 'chosen rule')
    transition(cen, ('UMa_LOS_NLOS', '38.901'), 3, ('A_lib', 'C_union', 'Gm10'), lab, 'UMa LOS to NLOS, method 38.901, seed 3; ramp of 10 m (shaded) centred on the change row',
               os.path.join(a.out, 'fig_transition_los_nlos.png'))
    transition(cen, ('UMa_NLOS_O2I', '38.901'), 2, ('A_lib', 'C_union', 'Gm10'), lab, 'UMa NLOS to O2I, method 38.901, seed 2; ramp of 10 m (shaded) centred on the change row',
               os.path.join(a.out, 'fig_transition_o2i.png'))
    counts(end, ('A_lib', 'C_union', 'D25_optuni', 'F100g', 'Gm10'), ('present "taps" rule', 'union at own levels', 'staggered at own levels (D25_optuni)', 'union at the dB-linear level (F100g)', 'chosen rule'),
           os.path.join(a.out, 'fig_counts.png'))
    tradeoff(end, os.path.join(a.out, 'fig_blend_tradeoff.png'))
    if not a.no_weights:
        weights(os.path.join(a.out, 'fig_chosen_weights.png'), a.out)


if __name__ == '__main__':
    main()
