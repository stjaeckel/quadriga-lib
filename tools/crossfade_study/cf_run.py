"""Runs cross-fade schedules on generated transitions and stores the curves of all metrics per transition.

Example:  python3 cf_run.py --out results_end --methods all
          python3 cf_run.py --out results_centred --methods centred --shift 200
"""
import argparse
import multiprocessing
import os
import pickle
import time
import numpy as np
import cf_segments as gen
import cf_schedules as xf

SECTION7 = ['A_lib', 'B_ng', 'C_union', 'D0_rand', 'D0_ng', 'D0_optds', 'D0_optall', 'D10_rand', 'D10_optall', 'D25_rand', 'D25_delay', 'D25_power',
            'D25_pw2', 'D25_optds', 'D25_optda', 'D25_optall', 'D25_optmix', 'D25_optall_sin', 'D25_optall_tap', 'D50_rand', 'D50_optall', 'D75_rand',
            'D75_optall', 'E25_free', 'E25_norm', 'D0_optuni', 'D10_optuni', 'D25_optuni', 'D50_optuni', 'D25_optuds', 'D25_optuni_coef',
            'D25_optuni_tap', 'D25_optuni_sin', 'F100g', 'F100l', 'F0g_optuni', 'F10g_optuni', 'F25g_optuni', 'F50g_optuni', 'F25g_rand', 'F25l_optuni']
SECTION8 = ['Gc3', 'Gc6', 'Gc10', 'Gc15', 'Gc20', 'Gcinf', 'Gb25', 'Gb50', 'Gb75', 'Hx6', 'Hx10', 'Hold0', 'Gx3', 'Gx6', 'Gx10', 'Gx15', 'Gx20',
            'Gy10', 'Gz10', 'Gm10']
PRESETS = {'all': SECTION7 + SECTION8, 'section7': SECTION7, 'section8': ['C_union', 'A_lib', 'D25_optuni', 'F100g', 'F25g_optuni'] + SECTION8,
           'centred': ['C_union', 'A_lib', 'D25_optuni', 'F100g', 'F25g_optuni', 'Gx6', 'Gx10', 'Gx15', 'Gz10', 'Gm10'],
           'chosen': ['C_union', 'A_lib', 'F100g', 'Gm10']}
KEEP = ('P_dB', 'KF_dB', 'DS_ns', 'ASD_deg', 'ESD_deg', 'ASA_deg', 'ESA_deg', 'n_path', 'n_tap', 'n_cluster', 'n_eff', 'n_tap20', 'Bc_MHz',
        'rank_rx', 'rank_tx', 'corr_rx', 'corr_tx', 'sv_spread_dB', 'cap')
STRIDE = 2      # the curves are stored at every second snapshot
np.seterr(divide='ignore', invalid='ignore')     # pure segments have no power outside their snapshots


def one(job):
    case, gm, seeds, methods, shift, out = job
    mob = 'phase' if gm == '38.901' else 'drift'
    lam = gen.C0 / gen.CASES[case]['fc']
    res = []
    for seed in seeds:
        t0 = time.time()
        o = gen.segments(case, seed, gm, mob, out, tag=f'{case}_{gm}', shift=shift)
        xf.prepare(o)
        pa, pb = xf.pure(o, 'A'), xf.pure(o, 'B')
        cA, cB = xf.curves(o, pa), xf.curves(o, pb)
        sl = slice(pa['r0'], pa['r0'] + pa['n_r'])
        sub = lambda c: {k: c[k][::STRIDE].astype(np.float32) for k in KEEP}
        rec = dict(case=case, gm=gm, mob=mob, seed=seed, r0=pa['r0'], n_r=pa['n_r'], nA=o['nA'], change=o['change'], w=pa['w'], dx=gen.DX,
                   stride=STRIDE, curves={'A': sub(cA), 'B': sub(cB)},
                   fade={'A': xf.fading_stats(cA, sl, gen.DX, lam), 'B': xf.fading_stats(cB, sl, gen.DX, lam)}, ex={}, exu={})
        for m in methods:
            s = xf.schedule(o, m, seed=seed)
            c = xf.curves(o, s)
            rec['ex'][m] = s['ex']
            rec['exu'][m] = s['ex_units'].astype(np.float32)
            rec['curves'][m] = sub(c)
            rec['fade'][m] = xf.fading_stats(c, sl, gen.DX, lam)
        res.append(rec)
        print(case, gm, 'seed', seed, '%.1f s' % (time.time() - t0), flush=True)
        with open(os.path.join(out, f'res_{case}_{gm}.pkl'), 'wb') as f:
            pickle.dump(res, f)
    return case, gm


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', required=True, help='folder for the results (created); the layout CSV files are written there too')
    ap.add_argument('--methods', default='chosen', help='preset (' + ', '.join(PRESETS) + ') or a comma-separated list of schedule names')
    ap.add_argument('--cases', default=','.join(gen.CASES), help='comma-separated cases: ' + ', '.join(gen.CASES))
    ap.add_argument('--gm', default='38.901,quadriga', help='cluster methods: 38.901 (phase mobility), quadriga (drift)')
    ap.add_argument('--seeds', type=int, default=24, help='number of master seeds, 1 ... n')
    ap.add_argument('--shift', type=int, default=0, help='snapshots by which the ramp ends behind the change row (200: centred)')
    ap.add_argument('--jobs', type=int, default=1, help='parallel processes, one per case and method')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    methods = PRESETS[a.methods] if a.methods in PRESETS else a.methods.split(',')
    jobs = [(c, g, list(range(1, a.seeds + 1)), methods, a.shift, a.out) for g in a.gm.split(',') for c in a.cases.split(',')]
    if a.jobs > 1:
        with multiprocessing.Pool(a.jobs) as pool:
            for r in pool.imap_unordered(one, jobs):
                print('done', *r, flush=True)
    else:
        for j in jobs:
            print('done', *one(j), flush=True)
