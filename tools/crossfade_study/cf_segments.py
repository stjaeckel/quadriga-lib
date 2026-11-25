"""Channels of two consecutive segments of one link over the same snapshots, from the layout functions of quadriga-lib."""
import os
import numpy as np
import quadriga_lib as qd

C0 = 299792458.0
CASES = {
    'UMa_LOS_NLOS': dict(tx='UMa', a='UMa_LOS', b='UMa_NLOS', hbs=25.0, p0=(120.0, 80.0), hd=20.0, fc=3.5e9),
    'UMa_NLOS_O2I': dict(tx='UMa', a='UMa_NLOS', b='UMa_NLOS_O2I', hbs=25.0, p0=(120.0, 80.0), hd=20.0, fc=3.5e9),
    'UMi_NLOS_LOS': dict(tx='UMi', a='UMi_NLOS', b='UMi_LOS', hbs=10.0, p0=(50.0, 30.0), hd=110.0, fc=3.5e9),
    'InF_LOS_NLOS': dict(tx='InF_DH', a='InF_LOS', b='InF_NLOS_DH', hbs=8.0, p0=(25.0, 15.0), hd=200.0, fc=3.5e9),
}
LA, LB, DX, V = 20.0, 10.0, 0.025, 1.0   # old segment, new segment [m], snapshot spacing [m], speed [m/s]


def _write(case, variant, work_dir, tag):
    c = CASES[case]
    hd = np.deg2rad(c['hd'])
    u = np.array([np.cos(hd), np.sin(hd)])
    inf = case.startswith('InF')
    with open(os.path.join(work_dir, f'tx_{tag}.csv'), 'w') as f:
        f.write('name,x,y,z,scenario,center_freq' + (',d_clutter,r_clutter,h_clutter' if inf else '') + '\n')
        f.write(f'BS,0,0,{c["hbs"]},{c["tx"]},{c["fc"]}' + (',10,0.5,6' if inf else '') + '\n')
    rows = ['name,x,y,z,time,scenario']

    def row(s, t, scen):
        rows.append(f'UT,{c["p0"][0] + s * u[0]:.6f},{c["p0"][1] + s * u[1]:.6f},1.5,{t:.6f},{scen}')
    t0 = 1.0
    if variant == 'AB':     # the track with both segments
        row(0, t0, c['a']); row(LA, t0 + LA / V, c['b']); row(LA + LB, t0 + (LA + LB) / V, c['b'])
    elif variant == 'A':    # the old scenario on all rows
        row(0, t0, c['a']); row(LA, t0 + LA / V, c['a']); row(LA + LB, t0 + (LA + LB) / V, c['a'])
    else:                   # the new scenario on all rows; an extra first row at the change row, where its rays are drawn
        row(LA, 0.0, c['b']); row(0, t0, c['b']); row(LA, t0 + LA / V, c['b']); row(LA + LB, t0 + (LA + LB) / V, c['b'])
    with open(os.path.join(work_dir, f'rx_{tag}.csv'), 'w') as f:
        f.write('\n'.join(rows) + '\n')


def times():
    s = np.arange(int(round((LA + LB) / DX))) * DX       # snapshot positions along the track [m]
    return 1.0 + s / V, s


def layout(case, variant, seed, method, work_dir, tag='w'):
    _write(case, variant, work_dir, tag)
    L, _, _ = qd.layout.csv_read(os.path.join(work_dir, f'tx_{tag}.csv'), os.path.join(work_dir, f'rx_{tag}.csv'))
    L = qd.layout.set_antenna(L, 'ula', {'panel': [1, 4]}, rx=False)
    L = qd.layout.set_antenna(L, 'ula', {'panel': [1, 4]}, rx=True)
    return qd.layout.init_link_states(L, method=method, seed=seed)


def _stack(ch, rays):
    kept = (rays['path_type'] < 2) | (rays['power'][:, 0] > 0)
    tap = rays['tap'][kept].astype(np.int64)
    co = ch['coeff']
    assert co.shape[2] == tap.size, (co.shape, tap.size)
    dl = ch['delay']
    dl = dl.mean(axis=(0, 1)) if dl.shape[0] > 1 or dl.shape[1] > 1 else dl[0, 0]
    return dict(coeff=co.astype(np.complex64), delay=np.ascontiguousarray(dl), pg=ch['path_gain'], ang=ch['path_angles'], tap=tap,
                ptype=rays['path_type'][kept].astype(np.int64), rx_pos=ch['rx_pos'])


def segments(case, seed, method, mobility, work_dir, tag='w', shift=0, overlap=0.5, with_lib=False):
    """'rays' channels of the old segment (A) and the new segment (B) at the same snapshots.

    The ramp has floor(overlap * n + 0.5) snapshots, n = snapshots of the old segment, and ends 'shift' snapshots behind the
    change row (shift = 0: at the change row as in the present library; shift = half the ramp: centred).
    Returns a dict: A, B (coeff [n_rx, n_tx, n_path, n_snap], delay, pg, ang, tap, ptype), nA (last ramp snapshot + 1),
    n_r (ramp snapshots), change (snapshot of the change row), s (positions)."""
    t, s = times()
    out = {}
    for v in ('A', 'B'):
        L = layout(case, v, seed, method, work_dir, tag)
        rays = qd.layout.generate_rays(L, scatterers=(mobility == 'drift'))
        ch, _ = qd.layout.get_channels_from_rays(L, rays, times=t, output='rays', mobility=mobility, stack=True)
        out[v] = _stack(ch[0], rays[0][0])
    change = int(np.sum(s < LA - 1e-9))
    nA = change + int(shift)
    for k in ('coeff', 'delay', 'pg', 'ang'):
        out['A'][k] = out['A'][k][..., :nA]
    out.update(s=s, nA=nA, change=change, n_r=int(np.floor(overlap * change + 0.5)))
    if with_lib:            # the channel of the library for the track with both segments (ramp ending at the change row)
        L = layout(case, 'AB', seed, method, work_dir, tag)
        rays = qd.layout.generate_rays(L, scatterers=(mobility == 'drift'))
        lr, _ = qd.layout.get_channels_from_rays(L, rays, times=t, output='rays', mobility=mobility, stack=False, overlap=overlap)
        lt, _ = qd.layout.get_channels_from_rays(L, rays, times=t, output='taps', mobility=mobility, stack=True, overlap=overlap)
        out['lib_rays'], out['lib_taps'] = lr[0], lt[0]
    return out
