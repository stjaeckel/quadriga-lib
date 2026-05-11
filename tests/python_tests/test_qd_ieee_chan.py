# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

# Adjusted according to the Python API
from quadriga_lib import channel, arrayant


# ============================================================================
# Helpers
# ============================================================================

def _snap_list(ch, key):
    v = ch[key]
    if not isinstance(v, (list, tuple)):
        raise TypeError(f'ch["{key}"] must be a list/tuple of per-snapshot arrays')
    return v


def _coeff_at(ch, snap=0):
    return np.asarray(_snap_list(ch, "coeff")[snap], dtype=np.complex128)  # (n_rx,n_tx,n_path)


def _delay_at(ch, snap=0):
    return np.asarray(_snap_list(ch, "delay")[snap], dtype=np.float64)     # (n_rx,n_tx,n_path)


def _path_gain_at(ch, snap=0):
    pg = ch["path_gain"]
    if isinstance(pg, (list, tuple)):
        return np.asarray(pg[snap], dtype=np.float64).reshape(-1)          # (n_path,)
    pg = np.asarray(pg, dtype=np.float64)
    if pg.ndim == 2:
        return pg[:, snap]
    return pg.reshape(-1)


def _n_snap(ch):
    return len(_snap_list(ch, "coeff"))


def _n_path(ch, snap=0):
    return _coeff_at(ch, snap).shape[2]


def _total_power(ch, snap=0):
    """Sum of path gains (linear) at a single snapshot."""
    return float(np.sum(_path_gain_at(ch, snap)))


def _ieee_call(ant_ap, ant_sta, model, *,
               CarrierFreq_Hz=5.25e9,
               tap_spacing_s=10e-9,
               n_users=1,
               observation_time=0.0,
               update_rate=1e-3,
               speed_station_kmh=0.0,
               speed_env_kmh=1.2,
               Dist_m=None,
               n_floors=None,
               uplink=False,
               offset_angles=None,
               n_subpath=20,
               Doppler_effect=50.0,
               seed=-1,
               KF_linear=np.nan,
               XPR_NLOS_linear=np.nan,
               SF_std_dB_LOS=np.nan,
               SF_std_dB_NLOS=np.nan,
               dBP_m=np.nan,
               n_walls=None,
               wall_loss=5.0):
    """Wrapper around channel.get_ieee_indoor that supplies the 24 positional
    arguments by name with the documented defaults."""
    if Dist_m is None:
        Dist_m = np.array([4.99])
    if n_floors is None:
        n_floors = np.array([0], dtype=np.uint64)
    if offset_angles is None:
        offset_angles = np.empty((0, 0))
    if n_walls is None:
        n_walls = np.array([0], dtype=np.uint64)

    return channel.get_ieee_indoor(
        ant_ap, ant_sta, model,
        CarrierFreq_Hz, tap_spacing_s, n_users,
        observation_time, update_rate,
        speed_station_kmh, speed_env_kmh,
        Dist_m, n_floors, uplink,
        offset_angles, n_subpath, Doppler_effect, seed,
        KF_linear, XPR_NLOS_linear,
        SF_std_dB_LOS, SF_std_dB_NLOS, dBP_m,
        n_walls, wall_loss)


# ============================================================================
# Tests
# ============================================================================

# Covered tests:
# - K-Factor model for Type A (xpol case)
# - Correctness of TX/RX positions and orientations (TGac default + manual offsets)
# - LOS Steering matrix (Xpol isolation)
# - Floors and distances: path counts, floor indicator, relative path gains
# - Seed consistency
# - Wall penetration loss (default 5 dB, custom 7 dB, per-user vector)
# - TGax floor penetration loss formula (n=2,3,4 vs n=1 reference)
# - Doppler shift on LOS path (moving station)
# - Uplink-downlink reciprocity (time-domain)


class test_case(unittest.TestCase):

    # ------------------------------------------------------------------
    # 1) Type A, single user, cross-polarised antenna
    # ------------------------------------------------------------------
    def test_ieee_chan_a_1user_xpol(self):
        ant = arrayant.generate("xpol", 30.0)
        chan = channel.get_ieee_indoor(ant, ant, "A")

        self.assertEqual(len(chan), 1)
        ch = chan[0]

        coeff_list = _snap_list(ch, "coeff")
        delay_list = _snap_list(ch, "delay")
        self.assertEqual(len(coeff_list), 1)
        self.assertEqual(len(delay_list), 1)

        coeff0 = _coeff_at(ch, 0)  # (n_rx,n_tx,n_path)
        n_rx, n_tx, n_path = coeff0.shape
        self.assertEqual(n_tx, 2)
        self.assertEqual(n_rx, 2)
        self.assertEqual(n_path, 2)

        path_gain0 = _path_gain_at(ch, 0)
        self.assertEqual(path_gain0.shape, (2,))

        # Both paths should have equal power at KF = 0 dB
        self.assertLess(abs(path_gain0[0] - path_gain0[1]), 1e-14)

        # Calculate path gain from coefficients and compare (C++: sum(|H|^2) * 0.5)
        vc = np.abs(coeff0) ** 2  # (2,2,2)
        vv = np.sum(vc.reshape(n_rx * n_tx, n_path), axis=0) * 0.5
        npt.assert_allclose(path_gain0, vv, rtol=0.0, atol=1e-12)

        # Tx should be at the origin, facing east
        tx_pos = np.asarray(ch["tx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        tx_ori = np.asarray(ch["tx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
        npt.assert_allclose(tx_pos, np.array([0.0, 0.0, 0.0]), rtol=0.0, atol=0.0)
        npt.assert_allclose(tx_ori, np.array([0.0, 0.0, 0.0]), rtol=0.0, atol=0.0)

        # Rx should be 4.99 meters east, facing west (yaw ~ pi)
        rx_pos = np.asarray(ch["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        rx_ori = np.asarray(ch["rx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
        npt.assert_allclose(rx_pos, np.array([4.99, 0.0, 0.0]), rtol=0.0, atol=0.0)
        npt.assert_allclose(rx_ori, np.array([0.0, 0.0, np.pi]), rtol=0.0, atol=1e-12)

        # Off-diagonal elements of the LOS steering matrix should be 0 due to perfect Xpol isolation
        H0 = coeff0[:, :, 0]  # LOS slice, (2,2)
        npt.assert_allclose(H0[0, 1], 0.0 + 0.0j, rtol=0.0, atol=0.0)
        npt.assert_allclose(H0[1, 0], 0.0 + 0.0j, rtol=0.0, atol=0.0)

        # Main diagonal elements should have equal power
        p00 = np.abs(H0[0, 0]) ** 2
        p11 = np.abs(H0[1, 1]) ** 2
        npt.assert_allclose(p00, p11, rtol=0.0, atol=0.0)

    # ------------------------------------------------------------------
    # 2) Type A, two users, default TGac offset angles
    # ------------------------------------------------------------------
    def test_ieee_chan_a_2user_default_offsets(self):
        ant = arrayant.generate("omni", 30.0)
        chan = _ieee_call(ant, ant, "A",
                          CarrierFreq_Hz=2.4e9,
                          tap_spacing_s=1e-8,
                          n_users=2)

        self.assertEqual(len(chan), 2)

        # User 0: AoD = -78.0189, AoA = -135.3011 (TGac doc IEEE 802.11-09/0308r12)
        aod_deg = -78.0189
        aoa_deg = -135.3011
        cx = 4.99 * np.cos(np.deg2rad(aod_deg))
        cy = 4.99 * np.sin(np.deg2rad(aod_deg))
        ori_deg = aod_deg - aoa_deg - 180.0

        rx_pos0 = np.asarray(chan[0]["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        rx_ori0 = np.asarray(chan[0]["rx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
        npt.assert_allclose(rx_pos0, np.array([cx, cy, 0.0]), atol=1e-5)
        self.assertLess(abs(ori_deg - np.rad2deg(rx_ori0[2])), 1e-4)

        # User 1: AoD = -142.9707, AoA = 115.1550
        aod_deg = -142.9707
        aoa_deg = 115.1550
        cx = 4.99 * np.cos(np.deg2rad(aod_deg))
        cy = 4.99 * np.sin(np.deg2rad(aod_deg))
        ori_deg = aod_deg - aoa_deg + 180.0

        rx_pos1 = np.asarray(chan[1]["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        rx_ori1 = np.asarray(chan[1]["rx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
        npt.assert_allclose(rx_pos1, np.array([cx, cy, 0.0]), atol=1e-5)
        self.assertLess(abs(ori_deg - np.rad2deg(rx_ori1[2])), 1e-4)

    # ------------------------------------------------------------------
    # 3) Type A, two users, manual offset angles
    # ------------------------------------------------------------------
    def test_ieee_chan_a_2user_manual_offset(self):
        ant = arrayant.generate("omni", 30.0)
        offset_angles = np.full((4, 2), 100.0)

        chan = _ieee_call(ant, ant, "A",
                          CarrierFreq_Hz=2.4e9,
                          tap_spacing_s=1e-8,
                          n_users=2,
                          Dist_m=np.array([1.99]),
                          offset_angles=offset_angles,
                          n_subpath=100,
                          seed=11)

        self.assertEqual(len(chan), 2)

        # All four angle rows set to 100° → AoD = AoA = 100°
        aod_deg = 100.0
        aoa_deg = 100.0
        cx = 1.99 * np.cos(np.deg2rad(aod_deg))
        cy = 1.99 * np.sin(np.deg2rad(aod_deg))
        ori_deg = aod_deg - aoa_deg + 180.0

        for u in range(2):
            rx_pos = np.asarray(chan[u]["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
            rx_ori = np.asarray(chan[u]["rx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
            npt.assert_allclose(rx_pos, np.array([cx, cy, 0.0]), atol=1e-5)
            self.assertLess(abs(ori_deg - np.rad2deg(rx_ori[2])), 1e-4)

    # ------------------------------------------------------------------
    # 4) Type B, three users, mixed distances and floors
    # ------------------------------------------------------------------
    def test_ieee_chan_b_3user_floors_distances(self):
        ant = arrayant.generate("omni", 30.0)
        chan = _ieee_call(ant, ant, "B",
                          CarrierFreq_Hz=2.4e9,
                          tap_spacing_s=10e-9,
                          n_users=3,
                          Dist_m=np.array([4.0, 8.0, 2.0]),
                          n_floors=np.array([0, 0, 1], dtype=np.uint64))

        self.assertEqual(len(chan), 3)

        # User 0: dist 4 < dBP=5, no floor → LOS exists, 13 paths
        self.assertEqual(_n_path(chan[0]), 13)
        # User 1: dist 8 > dBP=5 → NLOS only, 12 paths
        self.assertEqual(_n_path(chan[1]), 12)
        # User 2: n_floors=1 → NLOS only, 12 paths
        self.assertEqual(_n_path(chan[2]), 12)

        # Floor indicator on rx_position[2]
        rx_pos0 = np.asarray(chan[0]["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        rx_pos1 = np.asarray(chan[1]["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        rx_pos2 = np.asarray(chan[2]["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        self.assertEqual(rx_pos0[2], 0.0)
        self.assertEqual(rx_pos1[2], 0.0)
        self.assertEqual(rx_pos2[2], 3.0)

        # Relative path gains for user 0 (KF=0 dB → first two paths equal)
        p = _path_gain_at(chan[0], 0)
        p = p / (2.0 * p[0])
        p_dB = 10.0 * np.log10(p)
        expected = -np.array([3.01, 3.01, 5.40, 10.8, 3.20, 16.2, 6.30,
                              21.7, 9.40, 12.5, 15.6, 18.7, 21.8])
        npt.assert_allclose(p_dB, expected, atol=0.01)

        # Relative path gains for user 1 (NLOS only) — first 4 entries
        p = _path_gain_at(chan[1], 0)
        p = p / p[0]
        p_dB = 10.0 * np.log10(p)
        npt.assert_allclose(p_dB[:4], -np.array([0.00, 5.40, 10.8, 3.20]), atol=0.01)

        # Relative path gains for user 2 (NLOS only via floor) — same model B NLOS pattern
        p = _path_gain_at(chan[2], 0)
        p = p / p[0]
        p_dB = 10.0 * np.log10(p)
        npt.assert_allclose(p_dB[:4], -np.array([0.00, 5.40, 10.8, 3.20]), atol=0.01)

        # LOS steering matrix of user 0 should have unit magnitude after dividing
        # by the linear LOS gain (n_tx = n_rx = 1 here so it's a 1x1 phasor)
        H_los = _coeff_at(chan[0], 0)[:, :, 0]
        A = np.abs(H_los) / np.sqrt(_path_gain_at(chan[0], 0)[0])
        npt.assert_allclose(A, np.ones_like(A), atol=1e-12)

        # Spot-check delays for user 0 (column 0 ≈ 0 ns, col 2 ≈ 10 ns, …)
        d_ns = _delay_at(chan[0], 0) * 1e9  # (n_rx,n_tx,n_path)
        # collapse across rx,tx — for omni they're all the same
        d_per_path_ns = d_ns.reshape(-1, d_ns.shape[2]).mean(axis=0)
        expected_ns = np.array([0, 0, 10, 20, 20, 30, 30, 40, 40, 50, 60, 70, 80])
        npt.assert_allclose(d_per_path_ns, expected_ns, atol=1.5)

    # ------------------------------------------------------------------
    # 5) Seed consistency: same seed → identical channel
    # ------------------------------------------------------------------
    def test_ieee_chan_seed_consistency(self):
        ant = arrayant.generate("omni", 30.0)

        kw = dict(CarrierFreq_Hz=2.4e9,
                  tap_spacing_s=10e-9,
                  Dist_m=np.array([3.0]),
                  seed=1234)

        chan_a = _ieee_call(ant, ant, "B", **kw)
        chan_b = _ieee_call(ant, ant, "B", **kw)

        npt.assert_array_equal(_coeff_at(chan_a[0], 0), _coeff_at(chan_b[0], 0))
        npt.assert_array_equal(_delay_at(chan_a[0], 0), _delay_at(chan_b[0], 0))
        npt.assert_array_equal(_path_gain_at(chan_a[0], 0), _path_gain_at(chan_b[0], 0))

    # ------------------------------------------------------------------
    # 6) Wall penetration loss
    # ------------------------------------------------------------------
    def test_ieee_chan_b_wall_loss(self):
        """Wall loss is a deterministic additive term in the path loss. With a
        fixed seed and otherwise identical inputs (Dist, n_floors, model,
        frequency), adding walls neither changes the path structure nor the SF
        realisation, so the total path power in dB must differ from the
        reference by exactly -n_walls * wall_loss for every user."""
        ant = arrayant.generate("omni", 30.0)
        seed = 42

        common = dict(CarrierFreq_Hz=5.25e9,
                      tap_spacing_s=10e-9,
                      Dist_m=np.array([3.0]),
                      n_floors=np.array([0], dtype=np.uint64),
                      seed=seed)

        # Reference: no walls
        chan_ref = _ieee_call(ant, ant, "B",
                              n_walls=np.array([0], dtype=np.uint64),
                              wall_loss=5.0,
                              **common)

        # 3 walls @ default 5 dB → -15 dB
        chan_w3 = _ieee_call(ant, ant, "B",
                             n_walls=np.array([3], dtype=np.uint64),
                             wall_loss=5.0,
                             **common)

        self.assertEqual(_n_path(chan_ref[0]), _n_path(chan_w3[0]))
        P_ref = _total_power(chan_ref[0])
        P_w3 = _total_power(chan_w3[0])
        self.assertLess(abs(10.0 * np.log10(P_w3 / P_ref) + 15.0), 1e-9)

        # 2 walls @ custom 7 dB → -14 dB
        chan_w2_7 = _ieee_call(ant, ant, "B",
                               n_walls=np.array([2], dtype=np.uint64),
                               wall_loss=7.0,
                               **common)

        self.assertEqual(_n_path(chan_ref[0]), _n_path(chan_w2_7[0]))
        P_w2_7 = _total_power(chan_w2_7[0])
        self.assertLess(abs(10.0 * np.log10(P_w2_7 / P_ref) + 14.0), 1e-9)

        # Per-user n_walls vector with same seed — each user's ratio is
        # determined purely by its own wall count.
        common3 = dict(CarrierFreq_Hz=5.25e9,
                       tap_spacing_s=10e-9,
                       n_users=3,
                       Dist_m=np.array([3.0, 3.0, 3.0]),
                       n_floors=np.array([0, 0, 0], dtype=np.uint64),
                       seed=seed)

        chan3_ref = _ieee_call(ant, ant, "B",
                               n_walls=np.array([0, 0, 0], dtype=np.uint64),
                               wall_loss=5.0,
                               **common3)

        chan3_w = _ieee_call(ant, ant, "B",
                             n_walls=np.array([1, 2, 3], dtype=np.uint64),
                             wall_loss=5.0,
                             **common3)

        self.assertEqual(len(chan3_ref), 3)
        self.assertEqual(len(chan3_w), 3)
        for u in range(3):
            self.assertEqual(_n_path(chan3_ref[u]), _n_path(chan3_w[u]))
            pr = _total_power(chan3_ref[u])
            pw = _total_power(chan3_w[u])
            expected = -float(u + 1) * 5.0  # 1, 2, 3 walls
            self.assertLess(abs(10.0 * np.log10(pw / pr) - expected), 1e-9)

    # ------------------------------------------------------------------
    # 7) TGax floor penetration loss formula
    # ------------------------------------------------------------------
    def test_ieee_chan_b_tgax_floor_formula(self):
        """For TGax (CarrierFreq >= 1 GHz) the per-floor penetration loss is
            FL(n) = 18.3 * n^((n+2)/(n+1) - 0.46)  [dB], for n_floors >= 1
        (See IEEE 802.11-14/0882r4, Section 3.1.1.) Comparing two runs that
        both have n_floors > 0 ensures both lose the LOS path, so the path
        structure and SF realisation are identical at a fixed seed; the total
        power ratio is then determined purely by the floor-loss term."""
        ant = arrayant.generate("omni", 30.0)
        seed = 7

        def FL(n):
            return 18.3 * (n ** ((n + 2.0) / (n + 1.0) - 0.46))

        def run(nf):
            return _ieee_call(ant, ant, "B",
                              CarrierFreq_Hz=5.25e9,
                              tap_spacing_s=10e-9,
                              Dist_m=np.array([3.0]),
                              n_floors=np.array([nf], dtype=np.uint64),
                              seed=seed)

        chan1 = run(1)
        P1 = _total_power(chan1[0])

        for nf in (2, 3, 4):
            chan_n = run(nf)
            self.assertEqual(_n_path(chan_n[0]), _n_path(chan1[0]))
            Pn = _total_power(chan_n[0])
            expected = -(FL(float(nf)) - FL(1.0))
            self.assertLess(abs(10.0 * np.log10(Pn / P1) - expected), 1e-9)

    # ------------------------------------------------------------------
    # 8) Doppler shift on the LOS path (moving station)
    # ------------------------------------------------------------------
    def test_ieee_chan_a_doppler_los(self):
        """With a static environment and a station moving along the LOS axis
        the LOS path coefficient evolves as exp(j 2π f_D t), so an FFT of
        coeff[s][0,0,0] over the snapshots peaks at f_D = v * fc / c."""
        fGHz = 2.4
        update_rate_s = 0.0025
        observation_time_s = 10.0
        speed_station_kmh = 50.0

        ant = arrayant.generate("omni", 30.0)

        chan = _ieee_call(ant, ant, "A",
                          CarrierFreq_Hz=fGHz * 1e9,
                          tap_spacing_s=10e-9,
                          observation_time=observation_time_s,
                          update_rate=update_rate_s,
                          speed_station_kmh=speed_station_kmh,
                          speed_env_kmh=0.0,
                          seed=1234)

        ch = chan[0]
        n_snap = _n_snap(ch)
        self.assertGreater(n_snap, 100)

        # LOS path coefficient over all snapshots (n_tx = n_rx = 1 for omni)
        h_los = np.array([_coeff_at(ch, s)[0, 0, 0] for s in range(n_snap)],
                         dtype=np.complex128)

        H = np.fft.fftshift(np.fft.fft(h_los))
        f = np.fft.fftshift(np.fft.fftfreq(n_snap, d=update_rate_s))

        f_peak = f[int(np.argmax(np.abs(H)))]

        v = speed_station_kmh / 3.6
        fD_expected = v * (fGHz * 1e9) / 3.0e8

        self.assertLess(abs(abs(f_peak) - fD_expected), 5.0)

    # ------------------------------------------------------------------
    # 9) Uplink-downlink reciprocity (time domain)
    # ------------------------------------------------------------------
    def test_ieee_chan_reciprocity(self):
        """For zero observation time and a fixed seed, the uplink channel must
        equal the conjugate transpose of the downlink channel per path, with
        identical delays."""
        ant = arrayant.generate("xpol", 30.0)  # 2 elements at each end

        common = dict(CarrierFreq_Hz=2.4e9,
                      tap_spacing_s=5e-9,
                      n_users=2,
                      Doppler_effect=40.0,
                      seed=1234)

        chan_dl = _ieee_call(ant, ant, "F", uplink=False, **common)
        chan_ul = _ieee_call(ant, ant, "F", uplink=True,  **common)

        self.assertEqual(len(chan_dl), 2)
        self.assertEqual(len(chan_ul), 2)

        # User 0 only — same property holds per user
        coeff_dl = _coeff_at(chan_dl[0], 0)  # (n_rx_DL, n_tx_DL, n_path)
        coeff_ul = _coeff_at(chan_ul[0], 0)  # (n_rx_UL, n_tx_UL, n_path)
        delay_dl = _delay_at(chan_dl[0], 0)
        delay_ul = _delay_at(chan_ul[0], 0)

        self.assertEqual(coeff_dl.shape[2], coeff_ul.shape[2])
        self.assertEqual(coeff_dl.shape[0], coeff_ul.shape[1])  # n_rx_DL = n_tx_UL
        self.assertEqual(coeff_dl.shape[1], coeff_ul.shape[0])  # n_tx_DL = n_rx_UL

        max_err = 0.0
        for k in range(coeff_dl.shape[2]):
            H_dl = coeff_dl[:, :, k]
            H_ul = coeff_ul[:, :, k]
            max_err = max(max_err, float(np.max(np.abs(H_ul - H_dl.conj().T))))

            d_dl = delay_dl[:, :, k]
            d_ul = delay_ul[:, :, k]
            max_err = max(max_err, float(np.max(np.abs(d_ul - d_dl.T))))

        self.assertLess(max_err, 1e-10)


if __name__ == '__main__':
    unittest.main()