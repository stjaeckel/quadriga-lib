# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import channel

# Test data files (mirror the C++ tests)
FN_V4 = os.path.join(current_dir, '../data/test.qrt')
FN_V5 = os.path.join(current_dir, '../data/test_v5.qrt')


class test_case(unittest.TestCase):

    # ------------------------------------------------------------------
    # qrt_file_parse
    # ------------------------------------------------------------------
    def test_parse_v4(self):
        (no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version,
         center_freq, cir_pos, cir_orientation, orig_pos, orig_orientation) = \
            channel.qrt_file_parse(FN_V4)

        self.assertEqual(no_orig, 3)
        self.assertEqual(no_dest, 2)
        self.assertEqual(no_cir, 7)
        self.assertEqual(no_freq, 1)
        self.assertEqual(version, 4)

        self.assertEqual(len(cir_offset), no_dest)
        self.assertEqual(cir_offset[0], 0)
        self.assertEqual(cir_offset[1], 1)

        self.assertEqual(orig_names, ["TX1", "TX2", "TX3"])
        self.assertEqual(dest_names, ["RX1", "RX2"])

        # qrt_file_parse returns frequencies as stored in the file (GHz for v4/v5)
        self.assertEqual(len(center_freq), no_freq)
        npt.assert_allclose(center_freq[0], 3.75, atol=1e-6, rtol=0)

        self.assertEqual(cir_pos.shape, (no_cir, 3))
        self.assertEqual(cir_orientation.shape, (no_cir, 3))
        npt.assert_allclose(cir_pos[0, :], [-8.83498, 57.1893, 1.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(cir_pos[1, :], [-5.86144, 53.8124, 1.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(cir_orientation[1, :], [0.0, 0.0, 1.2753], atol=1.5e-4, rtol=0)

        self.assertEqual(orig_pos.shape, (no_orig, 3))
        self.assertEqual(orig_orientation.shape, (no_orig, 3))
        npt.assert_allclose(orig_pos[0, :], [-12.9607, 59.6906, 2.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(orig_pos[1, :], [-2.67888, 60.257, 2.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(orig_orientation[0, :], [0.0, 0.0, 0.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(orig_orientation[1, :], [0.0, 0.0, np.pi], atol=1.5e-4, rtol=0)

    def test_parse_v5(self):
        (no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version,
         center_freq, cir_pos, cir_orientation, orig_pos, orig_orientation) = \
            channel.qrt_file_parse(FN_V5)

        self.assertEqual(no_orig, 1)
        self.assertEqual(no_dest, 2)
        self.assertEqual(no_cir, 2)
        self.assertEqual(no_freq, 2)
        self.assertEqual(version, 5)

        self.assertEqual(len(cir_offset), no_dest)
        self.assertEqual(cir_offset[0], 0)
        self.assertEqual(cir_offset[1], 1)

        self.assertEqual(len(center_freq), no_freq)
        npt.assert_allclose(center_freq[0], 1.0, atol=1e-6, rtol=0)
        npt.assert_allclose(center_freq[1], 1.5, atol=1e-6, rtol=0)

        self.assertEqual(cir_pos.shape, (no_cir, 3))
        self.assertEqual(cir_orientation.shape, (no_cir, 3))
        self.assertEqual(orig_pos.shape, (no_orig, 3))
        self.assertEqual(orig_orientation.shape, (no_orig, 3))

    # ------------------------------------------------------------------
    # qrt_file_read - single snapshot, downlink / uplink
    # ------------------------------------------------------------------
    def test_read_v4(self):
        # --- downlink, first link: i_cir=0, i_orig=0 ---
        (center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation,
         fbs_pos, lbs_pos, path_gain, path_length, M,
         aod, eod, aoa, eoa, path_coord, no_int, coord) = \
            channel.qrt_file_read(FN_V4, [0], 0, True, 1)

        # qrt_file_read returns center_freq in Hz
        npt.assert_allclose(center_freq[0], 3.75e9, atol=1e-4, rtol=0)

        # Scalar-per-snapshot outputs are (3, n_out) -> here (3, 1)
        self.assertEqual(tx_pos.shape, (3, 1))
        npt.assert_allclose(tx_pos[:, 0], [-12.9607, 59.6906, 2.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(tx_orientation[:, 0], [0.0, 0.0, 0.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(rx_pos[:, 0], [-8.83498, 57.1893, 1.0], atol=1.5e-4, rtol=0)

        # Variable-per-snapshot outputs are lists of length n_out
        self.assertEqual(len(fbs_pos), 1)
        self.assertEqual(len(no_int), 1)
        self.assertEqual(len(coord), 1)
        self.assertIsInstance(path_coord[0], list)

        # --- downlink, second link: i_cir=1, i_orig=1 ---
        (center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation,
         fbs_dl, lbs_dl, path_gain, path_length, M,
         aod, eod, aoa_dl, eoa_dl, path_coord, no_int, coord) = \
            channel.qrt_file_read(FN_V4, [1], 1, True, 1)

        npt.assert_allclose(tx_pos[:, 0], [-2.67888, 60.257, 2.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(tx_orientation[:, 0], [0.0, 0.0, np.pi], atol=1.5e-4, rtol=0)
        npt.assert_allclose(rx_pos[:, 0], [-5.86144, 53.8124, 1.0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(rx_orientation[:, 0], [0.0, 0.0, 1.2753], atol=1.5e-4, rtol=0)

        # --- uplink: i_cir=1, i_orig=1, downlink=False ---
        (center_freq, tx_pos_ul, tx_ori_ul, rx_pos_ul, rx_ori_ul,
         fbs_ul, lbs_ul, path_gain, path_length, M,
         aod_ul, eod_ul, aoa, eoa, path_coord, no_int, coord) = \
            channel.qrt_file_read(FN_V4, [1], 1, False, 1)

        # TX / RX swap between downlink and uplink
        npt.assert_allclose(tx_pos_ul[:, 0], rx_pos[:, 0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(tx_ori_ul[:, 0], rx_orientation[:, 0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(rx_pos_ul[:, 0], tx_pos[:, 0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(rx_ori_ul[:, 0], tx_orientation[:, 0], atol=1.5e-4, rtol=0)

        # FBS / LBS swap
        npt.assert_allclose(fbs_ul[0], lbs_dl[0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(lbs_ul[0], fbs_dl[0], atol=1.5e-4, rtol=0)

        # AoD / EoD (uplink) match AoA / EoA (downlink)
        npt.assert_allclose(aod_ul[0], aoa_dl[0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(eod_ul[0], eoa_dl[0], atol=1.5e-4, rtol=0)

    # ------------------------------------------------------------------
    # qrt_file_read - v5 file, normalize_M switch
    # ------------------------------------------------------------------
    def test_read_v5(self):
        # normalize_M = 0: M and path_gain as stored in the QRT file
        (center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation,
         fbs_pos, lbs_pos, path_gain, path_length, M,
         aod, eod, aoa, eoa, path_coord, no_int, coord) = \
            channel.qrt_file_read(FN_V5, [1], 0, True, 0)

        self.assertEqual(len(center_freq), 2)
        npt.assert_allclose(center_freq[0], 1.0e9, atol=1e-4, rtol=0)
        npt.assert_allclose(center_freq[1], 1.5e9, atol=1e-4, rtol=0)

        M0 = M[0]
        self.assertEqual(M0.shape[0], 8)        # 8 interleaved real/imag rows
        self.assertEqual(M0.shape[2], 2)        # 2 frequencies
        self.assertEqual(path_gain[0].shape[1], 2)

        npt.assert_allclose(M0[:, 0, 0], [0.1131, 0, 0, 0, 0, 0, -0.1131, 0], atol=1.5e-4, rtol=0)
        npt.assert_allclose(M0[:, 0, 1], [0.0866, 0, 0, 0, 0, 0, -0.0866, 0], atol=1.5e-4, rtol=0)

        # normalize_M = 1: M columns normalized to max power 1
        res = channel.qrt_file_read(FN_V5, [1], 0, True, 1)
        M0n = res[9][0]   # output index 9 = M (list), [0] = first snapshot
        npt.assert_allclose(M0n[:, 0, 1], [1, 0, 0, 0, 0, 0, -1, 0], atol=1.5e-4, rtol=0)

    # ------------------------------------------------------------------
    # qrt_file_read - multiple snapshots in one call (shared read cache)
    # ------------------------------------------------------------------
    def test_read_multi_cir(self):
        (center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation,
         fbs_pos, lbs_pos, path_gain, path_length, M,
         aod, eod, aoa, eoa, path_coord, no_int, coord) = \
            channel.qrt_file_read(FN_V4, [0, 1, 2], 0, True, 1)

        # Scalar-per-snapshot outputs: (3, n_out) matrices
        self.assertEqual(tx_pos.shape, (3, 3))
        self.assertEqual(tx_orientation.shape, (3, 3))
        self.assertEqual(rx_pos.shape, (3, 3))
        self.assertEqual(rx_orientation.shape, (3, 3))

        # Variable-per-snapshot outputs: lists of length n_out
        for lst in (fbs_pos, lbs_pos, path_gain, path_length, M,
                    aod, eod, aoa, eoa, path_coord, no_int, coord):
            self.assertEqual(len(lst), 3)

        # path_coord is a list (per snapshot) of lists (per path)
        self.assertIsInstance(path_coord[0], list)
        self.assertEqual(len(path_coord[0]), len(aod[0]))

        # coord entries are (3, sum(no_int)) per snapshot
        for i in range(3):
            self.assertEqual(coord[i].shape[0], 3)
            self.assertEqual(coord[i].shape[1], int(np.sum(no_int[i])))

        # Cross-check the cached multi-read against individual single-snapshot reads
        r0 = channel.qrt_file_read(FN_V4, [0], 0, True, 1)
        r1 = channel.qrt_file_read(FN_V4, [1], 0, True, 1)

        npt.assert_allclose(tx_pos[:, 0], r0[1][:, 0], atol=1e-6, rtol=0)
        npt.assert_allclose(tx_pos[:, 1], r1[1][:, 0], atol=1e-6, rtol=0)
        npt.assert_allclose(rx_pos[:, 0], r0[3][:, 0], atol=1e-6, rtol=0)
        npt.assert_allclose(rx_pos[:, 1], r1[3][:, 0], atol=1e-6, rtol=0)

        npt.assert_allclose(fbs_pos[0], r0[5][0], atol=1e-6, rtol=0)
        npt.assert_allclose(fbs_pos[1], r1[5][0], atol=1e-6, rtol=0)
        npt.assert_allclose(path_gain[0], r0[7][0], atol=1e-6, rtol=0)
        npt.assert_allclose(path_gain[1], r1[7][0], atol=1e-6, rtol=0)
        npt.assert_allclose(path_length[0], r0[8][0], atol=1e-6, rtol=0)
        npt.assert_allclose(path_length[1], r1[8][0], atol=1e-6, rtol=0)
        npt.assert_allclose(M[0], r0[9][0], atol=1e-6, rtol=0)
        npt.assert_allclose(M[1], r1[9][0], atol=1e-6, rtol=0)
        npt.assert_allclose(aod[0], r0[10][0], atol=1e-6, rtol=0)
        npt.assert_allclose(aod[1], r1[10][0], atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # qrt_file_read - default arguments
    # ------------------------------------------------------------------
    def test_read_defaults(self):
        # Default i_cir (omitted) reads ALL snapshots in the file (no_cir = 7)
        res = channel.qrt_file_read(FN_V4)
        tx_pos = res[1]
        self.assertEqual(tx_pos.shape, (3, 7))

        # Explicit empty list also reads all snapshots
        res_e = channel.qrt_file_read(FN_V4, [])
        self.assertEqual(res_e[1].shape, (3, 7))

        # First column matches an explicit single-snapshot read
        r0 = channel.qrt_file_read(FN_V4, [0])
        npt.assert_allclose(tx_pos[:, 0], r0[1][:, 0], atol=1e-6, rtol=0)

    # ------------------------------------------------------------------
    # qrt_file_read - error handling
    # ------------------------------------------------------------------
    def test_read_errors(self):
        # CIR index out of range -> IndexError from the wrapper bounds check
        with self.assertRaisesRegex(IndexError, "CIR index exceeds"):
            channel.qrt_file_read(FN_V4, [999], 0, True, 1)


if __name__ == '__main__':
    unittest.main()