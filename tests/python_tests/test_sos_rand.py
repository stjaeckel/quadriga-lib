# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.rand.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_rand.cpp).

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


class test_case(unittest.TestCase):

    def setUp(self):
        self.dd = np.array([5.0, 20.0, 0.0])
        self.pos = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 4.0, 1.5], [100.0, -50.0, 2.0]])
        self.pos_b = self.pos[:, [1, 0, 2]] + 1.0

        # The documented composition: preset, phases, evaluation; here for generator 1 (20 m). A second device
        # on its own map draws its phases from a second stream with the seed + 2^62.
        self.freq, self.acf, self.dist = quadriga_lib.sos.preset("Comb256", 20.0)
        self.phase = quadriga_lib.sos.init_phases(256, 6, 7)

    def oracle(self, col, freq=None, **kwargs):
        freq = self.freq if freq is None else freq
        val = quadriga_lib.sos.val(freq, self.phase[:, col], self.pos, uniform=True, **kwargs)
        return val[:, 0]

    def test_basic_call(self):
        val = quadriga_lib.sos.rand(self.dd, self.pos, seed=7)
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (4, 3))  # (n_pos, n_gen)
        self.assertEqual(val.dtype, np.float64)
        self.assertTrue(np.all(val > 0.0) and np.all(val < 1.0))

        # Generator 1 against the composition; a decorrelation distance of 0 is replaced by 0.1 m
        npt.assert_allclose(val[:, 1], self.oracle(1), atol=1e-12, rtol=0)
        freq0, _, _ = quadriga_lib.sos.preset("Comb256", 0.1)
        npt.assert_allclose(val[:, 2], self.oracle(2, freq0), atol=1e-12, rtol=0)

    def test_keyword_arguments(self):
        dd_b = np.array([5.0, 8.0, 0.0])
        ref = quadriga_lib.sos.rand(self.dd, self.pos, self.pos_b, dd_b, "Exp128", 7)
        val = quadriga_lib.sos.rand(dist_decorr=self.dd, pos=self.pos, pos_b=self.pos_b, dist_decorr_b=dd_b,
                                    preset="Exp128", seed=7)
        npt.assert_array_equal(val, ref)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.rand(self.dd, self.pos, seed=7)

        # None and empty arrays are the same as omitted; "Comb256" is the default preset
        val = quadriga_lib.sos.rand(self.dd, self.pos, None, None, "Comb256", 7)
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.rand(self.dd, self.pos, np.zeros((0, 3)), [], seed=7)
        npt.assert_array_equal(val, ref)

        # Another preset
        val = quadriga_lib.sos.rand(self.dd, self.pos, preset="Exp128", seed=7)
        freq_e, _, _ = quadriga_lib.sos.preset("Exp128", 20.0)
        phase_e = quadriga_lib.sos.init_phases(128, 6, 7)
        s = quadriga_lib.sos.val(freq_e, phase_e[:, 1], self.pos, uniform=True)
        npt.assert_allclose(val[:, 1], s[:, 0], atol=1e-12, rtol=0)

        # seed = -1 (default): a new realisation on each call
        r1 = quadriga_lib.sos.rand(self.dd, self.pos)
        r2 = quadriga_lib.sos.rand(self.dd, self.pos)
        self.assertEqual(r1.shape, (4, 3))
        self.assertGreater(np.max(np.abs(r1 - r2)), 0.001)

        # Two devices on the same map
        val4 = quadriga_lib.sos.rand(self.dd, self.pos, self.pos_b, seed=7)
        npt.assert_allclose(val4[:, 1], self.oracle(1, pos_b=self.pos_b, acf=self.acf, dist=self.dist),
                            atol=1e-12, rtol=0)

        # Equal decorrelation distances of the second device: still the same map
        val5 = quadriga_lib.sos.rand(self.dd, self.pos, self.pos_b, self.dd, seed=7)
        npt.assert_array_equal(val5, val4)

        # Another distance for generator 1: two maps, phases of the second device from the second stream
        phase_b = quadriga_lib.sos.init_phases(256, 3, 7 + 2**62)
        val6 = quadriga_lib.sos.rand(self.dd, self.pos, self.pos_b, [5.0, 8.0, 0.0], seed=7)
        npt.assert_allclose(val6[:, 1], self.oracle(1, pos_b=self.pos_b, phase_b=phase_b[:, 1],
                                                    freq_scale_b=20.0 / 8.0), atol=1e-12, rtol=0)
        npt.assert_array_equal(val6[:, [0, 2]], val4[:, [0, 2]])

    def test_array_layout_and_dtype(self):
        ref = quadriga_lib.sos.rand(self.dd, self.pos, self.pos_b, seed=7)

        variants = [
            lambda a: np.ascontiguousarray(a),                            # C order
            lambda a: np.asfortranarray(a),                               # Fortran order
            lambda a: np.repeat(a, 2, axis=0)[::2],                       # non-contiguous view
            lambda a: a.tolist(),                                         # plain lists
        ]
        for fn in variants:
            val = quadriga_lib.sos.rand(fn(self.dd), fn(self.pos), fn(self.pos_b), seed=7)
            self.assertEqual(val.dtype, np.float64)
            npt.assert_array_equal(val, ref)

        # Integer decorrelation distances and numpy scalars; float32 positions are rounded
        val = quadriga_lib.sos.rand(self.dd.astype(np.int32), self.pos, self.pos_b, seed=np.int32(7))
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.rand(self.dd, self.pos.astype(np.float32), self.pos_b.astype(np.float32), seed=7)
        self.assertEqual(val.dtype, np.float64)
        npt.assert_allclose(val, ref, atol=1e-4, rtol=0)

    def test_inputs_not_modified(self):
        dd_b = np.array([5.0, 8.0, 0.0])
        args = [self.dd, self.pos, self.pos_b, dd_b]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.rand(self.dd, self.pos, self.pos_b, dd_b, seed=7)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.rand(self.dd, self.pos, preset="NoSuchPreset")
        self.assertIn("preset", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.rand(self.dd, np.zeros((4, 4)))
        self.assertIn("pos", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.rand(self.dd, self.pos, seed=-2)
        self.assertIn("seed", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.rand(self.dd)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.rand(self.dd, self.pos, no_such_argument=1)

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.dd, self.pos]
        quadriga_lib.sos.rand(*args)
        for i in range(2):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.rand(*bad)

    def test_one_dimensional_pos(self):
        # A 1-D array is a column: three positions on the x-axis, not one position {x, y, z}
        p = np.array([3.0, 4.0, 1.5])
        val = quadriga_lib.sos.rand(self.dd, p, seed=7)
        self.assertEqual(val.shape, (3, 3))
        npt.assert_array_equal(val, quadriga_lib.sos.rand(self.dd, p.reshape(3, 1), seed=7))
        one = quadriga_lib.sos.rand(self.dd, p.reshape(1, 3), seed=7)
        self.assertEqual(one.shape, (1, 3))
        self.assertTrue(np.max(np.abs(one[0, :] - val[0, :])) > 0)  # {3, 4, 1.5} is not {3, 0, 0}


if __name__ == "__main__":
    unittest.main()
