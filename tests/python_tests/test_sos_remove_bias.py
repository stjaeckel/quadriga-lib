# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.remove_bias.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_remove_bias.cpp).

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
        self.freq, self.acf, self.dist = quadriga_lib.sos.preset("Comb128")
        self.phase = quadriga_lib.sos.init_phases(128, 3, 5)
        self.phase_b = quadriga_lib.sos.init_phases(128, 3, 6)

        # 20 positions on a grid; the second device on another set of positions
        x, y = np.meshgrid(np.arange(0.0, 25.0, 5.0), np.arange(0.0, 20.0, 5.0))
        x, y = x.ravel(order="F"), y.ravel(order="F")
        self.pos = np.column_stack((x, y, np.full(20, 1.5)))
        self.pos_b = np.column_stack((y + 3.0, x - 2.0, np.full(20, 1.5)))

    def mean(self, freq, phase, pos, **kwargs):
        return np.mean(quadriga_lib.sos.val(freq, phase, pos, use_kernel=1, **kwargs), axis=0)

    def wrapped(self, a):
        return np.angle(np.exp(1j * a))

    def test_basic_call(self):
        out = quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 4)
        phase_out, phase_b_out, offset, mean_before = out

        self.assertEqual(phase_out.shape, (128, 3))
        self.assertIsNone(phase_b_out)  # phase_b_in was not given
        self.assertEqual(offset.shape, (3, 1))
        self.assertEqual(mean_before.shape, (3, 1))
        for x in (phase_out, offset, mean_before):
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.dtype, np.float64)

        # The mean over the positions is removed; mean_before has one row per generator
        m0 = self.mean(self.freq, self.phase, self.pos)
        self.assertTrue(np.all(np.abs(m0) > 1e-3))
        npt.assert_allclose(self.mean(self.freq, phase_out, self.pos), np.zeros(3), atol=1e-13, rtol=0)
        npt.assert_allclose(mean_before[:, 0], m0, atol=1e-13, rtol=0)

        # One common offset per column, phases in [-pi, pi)
        npt.assert_allclose(self.wrapped(phase_out - self.phase - offset[:, 0]), np.zeros((128, 3)), atol=1e-13, rtol=0)
        self.assertTrue(np.all(phase_out >= -np.pi) and np.all(phase_out < np.pi))
        self.assertTrue(np.all(np.abs(offset) <= np.pi / 2 + 1e-12))
        self.assertGreater(np.max(np.abs(offset)), 1e-3)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, self.pos_b, self.phase_b, self.acf,
                                           self.dist, 2.5)
        out = quadriga_lib.sos.remove_bias(freq=self.freq, phase_in=self.phase, pos=self.pos, pos_b=self.pos_b,
                                           phase_b_in=self.phase_b, acf=self.acf, dist=self.dist, freq_scale_b=2.5)
        self.assertEqual(len(out), 4)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos)
        m0 = self.mean(self.freq, self.phase, self.pos)

        # None and empty arrays are the same as omitted
        for out in (quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, None, None, None, None, 1.0),
                    quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, np.zeros((0, 3)), [], [], [])):
            npt.assert_array_equal(out[0], ref[0])
            self.assertIsNone(out[1])
            npt.assert_array_equal(out[2], ref[2])
            npt.assert_array_equal(out[3], ref[3])

        # Two devices on the same map: one offset per generator, no phases for the second device
        p3, pb3, o3, mb3 = quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, self.pos_b, acf=self.acf,
                                                        dist=self.dist)
        self.assertIsNone(pb3)
        self.assertEqual(o3.shape, (3, 1))
        self.assertEqual(mb3.shape, (3, 1))
        same_map = dict(pos_b=self.pos_b, acf=self.acf, dist=self.dist)
        npt.assert_allclose(self.mean(self.freq, p3, self.pos, **same_map), np.zeros(3), atol=1e-13, rtol=0)
        npt.assert_allclose(mb3[:, 0], self.mean(self.freq, self.phase, self.pos, **same_map), atol=1e-13, rtol=0)

        # The returned None can be passed on as "same map"
        val = quadriga_lib.sos.val(self.freq, p3, self.pos, self.pos_b, pb3, self.acf, self.dist, use_kernel=1)
        npt.assert_allclose(np.mean(val, axis=0), np.zeros(3), atol=1e-13, rtol=0)

        # Two devices on different maps, with a frequency scale for the second device
        p4, pb4, o4, mb4 = quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, self.pos_b, self.phase_b,
                                                        freq_scale_b=2.5)
        self.assertEqual(pb4.shape, (128, 3))
        self.assertEqual(pb4.dtype, np.float64)
        self.assertEqual(o4.shape, (3, 2))
        self.assertEqual(mb4.shape, (3, 2))
        npt.assert_array_equal(p4, ref[0])  # the first device is treated on its own
        npt.assert_allclose(self.mean(self.freq * 2.5, pb4, self.pos_b), np.zeros(3), atol=1e-13, rtol=0)

        # Column 0 belongs to the first device, column 1 to the second
        m0b = self.mean(self.freq * 2.5, self.phase_b, self.pos_b)
        npt.assert_allclose(mb4, np.column_stack((m0, m0b)), atol=1e-13, rtol=0)
        npt.assert_allclose(o4[:, 0], ref[2][:, 0], atol=1e-14, rtol=0)
        npt.assert_allclose(self.wrapped(pb4 - self.phase_b - o4[:, 1]), np.zeros((128, 3)), atol=1e-13, rtol=0)

    def test_array_layout_and_dtype(self):
        args = [self.freq, self.phase, self.pos, self.pos_b, self.phase_b]
        ref = quadriga_lib.sos.remove_bias(*args)

        variants = [
            lambda a: np.ascontiguousarray(a),            # C order
            lambda a: np.asfortranarray(a),               # Fortran order
            lambda a: np.repeat(a, 2, axis=0)[::2, :],    # non-contiguous view
            lambda a: a.tolist(),                         # plain lists
        ]
        for fn in variants:
            out = quadriga_lib.sos.remove_bias(*[fn(a) for a in args])
            for a, b in zip(out, ref):
                self.assertEqual(a.dtype, np.float64)
                npt.assert_array_equal(a, b)

        # float32 input: the phases are rounded, the offsets follow
        out = quadriga_lib.sos.remove_bias(*[a.astype(np.float32) for a in args])
        for a, b in zip(out[:2], ref[:2]):
            self.assertEqual(a.dtype, np.float64)
            npt.assert_allclose(self.wrapped(a - b), np.zeros((128, 3)), atol=1e-4, rtol=0)

    def test_inputs_not_modified(self):
        # Fortran-ordered float64 arrays are the case in which the helpers could hand out a view
        args = [np.asfortranarray(a) for a in (self.freq, self.phase, self.pos, self.pos_b, self.phase_b)]
        copies = [a.copy() for a in args]
        phase_out, phase_b_out, _, _ = quadriga_lib.sos.remove_bias(*args)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

        # The outputs are new arrays and differ from the inputs
        self.assertFalse(np.shares_memory(phase_out, args[1]))
        self.assertFalse(np.shares_memory(phase_b_out, args[4]))
        self.assertGreater(np.max(np.abs(phase_out - args[1])), 1e-3)
        self.assertGreater(np.max(np.abs(phase_b_out - args[4])), 1e-3)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.freq, self.phase[:5, :], self.pos)
        self.assertIn("phase", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, self.pos_b)
        self.assertIn("acf", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, self.pos_b, None, self.acf, self.dist, 2.0)
        self.assertIn("freq_scale_b", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.remove_bias(self.freq, self.phase)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, no_such_argument=1)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.remove_bias(self.freq, self.phase, self.pos, phase_b=self.phase_b)  # the name is phase_b_in

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.freq, self.phase, self.pos]
        quadriga_lib.sos.remove_bias(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.remove_bias(*bad)

    def test_one_dimensional_pos(self):
        # A 1-D array is a column: three positions on the x-axis, not one position {x, y, z}
        p = np.array([3.0, 14.0, 31.5])
        out = quadriga_lib.sos.remove_bias(self.freq, self.phase, p)
        ref = quadriga_lib.sos.remove_bias(self.freq, self.phase, p.reshape(3, 1))
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)
        npt.assert_allclose(self.mean(self.freq, out[0], p.reshape(3, 1)), 0.0, atol=1e-13, rtol=0)


if __name__ == "__main__":
    unittest.main()
