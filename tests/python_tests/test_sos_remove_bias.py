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
        self.spatial_freq, _, _ = quadriga_lib.sos.preset("Comb128")
        self.phase = quadriga_lib.sos.init_phases(128, 3, seed=5)

        # 20 positions on a grid; the second device on another set of positions
        x, y = np.meshgrid(np.arange(0.0, 25.0, 5.0), np.arange(0.0, 20.0, 5.0))
        x, y = x.ravel(order="F"), y.ravel(order="F")
        self.pos = np.column_stack((x, y, np.full(20, 1.5)))
        self.pos_b = np.column_stack((y + 3.0, x - 2.0, np.full(20, 1.5)))

    def mean(self, spatial_freq, phase, pos, **kwargs):
        return np.mean(quadriga_lib.sos.val(spatial_freq, phase, pos, use_kernel=1, **kwargs), axis=0)

    def wrapped(self, a):
        return np.angle(np.exp(1j * a))

    def test_basic_call(self):
        out = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 3)
        phase_out, offset, mean_before = out

        self.assertEqual(phase_out.shape, (128, 3))
        self.assertEqual(offset.shape, (3, 1))
        self.assertEqual(mean_before.shape, (3, 1))
        for x in (phase_out, offset, mean_before):
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.dtype, np.float64)

        # The mean over the positions is removed; mean_before has one row per generator
        m0 = self.mean(self.spatial_freq, self.phase, self.pos)
        self.assertTrue(np.all(np.abs(m0) > 1e-3))
        npt.assert_allclose(self.mean(self.spatial_freq, phase_out, self.pos), np.zeros(3), atol=1e-13, rtol=0)
        npt.assert_allclose(mean_before[:, 0], m0, atol=1e-13, rtol=0)

        # One common offset per column, phases in [-pi, pi)
        npt.assert_allclose(self.wrapped(phase_out - self.phase - offset[:, 0]), np.zeros((128, 3)), atol=1e-13, rtol=0)
        self.assertTrue(np.all(phase_out >= -np.pi) and np.all(phase_out < np.pi))
        self.assertTrue(np.all(np.abs(offset) <= np.pi / 2 + 1e-12))
        self.assertGreater(np.max(np.abs(offset)), 1e-3)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b,
                                           spatial_freq_scale_b=2.5, reciprocal=False)
        out = quadriga_lib.sos.remove_bias(spatial_freq=self.spatial_freq, phase_in=self.phase, pos=self.pos,
                                           pos_b=self.pos_b, spatial_freq_scale_b=2.5, reciprocal=False)
        self.assertEqual(len(out), 3)
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos)

        # None and empty arrays are the same as omitted; reciprocal has no effect for one device
        for out in (quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, pos_b=None,
                                                 spatial_freq_scale_b=1.0, reciprocal=True),
                    quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, pos_b=np.zeros((0, 3))),
                    quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, reciprocal=False)):
            for a, b in zip(out, ref):
                npt.assert_array_equal(a, b)

        # Two devices: one offset per generator; the mean of sos.val with the same arguments is removed
        for kw in (dict(), dict(reciprocal=False), dict(reciprocal=False, spatial_freq_scale_b=2.5)):
            p, o, mb = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b, **kw)
            self.assertEqual(p.shape, (128, 3))
            self.assertEqual(o.shape, (3, 1))
            self.assertEqual(mb.shape, (3, 1))
            npt.assert_allclose(self.mean(self.spatial_freq, p, self.pos, pos_b=self.pos_b, **kw), np.zeros(3),
                                atol=1e-12, rtol=0)
            npt.assert_allclose(mb[:, 0], self.mean(self.spatial_freq, self.phase, self.pos, pos_b=self.pos_b, **kw),
                                atol=1e-12, rtol=0)
            npt.assert_allclose(self.wrapped(p - self.phase - o[:, 0]), np.zeros((128, 3)), atol=1e-13, rtol=0)

        # pos_b, reciprocal and the frequency scale reach the library
        o1 = ref[1]
        o2 = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b)[1]
        o3 = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b, reciprocal=False)[1]
        o4 = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b, reciprocal=False,
                                          spatial_freq_scale_b=2.5)[1]
        for a, b in ((o1, o2), (o2, o3), (o3, o4)):
            self.assertGreater(np.max(np.abs(a - b)), 1e-3)

    def test_array_layout_and_dtype(self):
        args = [self.spatial_freq, self.phase, self.pos, self.pos_b]
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
        self.assertEqual(out[0].dtype, np.float64)
        npt.assert_allclose(self.wrapped(out[0] - ref[0]), np.zeros((128, 3)), atol=1e-4, rtol=0)

        # Integer and bool scalars
        out = quadriga_lib.sos.remove_bias(*args, spatial_freq_scale_b=np.int32(2), reciprocal=np.bool_(False))
        ref2 = quadriga_lib.sos.remove_bias(*args, spatial_freq_scale_b=2.0, reciprocal=False)
        for a, b in zip(out, ref2):
            npt.assert_array_equal(a, b)

    def test_inputs_not_modified(self):
        # Fortran-ordered float64 arrays are the case in which the helpers could hand out a view
        args = [np.asfortranarray(a) for a in (self.spatial_freq, self.phase, self.pos, self.pos_b)]
        copies = [a.copy() for a in args]
        phase_out, _, _ = quadriga_lib.sos.remove_bias(*args)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

        # The output is a new array and differs from the input
        self.assertFalse(np.shares_memory(phase_out, args[1]))
        self.assertGreater(np.max(np.abs(phase_out - args[1])), 1e-3)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase[:5, :], self.pos)
        self.assertIn("phase", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, pos_b=self.pos_b[:5, :])
        self.assertEqual(str(ctx.exception), "Number of rows in 'pos_b' does not match number of rows in 'pos'.")

        # A frequency scale other than 1 needs a second device and reciprocal=False
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, spatial_freq_scale_b=2.0,
                                         reciprocal=False)
        self.assertEqual(str(ctx.exception), "Input 'spatial_freq_scale_b' other than 1 requires 'pos_b'.")

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, pos_b=self.pos_b,
                                         spatial_freq_scale_b=2.0)
        self.assertEqual(str(ctx.exception), "Input 'spatial_freq_scale_b' other than 1 requires 'reciprocal = false'.")

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, no_such_argument=1)
        for removed in ("phase_b_in", "acf", "dist"):  # inputs of the former two-map model
            with self.assertRaises(TypeError):
                quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b, **{removed: [0.0]})
        with self.assertRaises(TypeError) as ctx:
            quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, self.pos, self.pos_b, reciprocal=1)
        self.assertIn("'reciprocal'", str(ctx.exception))

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.spatial_freq, self.phase, self.pos]
        quadriga_lib.sos.remove_bias(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.remove_bias(*bad)

    def test_one_dimensional_pos(self):
        # A 1-D array is a column: three positions on the x-axis, not one position {x, y, z}
        p = np.array([3.0, 14.0, 31.5])
        out = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, p)
        ref = quadriga_lib.sos.remove_bias(self.spatial_freq, self.phase, p.reshape(3, 1))
        for a, b in zip(out, ref):
            npt.assert_array_equal(a, b)
        npt.assert_allclose(self.mean(self.spatial_freq, out[0], p.reshape(3, 1)), 0.0, atol=1e-13, rtol=0)


if __name__ == "__main__":
    unittest.main()
