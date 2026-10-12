# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.val.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_val.cpp).

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
        # Two sinusoids along x and y with 4 m wavelength, amplitude sqrt(2/2) = 1
        # Generator 0: cos( pi/2 x ) + cos( pi/2 y );  generator 1: -sin( pi/2 x ) + cos( pi/2 y )
        self.freq = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0]])
        self.phase = np.array([[0.0, np.pi / 2], [0.0, 0.0]])
        self.pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        self.ref = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, -1.0]])

        # ACF table for two devices on the same map; second device at the origin: distances 0, 1, 2
        self.acf = np.array([1.0, 0.5, 0.0])
        self.dist = np.array([0.0, 1.0, 2.0])
        self.pos_b = np.zeros((3, 3))

    def test_basic_call(self):
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos)
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (3, 2))  # (n_pos, n_gen)
        self.assertEqual(val.dtype, np.float64)
        npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)  # default kernel may be AVX2 (single precision)

        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, use_kernel=1)
        npt.assert_allclose(val, self.ref, atol=1e-14, rtol=0)

    def test_keyword_arguments(self):
        phase_b = self.phase[:, ::-1]
        ref = quadriga_lib.sos.val(self.freq, self.phase, self.pos, self.pos_b, phase_b, self.acf, self.dist,
                                   2.0, True, 1)
        val = quadriga_lib.sos.val(freq=self.freq, phase=self.phase, pos=self.pos, pos_b=self.pos_b,
                                   phase_b=phase_b, acf=self.acf, dist=self.dist, freq_scale_b=2.0,
                                   uniform=True, use_kernel=1)
        npt.assert_array_equal(val, ref)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.val(self.freq, self.phase, self.pos, use_kernel=1)

        # None and empty arrays are the same as omitted
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, None, None, None, None, 1.0, False, 1)
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, np.zeros((0, 3)), [], [], [], use_kernel=1)
        npt.assert_array_equal(val, ref)

        # uniform
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, uniform=True, use_kernel=1)
        npt.assert_allclose(val, [[0.9772498680518208, 0.8413447460685429], [0.8413447460685429, 0.5],
                                  [0.5, 0.15865525393145707]], atol=1e-14, rtol=0)

        # Two devices on the same map: ( s(a) + s(b) ) / sqrt( 2 + 2 A(D) ), A = 1, 0.5, 0
        s3, s2 = np.sqrt(3.0), np.sqrt(2.0)
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, self.pos_b, acf=self.acf, dist=self.dist,
                                   use_kernel=1)
        npt.assert_allclose(val, [[2.0, 1.0], [3.0 / s3, 1.0 / s3], [2.0 / s2, 0.0]], atol=1e-14, rtol=0)

        # Two devices on different maps: the second device uses phase_b, here the two generators swapped
        phase_b = self.phase[:, ::-1]
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, self.pos_b, phase_b, use_kernel=1)
        npt.assert_allclose(val, np.array([[3.0, 3.0], [2.0, 2.0], [1.0, 1.0]]) / s2, atol=1e-14, rtol=0)

        # Frequency scale of the second device: at x = 1 m a scale of 2 turns the quarter wave into a half wave,
        # s_b = cos(pi) + 1 = 0 and -sin(pi) + 1 = 1
        pos_b = np.tile([1.0, 0.0, 0.0], (3, 1))
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, pos_b, self.phase, freq_scale_b=2.0,
                                   use_kernel=1)
        npt.assert_allclose(val, (self.ref + [0.0, 1.0]) / s2, atol=1e-14, rtol=0)

    def test_kernels(self):
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, use_kernel=0)
        npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)

        # AVX2 is a build option: the kernel either works (single precision) or the library reports it missing
        try:
            val = quadriga_lib.sos.val(self.freq, self.phase, self.pos, use_kernel=2)
        except ValueError as err:
            self.assertIn("AVX2", str(err))
        else:
            npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)

    def test_shapes(self):
        # Positions with fewer columns than freq; missing coordinates are 0
        val = quadriga_lib.sos.val(self.freq, self.phase, self.pos[:, :2], use_kernel=1)
        npt.assert_allclose(val, self.ref, atol=1e-14, rtol=0)

        # 1-D: sqrt(2) cos( pi/2 x ); a 1-D position array is one coordinate per position
        val = quadriga_lib.sos.val([[0.25]], [[0.0]], [0.0, 1.0, 2.0], use_kernel=1)
        self.assertEqual(val.shape, (3, 1))
        npt.assert_allclose(val[:, 0], np.sqrt(2.0) * np.array([1.0, 0.0, -1.0]), atol=1e-14, rtol=0)

    def test_array_layout_and_dtype(self):
        phase_b = self.phase[:, ::-1]
        args = [self.freq, self.phase, self.pos, self.pos_b, phase_b]
        ref = quadriga_lib.sos.val(*args, use_kernel=1)

        variants = [
            lambda a: np.ascontiguousarray(a),            # C order
            lambda a: np.asfortranarray(a),               # Fortran order
            lambda a: np.repeat(a, 2, axis=0)[::2, :],    # non-contiguous view
            lambda a: a.astype(np.float32),               # other dtype
            lambda a: a.tolist(),                         # plain lists
        ]
        for fn in variants:
            val = quadriga_lib.sos.val(*[fn(a) for a in args], use_kernel=1)
            self.assertEqual(val.dtype, np.float64)
            npt.assert_allclose(val, ref, atol=1e-6, rtol=0)  # float32 input

        # Integer and bool scalars
        val = quadriga_lib.sos.val(*args, freq_scale_b=1, uniform=0, use_kernel=np.int32(1))
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.val(*args, uniform=np.bool_(True), use_kernel=1)
        npt.assert_array_equal(val, quadriga_lib.sos.val(*args, uniform=True, use_kernel=1))

    def test_inputs_not_modified(self):
        phase_b = self.phase[:, ::-1].copy()
        args = [self.freq, self.phase, self.pos, self.pos_b, phase_b, self.acf, self.dist]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.val(*args)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.freq, self.phase[:1, :], self.pos)
        self.assertIn("phase", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.freq, self.phase, self.pos, self.pos_b)
        self.assertIn("acf", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.freq, self.phase, self.pos, self.pos_b, None, self.acf, self.dist, 2.0)
        self.assertIn("freq_scale_b", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.freq, self.phase, self.pos, use_kernel=3)
        self.assertIn("use_kernel", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.freq, self.phase)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.freq, self.phase, self.pos, no_such_argument=1)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.freq, self.phase, self.pos, uniform="yes")
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.freq, self.phase, self.pos, use_kernel=1.0)

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.freq, self.phase, self.pos]
        quadriga_lib.sos.val(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.val(*bad)


if __name__ == "__main__":
    unittest.main()
