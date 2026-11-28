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
        self.spatial_freq = np.array([[0.25, 0.0, 0.0], [0.0, 0.25, 0.0]])
        self.phase = np.array([[0.0, np.pi / 2], [0.0, 0.0]])
        self.pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
        self.ref = np.array([[2.0, 1.0], [1.0, 0.0], [0.0, -1.0]])

        # Second device; with two sinusoids the permutation of the two-device generator swaps them, so for
        # reciprocal=False s(a, b) = cos( pi/2 (a_x + b_y) + phi_0 ) + cos( pi/2 (a_y + b_x) + phi_1 )
        self.pos_b = np.array([[0.5, 1.0, 0.0], [2.0, 0.0, 0.0], [1.0, 3.0, 0.0]])

    def test_basic_call(self):
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos)
        self.assertIsInstance(val, np.ndarray)
        self.assertEqual(val.shape, (3, 2))  # (n_pos, n_gen)
        self.assertEqual(val.dtype, np.float64)
        npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)  # default kernel may be AVX2 (single precision)

        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, use_kernel=1)
        npt.assert_allclose(val, self.ref, atol=1e-14, rtol=0)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, spatial_freq_scale_b=2.0,
                                   reciprocal=False, uniform=True, use_kernel=1)
        val = quadriga_lib.sos.val(spatial_freq=self.spatial_freq, phase=self.phase, pos=self.pos, pos_b=self.pos_b,
                                   spatial_freq_scale_b=2.0, reciprocal=False, uniform=True, use_kernel=1)
        npt.assert_array_equal(val, ref)

    def test_optional_inputs(self):
        ref = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, use_kernel=1)

        # None and empty arrays are the same as omitted; reciprocal has no effect for one device
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, pos_b=None, spatial_freq_scale_b=1.0,
                                   reciprocal=True, uniform=False, use_kernel=1)
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, pos_b=np.zeros((0, 3)), use_kernel=1)
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, reciprocal=False, use_kernel=1)
        npt.assert_array_equal(val, ref)

        # uniform
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, uniform=True, use_kernel=1)
        npt.assert_allclose(val, [[0.9772498680518208, 0.8413447460685429], [0.8413447460685429, 0.5],
                                  [0.5, 0.15865525393145707]], atol=1e-14, rtol=0)

        # Two devices, reciprocal (default): the value does not change when the devices swap positions
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, use_kernel=1)
        self.assertEqual(val.shape, (3, 2))
        npt.assert_array_equal(val, quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos_b, self.pos,
                                                         use_kernel=1))

        # Two devices, not reciprocal: the formula of setUp, other values than the reciprocal ones
        p, q, phi = self.pos, self.pos_b, self.phase
        ref2 = np.column_stack([np.cos(np.pi / 2 * (p[:, 0] + q[:, 1]) + phi[0, g]) +
                                np.cos(np.pi / 2 * (p[:, 1] + q[:, 0]) + phi[1, g]) for g in range(2)])
        val2 = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, reciprocal=False,
                                    use_kernel=1)
        npt.assert_allclose(val2, ref2, atol=1e-14, rtol=0)
        self.assertGreater(np.max(np.abs(val2 - val)), 0.1)

        # Second device at the origin, not reciprocal: the value of one device, with any frequency scale
        origin = np.zeros((3, 3))
        for scale in (1.0, 2.0):
            val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, origin, spatial_freq_scale_b=scale,
                                       reciprocal=False, use_kernel=1)
            npt.assert_allclose(val, ref, atol=1e-14, rtol=0)

        # Frequency scale of the second device: a scale of 2 equals doubled positions of the second device
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, origin, self.pos_b, spatial_freq_scale_b=2.0,
                                   reciprocal=False, use_kernel=1)
        val2 = quadriga_lib.sos.val(self.spatial_freq, self.phase, origin, 2.0 * self.pos_b, reciprocal=False,
                                    use_kernel=1)
        npt.assert_allclose(val, val2, atol=1e-14, rtol=0)

    def test_kernels(self):
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, use_kernel=0)
        npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)

        # AVX2 is a build option: the kernel either works (single precision) or the library reports it missing
        try:
            val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, use_kernel=2)
        except ValueError as err:
            self.assertIn("AVX2", str(err))
        else:
            npt.assert_allclose(val, self.ref, atol=1e-6, rtol=0)

    def test_shapes(self):
        # Positions with fewer columns than spatial_freq; missing coordinates are 0
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos[:, :2], use_kernel=1)
        npt.assert_allclose(val, self.ref, atol=1e-14, rtol=0)

        # 1-D: sqrt(2) cos( pi/2 x ); a 1-D position array is one coordinate per position
        val = quadriga_lib.sos.val([[0.25]], [[0.0]], [0.0, 1.0, 2.0], use_kernel=1)
        self.assertEqual(val.shape, (3, 1))
        npt.assert_allclose(val[:, 0], np.sqrt(2.0) * np.array([1.0, 0.0, -1.0]), atol=1e-14, rtol=0)

    def test_array_layout_and_dtype(self):
        args = [self.spatial_freq, self.phase, self.pos, self.pos_b]
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
        val = quadriga_lib.sos.val(*args, spatial_freq_scale_b=1, reciprocal=np.bool_(True), uniform=np.bool_(False),
                                   use_kernel=np.int32(1))
        npt.assert_array_equal(val, ref)
        val = quadriga_lib.sos.val(*args, uniform=np.bool_(True), use_kernel=1)
        npt.assert_array_equal(val, quadriga_lib.sos.val(*args, uniform=True, use_kernel=1))
        val = quadriga_lib.sos.val(*args, spatial_freq_scale_b=np.int32(2), reciprocal=np.bool_(False), use_kernel=1)
        npt.assert_array_equal(val, quadriga_lib.sos.val(*args, spatial_freq_scale_b=2.0, reciprocal=False,
                                                         use_kernel=1))

    def test_inputs_not_modified(self):
        args = [self.spatial_freq, self.phase, self.pos, self.pos_b]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.val(*args)
        quadriga_lib.sos.val(*args, spatial_freq_scale_b=2.0, reciprocal=False)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.spatial_freq, self.phase[:1, :], self.pos)
        self.assertIn("phase", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, pos_b=self.pos_b[:2, :])
        self.assertIn("pos_b", str(ctx.exception))

        # A frequency scale other than 1 needs a second device and reciprocal=False
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, spatial_freq_scale_b=2.0, reciprocal=False)
        self.assertEqual(str(ctx.exception), "Input 'spatial_freq_scale_b' other than 1 requires 'pos_b'.")

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, pos_b=self.pos_b, spatial_freq_scale_b=2.0)
        self.assertEqual(str(ctx.exception), "Input 'spatial_freq_scale_b' other than 1 requires 'reciprocal = false'.")

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, use_kernel=3)
        self.assertIn("use_kernel", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.spatial_freq, self.phase)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, no_such_argument=1)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, uniform="yes")
        with self.assertRaises(TypeError):
            quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, use_kernel=1.0)
        for removed in ("phase_b", "acf", "dist"):  # inputs of the former two-map model
            with self.assertRaises(TypeError):
                quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, **{removed: [0.0]})

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.spatial_freq, self.phase, self.pos]
        quadriga_lib.sos.val(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.val(*bad)


    def test_none_and_flag_types(self):
        # uniform and reciprocal: None selects the default (False, True), other types than bool raise TypeError
        ref = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, use_kernel=1)
        val = quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, uniform=None, reciprocal=None,
                                   use_kernel=1)
        npt.assert_array_equal(val, ref)
        for flag in ("uniform", "reciprocal"):
            for bad in (2.5, 1):
                with self.assertRaises(TypeError) as ctx:
                    quadriga_lib.sos.val(self.spatial_freq, self.phase, self.pos, self.pos_b, **{flag: bad})
                self.assertIn(f"'{flag}'", str(ctx.exception))

if __name__ == "__main__":
    unittest.main()
