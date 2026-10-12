# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.sos.map.
# Scope: argument and output handling, defaults, array handling, error paths.
# Physics validation lives in the Catch2 suite (test_sos_map.cpp).

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
        # One sinusoid per axis with different wavelengths (4 m, 8 m, 16 m), amplitude sqrt(2/3)
        self.freq = np.diag([1.0 / 4, 1.0 / 8, 1.0 / 16])
        self.phase = np.array([[0.0, np.pi / 2], [0.0, 0.0], [0.0, 0.0]])
        self.x = np.array([0.0, 1.0, 2.0])
        self.y = np.array([0.0, 4.0])
        self.z = np.array([0.0, 4.0, 8.0, 16.0])

        # Reference in meshgrid order (ny, nx, nz, n_gen)
        a = np.sqrt(2.0 / 3.0)
        yy, xx, zz = np.meshgrid(self.y, self.x, self.z, indexing="ij")
        common = np.cos(np.pi / 4 * yy) + np.cos(np.pi / 8 * zz)
        self.ref = a * np.stack((np.cos(np.pi / 2 * xx) + common, -np.sin(np.pi / 2 * xx) + common), axis=3)

    def test_basic_call(self):
        out = quadriga_lib.sos.map(self.freq, self.phase, self.x, self.y, self.z, False, 1)
        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (2, 3, 4, 2))  # (ny, nx, nz, n_gen)
        self.assertEqual(out.dtype, np.float64)
        npt.assert_allclose(out, self.ref, atol=1e-13, rtol=0)

    def test_keyword_arguments(self):
        ref = quadriga_lib.sos.map(self.freq, self.phase, self.x, self.y, self.z, True, 1)
        out = quadriga_lib.sos.map(freq=self.freq, phase=self.phase, x=self.x, y=self.y, z=self.z,
                                   uniform=True, use_kernel=1)
        npt.assert_array_equal(out, ref)

    def test_optional_inputs(self):
        # y and z omitted: one sample at 0 on these axes
        m1 = quadriga_lib.sos.map(self.freq, self.phase, self.x, use_kernel=1)
        self.assertEqual(m1.shape, (1, 3, 1, 2))
        npt.assert_allclose(m1, self.ref[:1, :, :1, :], atol=1e-13, rtol=0)

        # None and empty arrays are the same as omitted
        m2 = quadriga_lib.sos.map(self.freq, self.phase, self.x, None, None, False, 1)
        npt.assert_array_equal(m2, m1)
        m2 = quadriga_lib.sos.map(self.freq, self.phase, self.x, [], np.zeros(0), use_kernel=1)
        npt.assert_array_equal(m2, m1)

        # y only, and z without y
        m3 = quadriga_lib.sos.map(self.freq, self.phase, self.x, self.y, use_kernel=1)
        self.assertEqual(m3.shape, (2, 3, 1, 2))
        npt.assert_allclose(m3, self.ref[:, :, :1, :], atol=1e-13, rtol=0)
        m4 = quadriga_lib.sos.map(self.freq, self.phase, self.x, z=self.z, use_kernel=1)
        self.assertEqual(m4.shape, (1, 3, 4, 2))
        npt.assert_allclose(m4, self.ref[:1, :, :, :], atol=1e-13, rtol=0)

        # uniform: 0.5 erfc( -val / sqrt(2) ), i.e. the value of sos.val with the same flag
        m5 = quadriga_lib.sos.map(self.freq, self.phase, self.x, self.y, self.z, uniform=True, use_kernel=1)
        yy, xx, zz = np.meshgrid(self.y, self.x, self.z, indexing="ij")
        pos = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
        val = quadriga_lib.sos.val(self.freq, self.phase, pos, uniform=True, use_kernel=1)
        npt.assert_allclose(m5.reshape(-1, 2), val, atol=1e-14, rtol=0)
        self.assertTrue(np.all(m5 > 0.0) and np.all(m5 < 1.0))

    def test_kernels(self):
        out = quadriga_lib.sos.map(self.freq, self.phase, self.x, self.y, self.z)  # use_kernel = 0
        npt.assert_allclose(out, self.ref, atol=1e-6, rtol=0)  # may be AVX2 (single precision)

        # AVX2 is a build option: the kernel either works (single precision) or the library reports it missing
        try:
            out = quadriga_lib.sos.map(self.freq, self.phase, self.x, self.y, self.z, use_kernel=2)
        except ValueError as err:
            self.assertIn("AVX2", str(err))
        else:
            npt.assert_allclose(out, self.ref, atol=1e-6, rtol=0)

    def test_array_layout_and_dtype(self):
        mats = [
            lambda a: np.ascontiguousarray(a),            # C order
            lambda a: np.asfortranarray(a),               # Fortran order
            lambda a: np.repeat(a, 2, axis=0)[::2, :],    # non-contiguous view
            lambda a: a.astype(np.float32),               # other dtype
            lambda a: a.tolist(),                         # plain lists
        ]
        vecs = [
            lambda a: a,
            lambda a: a.reshape(-1, 1),                   # column
            lambda a: np.repeat(a, 2)[::2],               # non-contiguous view
            lambda a: a.astype(np.int32),                 # integer grid
            lambda a: a.tolist(),
        ]
        for fm, fv in zip(mats, vecs):
            out = quadriga_lib.sos.map(fm(self.freq), fm(self.phase), fv(self.x), fv(self.y), fv(self.z),
                                       use_kernel=1)
            self.assertEqual(out.shape, (2, 3, 4, 2))
            self.assertEqual(out.dtype, np.float64)
            npt.assert_allclose(out, self.ref, atol=1e-6, rtol=0)  # float32 input

    def test_inputs_not_modified(self):
        args = [self.freq, self.phase, self.x, self.y, self.z]
        copies = [a.copy() for a in args]
        quadriga_lib.sos.map(*args)
        for a, c in zip(args, copies):
            npt.assert_array_equal(a, c)

    def test_errors(self):
        # Thrown by C++: std::invalid_argument -> ValueError
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.map(self.freq, self.phase[:2, :], self.x)
        self.assertIn("phase", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.map(self.freq, self.phase, [])
        self.assertIn("'x'", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.map(self.freq[:, :2], self.phase, self.x, self.y, self.z)
        self.assertIn("'z'", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.sos.map(self.freq, self.phase, self.x, use_kernel=3)
        self.assertIn("use_kernel", str(ctx.exception))

        # Rejected by pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.sos.map(self.freq, self.phase)
        with self.assertRaises(TypeError):
            quadriga_lib.sos.map(self.freq, self.phase, self.x, no_such_argument=1)

    def test_none_for_mandatory_arrays(self):
        # None for a mandatory array is rejected by pybind11; it must not be read as one value of NaN
        args = [self.freq, self.phase, self.x]
        quadriga_lib.sos.map(*args)
        for i in range(3):
            bad = list(args)
            bad[i] = None
            with self.assertRaises(TypeError):
                quadriga_lib.sos.map(*bad)


if __name__ == "__main__":
    unittest.main()
