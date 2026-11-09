# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.arrayant.interpolate.
# Scope: argument and output handling, defaults, single- and multi-frequency mode, kernels, flags, error paths.
# Physics validation lives in the Catch2 suites (test_arrayant_interpolate.cpp, test_arrayant_interpolate_multi.cpp).

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


def two_element_ant():
    """Two elements with two azimuth samples (0 and pi): element 0 has e-theta in opposite phase (linear) and e-phi
    in phase, element 1 has e-theta with a 90 degree phase step. Only the mandatory keys."""
    return {
        "e_theta_re": np.stack([[[-2.0, 2.0]], [[1.0, 0.0]]], axis=2),
        "e_theta_im": np.stack([[[-1.0, 1.0]], [[0.0, 1.0]]], axis=2),
        "e_phi_re": np.stack([[[3.0, 1.0]], [[0.0, 0.0]]], axis=2),
        "e_phi_im": np.stack([[[6.0, 2.0]], [[0.0, 0.0]]], axis=2),
        "azimuth_grid": np.array([0.0, np.pi]),
        "elevation_grid": np.array([0.0]),
    }


class test_case(unittest.TestCase):

    def setUp(self):
        self.ant = two_element_ant()
        self.az = np.array([[0.0, 1.0, 2.0]]) * np.pi / 4
        self.el = np.array([[-0.5, 0.0, 0.5]])
        c, s = np.cos(np.arange(3) * np.pi / 8), np.sin(np.arange(3) * np.pi / 8)
        self.V_re = np.array([[-2.0, -1.0, 0.0], c])
        self.V_im = np.array([[-1.0, -0.5, 0.0], s])
        self.H_re = np.array([[3.0, 2.5, 2.0], [0.0, 0.0, 0.0]])
        self.H_im = np.array([[6.0, 5.0, 4.0], [0.0, 0.0, 0.0]])
        # Multi-frequency dict: entry 1 is twice entry 0
        a = self.ant
        self.ant_m = {k: np.stack([a[k], 2.0 * a[k]], axis=3) for k in ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im")}
        self.ant_m.update({"azimuth_grid": a["azimuth_grid"], "elevation_grid": a["elevation_grid"],
                           "center_freq": np.array([1e9, 2e9])})

    def test_basic_call(self):
        res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el)
        self.assertIsInstance(res, tuple)
        self.assertEqual(len(res), 4)
        for x in res:
            self.assertIsInstance(x, np.ndarray)
            self.assertEqual(x.shape, (2, 3))
            self.assertEqual(x.dtype, np.float64)
        for x, ref in zip(res, (self.V_re, self.V_im, self.H_re, self.H_im)):
            npt.assert_allclose(x, ref, atol=1e-14, rtol=0)

    def test_keyword_arguments(self):
        ori = np.array([[0.1], [0.2], [0.3]])
        pos = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.5]])
        pos_res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, [1, 0], ori, pos, None, True, True,
                                                    False, 1, False)
        kw_res = quadriga_lib.arrayant.interpolate(arrayant=self.ant, azimuth=self.az, elevation=self.el, element=[1, 0],
                                                   orientation=ori, element_pos=pos, frequency=None, dist=True,
                                                   local_angles=True, fast_access=False, use_kernel=1, complex=False)
        self.assertEqual(len(kw_res), 8)
        for a, b in zip(pos_res, kw_res):
            npt.assert_array_equal(a, b)

    def test_element(self):
        V_re = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el)[0]
        for e in (None, [], np.zeros(0)):
            npt.assert_array_equal(quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, element=e)[0], V_re)
        # 0-based, permuted, repeated; integer and list input
        ref = V_re[[1, 0, 1], :]
        for e in ([1, 0, 1], np.array([1, 0, 1], dtype=np.uint32), np.array([1.0, 0.0, 1.0])):
            npt.assert_array_equal(quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, element=e)[0], ref)
        npt.assert_array_equal(quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, element=[1])[0], V_re[[1], :])

    def test_orientation(self):
        V_re = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el)[0]
        npt.assert_array_equal(quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, orientation=None)[0], V_re)
        # Heading -pi/8 adds pi/8 to the local azimuth
        res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, orientation=[[0.0], [0.0], [-np.pi / 8]],
                                                local_angles=True)
        npt.assert_allclose(res[4], np.vstack([self.az, self.az]) + np.pi / 8, atol=1e-14, rtol=0)
        npt.assert_allclose(res[0][1], np.cos(np.arange(3) * np.pi / 8 + np.pi / 16), atol=1e-14, rtol=0)
        # Per output and angle
        ori = np.zeros((3, 2, 3))
        ori[2, 1, :] = -np.pi / 8
        res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, orientation=ori, local_angles=True)
        npt.assert_allclose(res[4], np.vstack([self.az, self.az + np.pi / 8]), atol=1e-14, rtol=0)

    def test_element_pos_and_dist(self):
        pos = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
        az, el = np.array([[0.0, np.pi / 2]]), np.zeros((1, 2))
        res = quadriga_lib.arrayant.interpolate(self.ant, az, el, element_pos=pos, dist=True)
        self.assertEqual(len(res), 5)
        npt.assert_allclose(res[4], [[-1.0, 0.0], [0.0, -2.0]], atol=1e-14, rtol=0)
        # Default: the positions of the arrayant, zeros if it has none
        ant_p = dict(self.ant, element_pos=pos)
        npt.assert_array_equal(quadriga_lib.arrayant.interpolate(ant_p, az, el, dist=True)[4], res[4])
        npt.assert_array_equal(quadriga_lib.arrayant.interpolate(self.ant, az, el, dist=True)[4], np.zeros((2, 2)))

    def test_local_angles(self):
        res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, local_angles=True)
        self.assertEqual(len(res), 7)
        npt.assert_allclose(res[4], np.vstack([self.az, self.az]), atol=1e-14, rtol=0)
        npt.assert_allclose(res[5], np.vstack([self.el, self.el]), atol=1e-14, rtol=0)
        npt.assert_allclose(res[6], np.zeros((2, 3)), atol=1e-14, rtol=0)
        res8 = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, dist=True, local_angles=True)
        self.assertEqual(len(res8), 8)
        for k in range(4):
            npt.assert_array_equal(res8[k], res[k])
        npt.assert_array_equal(res8[5], res[4])

    def test_multi_frequency(self):
        V_re, V_im, H_re, H_im = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el)
        res = quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el, frequency=[1e9, 1.5e9, 2e9])
        self.assertEqual(len(res), 4)
        self.assertEqual(res[0].shape, (2, 3, 3))
        for x, ref in zip(res, (V_re, V_im, H_re, H_im)):
            npt.assert_allclose(x, np.stack([ref, 1.5 * ref, 2.0 * ref], axis=2), atol=1e-12, rtol=0)  # In phase: linear
        # Default: the center frequencies of the entries
        res_d = quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el)
        npt.assert_array_equal(res_d[0], res[0][:, :, [0, 2]])
        npt.assert_array_equal(quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el, frequency=None)[0], res_d[0])
        # A 3D pattern with target frequencies is the same at every frequency
        r3 = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, frequency=1e9)
        self.assertEqual(r3[0].shape, (2, 3, 1))
        npt.assert_array_equal(r3[0][:, :, 0], V_re)
        # Element indices in the multi-frequency mode
        r_e = quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el, element=[1, 0, 1], frequency=[1.5e9])
        npt.assert_array_equal(r_e[0][:, :, 0], res[0][[1, 0, 1], :, 1])
        # Complex output
        V, H = quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el, frequency=[1e9, 1.5e9, 2e9], complex=True)
        self.assertEqual(V.dtype, np.complex128)
        npt.assert_array_equal(V.real, res[0])
        npt.assert_array_equal(H.imag, res[3])

    def test_use_kernel(self):
        ref = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el)
        for k in (0, 1):
            res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, use_kernel=k)
            for a, b in zip(res, ref):
                npt.assert_array_equal(a, b)
        try:
            res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, use_kernel=2)
            for a, b in zip(res, ref):
                npt.assert_allclose(a, b, atol=1e-5, rtol=0)  # Single precision, values up to 6
        except ValueError as e:
            self.assertIn("AVX2 kernel requested but not available", str(e))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, use_kernel=3)
        self.assertIn("'use_kernel' must be 0, 1 or 2", str(ctx.exception))

    def test_complex_flag(self):
        ref = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, dist=True, local_angles=True)
        res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, dist=True, local_angles=True, complex=True)
        self.assertEqual(len(res), 6)
        self.assertEqual(res[0].dtype, np.complex128)
        npt.assert_array_equal(res[0].real, ref[0])
        npt.assert_array_equal(res[0].imag, ref[1])
        npt.assert_array_equal(res[1].real, ref[2])
        npt.assert_array_equal(res[1].imag, ref[3])
        for k in range(4):
            npt.assert_array_equal(res[2 + k], ref[4 + k])

    def test_flags(self):
        ref = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el)
        res = quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, dist=None, local_angles=None,
                                                fast_access=None, complex=None)
        self.assertEqual(len(res), 4)
        for a, b in zip(res, ref):
            npt.assert_array_equal(a, b)
        self.assertEqual(len(quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, dist=np.bool_(True))), 5)
        for flag in ("dist", "local_angles", "fast_access", "complex"):
            for bad in (1, 2.5, "yes"):
                with self.assertRaises(TypeError) as ctx:
                    quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, **{flag: bad})
                self.assertIn("Input '%s' must be a bool or None." % flag, str(ctx.exception))

    def test_memory_layout_and_dtype(self):
        ant = quadriga_lib.arrayant.generate("3gpp", M=2, N=2, pol=2, res=10.0)
        n_out = ant["e_theta_re"].shape[2]
        rng = np.random.default_rng(0)
        az = rng.uniform(-3.0, 3.0, (n_out, 20)).astype(np.float32).astype(np.float64)
        el = rng.uniform(-1.5, 1.5, (n_out, 20)).astype(np.float32).astype(np.float64)
        ref = quadriga_lib.arrayant.interpolate(ant, np.asfortranarray(az), np.asfortranarray(el), dist=True,
                                                fast_access=True)
        big_az, big_el = np.zeros((n_out, 40)), np.zeros((n_out, 40))
        big_az[:, ::2], big_el[:, ::2] = az, el
        variants = {"C-order": (np.ascontiguousarray(az), np.ascontiguousarray(el)),
                    "float32": (az.astype(np.float32), el.astype(np.float32)),
                    "sliced": (big_az[:, ::2], big_el[:, ::2]),
                    "list": (az.tolist(), el.tolist())}
        ant_c = {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in ant.items()}
        for name, (a, e) in variants.items():
            for pattern in (ant, ant_c):
                res = quadriga_lib.arrayant.interpolate(pattern, a, e, dist=True)
                for k in range(5):
                    npt.assert_array_equal(res[k], ref[k], err_msg=name)
        # fast_access requires Fortran-ordered float64 pattern arrays
        with self.assertRaises(ValueError):
            quadriga_lib.arrayant.interpolate(ant_c, az, el, fast_access=True)

    def test_inputs_not_modified(self):
        ant_copy = {k: np.copy(v) for k, v in self.ant.items()}
        element = np.array([1, 0, 1], dtype=np.uint64)
        ori = np.array([[0.1], [0.2], [0.3]])
        pos = np.array([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
        inputs = (np.asfortranarray(self.az), np.asfortranarray(self.el), element, ori, pos)
        copies = [np.copy(x) for x in inputs]
        quadriga_lib.arrayant.interpolate(self.ant, *inputs, dist=True, local_angles=True)
        for x, y in zip(inputs, copies):
            npt.assert_array_equal(x, y)
        for k in self.ant:
            npt.assert_array_equal(self.ant[k], ant_copy[k])

    def test_empty_and_nonfinite_angles(self):
        res = quadriga_lib.arrayant.interpolate(self.ant, np.zeros((1, 0)), np.zeros((1, 0)), dist=True)
        for x in res:
            self.assertEqual(x.shape, (2, 0))
        res = quadriga_lib.arrayant.interpolate(self.ant_m, np.zeros((2, 0)), np.zeros((2, 0)))
        self.assertEqual(res[0].shape, (2, 0, 2))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.arrayant.interpolate(self.ant, np.array([]), np.array([]))
        self.assertIn("'azimuth' must have 1 or n_out rows", str(ctx.exception))
        res = quadriga_lib.arrayant.interpolate(self.ant, [[0.1, np.nan]], [[0.0, 0.0]])
        self.assertTrue(np.all(np.isnan(res[0][:, 1])))
        self.assertTrue(np.all(np.isfinite(res[0][:, 0])))

    def test_errors(self):
        # From C++
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el[:, :2])
        self.assertIn("'elevation' must have the same size as 'azimuth'", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, element=[2])
        self.assertIn("exceed the number of elements", str(ctx.exception))
        bad = dict(self.ant, azimuth_grid=np.array([0.0, np.nan]))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.arrayant.interpolate(bad, self.az, self.el)
        self.assertIn("'azimuth_grid' must be between -pi and pi", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el, frequency=[1e9, np.inf])
        self.assertIn("'frequency' must contain finite values", str(ctx.exception))
        # From the wrapper
        for flag in ("dist", "local_angles"):
            with self.assertRaises(ValueError) as ctx:
                quadriga_lib.arrayant.interpolate(self.ant_m, self.az, self.el, **{flag: True})
            self.assertIn("not available in the multi-frequency mode", str(ctx.exception))
        # From pybind11
        with self.assertRaises(TypeError):
            quadriga_lib.arrayant.interpolate(self.ant, self.az)
        with self.assertRaises(TypeError):
            quadriga_lib.arrayant.interpolate(self.ant, None, self.el)
        with self.assertRaises(TypeError):
            quadriga_lib.arrayant.interpolate(self.ant, self.az, self.el, freq=[1e9])


if __name__ == "__main__":
    unittest.main()
