# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.arrayant.generate_antenna.
# Scope: each argument once on one type, omitted and None arguments, output dict, layouts and dtypes of 'beams' and
# 'pattern', error paths. Physics validation lives in the Catch2 suite (test_generate_antenna.cpp).

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

OPTIONAL = ("center_freq", "element", "panel", "spacing", "pol", "ports", "steer", "beams", "device_size",
            "locations", "pattern", "separate_beams", "apply_weights")


class test_case(unittest.TestCase):

    def test_basic_call(self):
        ant = quadriga_lib.arrayant.generate_antenna("omni", 30)
        self.assertIsInstance(ant, dict)
        self.assertEqual(ant["e_theta_re"].shape, (7, 13, 1))
        self.assertEqual(ant["e_theta_re"].dtype, np.float64)
        self.assertTrue(ant["e_theta_re"].flags.f_contiguous)
        npt.assert_array_equal(ant["e_theta_re"], np.ones((7, 13, 1)))
        npt.assert_array_equal(ant["e_phi_re"], np.zeros((7, 13, 1)))
        npt.assert_allclose(ant["azimuth_grid"], np.linspace(-np.pi, np.pi, 13), atol=1e-14, rtol=0)
        npt.assert_allclose(ant["elevation_grid"], np.linspace(-np.pi / 2, np.pi / 2, 7), atol=1e-14, rtol=0)
        npt.assert_array_equal(ant["element_pos"], np.zeros((3, 1)))
        npt.assert_array_equal(ant["port_weights_re"], [[1.0]])
        npt.assert_array_equal(ant["port_weights_im"], [[0.0]])
        self.assertEqual(ant["center_freq"], 299792458.0)
        self.assertEqual(ant["name"], "omni")
        ref = quadriga_lib.arrayant.generate_antenna(type="omni", res=30)  # by keyword
        for key in ant:
            npt.assert_array_equal(ant[key], ref[key])

    def test_omitted_and_none(self):
        ref = quadriga_lib.arrayant.generate_antenna("omni")
        self.assertEqual(ref["e_theta_re"].shape, (19, 37, 1))  # res default 10 for omni
        for res in (np.nan, 0.0, -1.0):
            npt.assert_array_equal(quadriga_lib.arrayant.generate_antenna("omni", res)["e_theta_re"], ref["e_theta_re"])
        for name in OPTIONAL[1:]:
            out = quadriga_lib.arrayant.generate_antenna("omni", **{name: None})
            for key in ref:
                npt.assert_array_equal(out[key], ref[key])
        out = quadriga_lib.arrayant.generate_antenna("omni", pattern={})  # an empty dict is no pattern
        npt.assert_array_equal(out["e_theta_re"], ref["e_theta_re"])

    def test_arguments(self):
        f = quadriga_lib.arrayant.generate_antenna
        a = f("xpol", 30, center_freq=3e9)
        self.assertEqual(a["e_theta_re"].shape, (7, 13, 2))
        self.assertEqual(a["center_freq"], 3e9)
        self.assertEqual(f("omni", 30, center_freq=float("nan"))["center_freq"], 299792458.0)  # NaN: default

        a = f("3gpp", 30, panel=[2, 3])
        self.assertEqual(a["e_theta_re"].shape[2], 6)
        npt.assert_allclose(a["element_pos"][:, :2], [[0, 0], [-0.5, -0.5], [-0.25, 0.25]], atol=1e-12, rtol=0)
        npt.assert_array_equal(f("3gpp", 30, panel=2)["element_pos"], f("3gpp", 30, panel=[2])["element_pos"])

        a = f("ula", 30, panel=[1, 2], spacing=0.7)
        npt.assert_allclose(a["element_pos"], [[0, 0], [-0.35, 0.35], [0, 0]], atol=1e-12, rtol=0)

        self.assertEqual(f("3gpp", 30, panel=4, ports=1)["port_weights_re"].shape, (4, 1))

        a = f("3gpp", 30, pol=[2, 2])  # Model-2, zeta = 90 deg
        self.assertEqual(a["e_theta_re"].shape[2], 2)
        npt.assert_allclose(a["e_theta_re"][:, :, 1], 0.0, atol=1e-12, rtol=0)

        # steer: eq. 7.3-1 with d_V = 0.5; a 1D input is one row [tilt, pan], as the 2D (1, 2) input
        for steer in ([30.0, 0.0], np.array([[30.0, 0.0]])):
            a = f("3gpp", 30, panel=2, ports=1, steer=steer)
            npt.assert_allclose(a["port_weights_re"], [[0.5], [0.5]], atol=1e-12, rtol=0)
            npt.assert_allclose(a["port_weights_im"], [[-0.5], [0.5]], atol=1e-12, rtol=0)
        a = f("3gpp", 30, panel=[1, 2], ports=[1, 1], steer=[0.0, 30.0])  # pan: d_H = 0.5, the beam towards +y
        npt.assert_allclose(a["port_weights_re"], [[0.5], [0.5]], atol=1e-12, rtol=0)
        npt.assert_allclose(a["port_weights_im"], [[0.5], [-0.5]], atol=1e-12, rtol=0)
        a = f("3gpp", 30, panel=[2, 1, 2], ports=1, steer=[[30.0], [-30.0]])  # one row per panel, pan 0
        npt.assert_allclose(a["port_weights_im"][:2, 0], [-0.5, 0.5], atol=1e-12, rtol=0)
        npt.assert_allclose(a["port_weights_im"][2:, 1], [0.5, -0.5], atol=1e-12, rtol=0)

        a = f("custom", 30, element=[60.0, 90.0, 10.0])  # floor 10 dB below the peak
        p = a["e_theta_re"] ** 2
        npt.assert_allclose(p.min() / p.max(), 0.1, atol=1e-12, rtol=0)
        self.assertGreater(np.abs(a["e_theta_re"] - f("custom", 30, element=60.0)["e_theta_re"]).max(), 0.01)

        pat = f("custom", 30, element=[60.0, 40.0])
        a = f("3gpp", panel=2, pattern=pat)
        self.assertEqual(a["e_theta_re"].shape, (7, 13, 2))
        npt.assert_allclose(a["e_theta_re"][:, :, 1], pat["e_theta_re"][:, :, 0], atol=1e-12, rtol=0)

        beams = [[-30.0, 30.0], [0.0, 0.0]]
        self.assertEqual(f("multibeam", 30, panel=[1, 4], beams=beams, separate_beams=True)["port_weights_re"].shape, (4, 2))
        self.assertEqual(f("multibeam", 30, panel=[1, 4], beams=beams)["port_weights_re"].shape, (4, 1))

        a = f("3gpp", 30, panel=2, ports=1, apply_weights=np.bool_(True))
        self.assertEqual(a["e_theta_re"].shape, (7, 13, 1))
        npt.assert_allclose(a["element_pos"], np.zeros((3, 1)), atol=1e-12, rtol=0)

    def test_beams_layout_and_dtype(self):
        f = quadriga_lib.arrayant.generate_antenna
        b = np.array([[20.0, 0.0], [-7.0, 30.0], [2.0, 1.0]])
        ref = f("multibeam", 30, panel=[2, 3], beams=b)
        for beams in (b.tolist(), np.asfortranarray(b), b.astype(np.float32), np.ascontiguousarray(b), b.astype(np.int64)):
            out = f("multibeam", 30, panel=[2, 3], beams=beams)
            npt.assert_allclose(out["port_weights_re"], ref["port_weights_re"], atol=1e-6, rtol=0)
            npt.assert_allclose(out["port_weights_im"], ref["port_weights_im"], atol=1e-6, rtol=0)
        two = f("multibeam", 30, panel=[2, 3], beams=b[:2])  # 2 rows: weight 1
        npt.assert_array_equal(two["port_weights_re"], f("multibeam", 30, panel=[2, 3], beams=np.vstack((b[:2], [1, 1])))["port_weights_re"])
        one = f("multibeam", 30, panel=[2, 3], beams=[20.0, -7.0])  # 1D: one beam
        npt.assert_array_equal(one["port_weights_re"], f("multibeam", 30, panel=[2, 3], beams=[[20.0], [-7.0]])["port_weights_re"])

    def test_pattern_layout_and_dtype(self):
        f = quadriga_lib.arrayant.generate_antenna
        pat = f("custom", 30, element=[60.0, 40.0])
        ref = f("3gpp", panel=2, pattern=pat)
        variants = [{k: (v.astype(np.float32) if isinstance(v, np.ndarray) and k.startswith("e_") else v) for k, v in pat.items()},
                    {k: (np.ascontiguousarray(v) if isinstance(v, np.ndarray) else v) for k, v in pat.items()},
                    {k: pat[k] for k in ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im", "azimuth_grid", "elevation_grid")}]
        for p in variants:
            out = f("3gpp", panel=2, pattern=p)
            npt.assert_allclose(out["e_theta_re"], ref["e_theta_re"], atol=1e-6, rtol=0)
            npt.assert_allclose(out["element_pos"], ref["element_pos"], atol=1e-12, rtol=0)

    def test_inputs_not_modified(self):
        pat = quadriga_lib.arrayant.generate_antenna("custom", 30, element=[60.0, 40.0])
        keep = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in pat.items()}
        steer = np.array([30.0, 5.0])
        beams = np.array([[20.0, 0.0], [-7.0, 30.0], [2.0, 1.0]])
        s0, b0 = steer.copy(), beams.copy()
        quadriga_lib.arrayant.generate_antenna("3gpp", panel=[2, 2], ports=[1, 1], pol=[3, 2], steer=steer, pattern=pat)
        quadriga_lib.arrayant.generate_antenna("multibeam", 30, panel=[2, 3], beams=beams, apply_weights=True)
        npt.assert_array_equal(steer, s0)
        npt.assert_array_equal(beams, b0)
        for k in keep:
            npt.assert_array_equal(pat[k], keep[k])

    def test_errors(self):
        f = quadriga_lib.arrayant.generate_antenna
        with self.assertRaises(ValueError) as ctx:  # C++
            f("omni", 30, element=[90.0])
        self.assertIn("Input 'element' is not used by type 'omni'.", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f("bla")
        self.assertIn("Input 'type' has an unsupported value 'bla'.", str(ctx.exception))
        for name, value in (("panel", -1), ("ports", 1.5), ("pol", np.nan)):  # integer tuples
            with self.assertRaises(ValueError) as ctx:
                f("3gpp", 30, **{name: value})
            self.assertIn("Input '" + name + "' must contain integers from 0 to", str(ctx.exception))
        for name in ("separate_beams", "apply_weights"):
            with self.assertRaises(TypeError) as ctx:
                f("multibeam", 30, **{name: 1})
            self.assertIn("Input '" + name + "' must be a bool or None.", str(ctx.exception))
        with self.assertRaises(TypeError) as ctx:
            f("3gpp", 30, pattern=[1.0])
        self.assertIn("Input 'pattern' must be a dict or None.", str(ctx.exception))
        with self.assertRaises(KeyError):
            f("3gpp", 30, pattern={"e_theta_re": np.ones((7, 13, 1))})
        with self.assertRaises(TypeError) as ctx:  # pybind11: center_freq is keyword-only
            f("omni", 30, 3e9)
        self.assertIn("incompatible function arguments", str(ctx.exception))
        with self.assertRaises(TypeError):
            f("omni", 30, freq=3e9)  # the old name
        with self.assertRaises(TypeError):
            f()
        with self.assertRaises(TypeError):
            f("omni", None)

    def test_device_size_and_locations(self):
        f = quadriga_lib.arrayant.generate_antenna
        a = f("3GPP-UT", 30, device_size=[0.2, 0.1], locations=[5, 1])  # labels stay 1-based
        self.assertEqual(a["e_theta_re"].shape, (7, 13, 2))
        npt.assert_allclose(a["element_pos"], [[0.1, -0.1], [0.05, -0.05], [0, 0]], atol=1e-12, rtol=0)
        a = f("3GPP-CPE", 30, locations=np.array([7], dtype=np.int32))
        npt.assert_allclose(a["element_pos"], [[0], [0.1], [0.1]], atol=1e-12, rtol=0)
        with self.assertRaises(ValueError) as ctx:
            f("3GPP-UT", 30, locations=1.5)
        self.assertIn("Input 'locations' must contain integers from 0 to", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:  # C++
            f("3GPP-UT", 30, locations=[9])
        self.assertIn("Input 'locations' must have labels from 1 to 8.", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
