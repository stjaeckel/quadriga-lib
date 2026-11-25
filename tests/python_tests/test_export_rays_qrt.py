# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.layout.export_rays_qrt.
# Scope: argument and output handling, defaults, the file written, error paths.
# The export is tested in the Catch2 suite (test_export_rays_qrt.cpp).
# Physics validation lives in the Catch2 suite.

import sys
import os
import copy
import pathlib
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def write_text(fn, text):
    with open(fn, "w") as f:
        f.write(text)


def same(a, b):
    # Deep equality of dicts, lists and arrays; NaN equals NaN, dtypes and shapes must match
    if isinstance(a, dict):
        return isinstance(b, dict) and list(a.keys()) == list(b.keys()) and all(same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return (isinstance(b, np.ndarray) and a.shape == b.shape and a.dtype == b.dtype and
                np.array_equal(a, b, equal_nan=a.dtype.kind in "fc"))
    if isinstance(a, float) and np.isnan(a):
        return isinstance(b, float) and np.isnan(b)
    return a == b


class test_case(unittest.TestCase):

    def setUp(self):
        # 2 static Tx sites (0 m and 1 km) and 2 Rx terminals (100 m and 900 m), one band
        self.tmp = tempfile.TemporaryDirectory()
        d = self.tmp.name
        fn_tx, fn_rx = os.path.join(d, "tx.csv"), os.path.join(d, "rx.csv")
        self.q1, self.q2 = os.path.join(d, "a.qrt"), os.path.join(d, "b.qrt")
        write_text(fn_tx, "name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n")
        write_text(fn_rx, "name,x,y,z,indoor\nU1,100,0,1.5,0\nU2,900,0,1.5,0\n")
        self.L = quadriga_lib.layout.csv_read(fn_tx, fn_rx, center_freq=3.5e9)[0]
        self.L2 = quadriga_lib.layout.init_link_states(self.L, seed=5)

    def tearDown(self):
        self.tmp.cleanup()

    def gains(self, fn):
        return quadriga_lib.channel.qrt_file_read(fn, [0, 1], 0)[7]

    def test_basic_call(self):
        shift = quadriga_lib.layout.export_rays_qrt(self.q1, self.L2)  # centroid of the sites and the CIR positions in x and y
        self.assertEqual((shift.shape, shift.dtype), ((2,), np.float64))
        npt.assert_allclose(shift, [500, 0], atol=1e-9, rtol=0)
        self.assertTrue(os.path.exists(self.q1))
        self.assertFalse(os.path.exists(self.q1 + ".part"))
        info = quadriga_lib.channel.qrt_file_parse(self.q1)
        self.assertEqual(info[5], ["A", "B"])  # one origin per site
        self.assertEqual(info[6], ["U1", "U2"])  # one destination per terminal
        npt.assert_array_equal(info[8], [3.5e9])
        out = quadriga_lib.channel.qrt_file_read(self.q1, [0, 1], 0)  # origin A, both CIRs
        npt.assert_allclose(out[1], [[-500, -500], [0, 0], [25, 25]], atol=1e-4, rtol=0)  # local positions: add the shift back
        npt.assert_allclose(out[3], [[-400, 400], [0, 0], [1.5, 1.5]], atol=1e-4, rtol=0)
        npt.assert_allclose(quadriga_lib.channel.qrt_file_read(self.q1, [0, 1], 1)[1], [[500, 500], [0, 0], [25, 25]], atol=1e-4, rtol=0)

    def test_optional_inputs(self):
        L2 = self.L2
        quadriga_lib.layout.export_rays_qrt(self.q1, L2)
        rays = quadriga_lib.layout.generate_rays(L2)
        shift2 = quadriga_lib.layout.export_rays_qrt(self.q2, L2, rays=rays)  # the rays of all links give the same file
        npt.assert_allclose(shift2, [500, 0], atol=1e-9, rtol=0)
        g1, g2 = self.gains(self.q1), self.gains(self.q2)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(g1, g2)))
        quadriga_lib.layout.export_rays_qrt(self.q2, L2, rays=None)  # None = generated
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(g1, self.gains(self.q2))))
        edited = [[dict(r, power=np.where(np.arange(r["power"].shape[0])[:, None] > 0, 0.0, r["power"])) for r in link] for link in rays]
        quadriga_lib.layout.export_rays_qrt(self.q2, L2, rays=edited)  # edited rays: only the first ray keeps its power
        self.assertFalse(all(np.array_equal(a, b) for a, b in zip(g1, self.gains(self.q2))))
        shift3 = quadriga_lib.layout.export_rays_qrt(pathlib.Path(self.q2), L2, rays=[link[0] for link in rays])  # path-like, ray dicts alone
        npt.assert_allclose(shift3, shift2, atol=0, rtol=0)

    def test_input_is_not_modified(self):
        L2 = self.L2
        rays = quadriga_lib.layout.generate_rays(L2)
        before_layout, before_rays = copy.deepcopy(L2), copy.deepcopy(rays)
        quadriga_lib.layout.export_rays_qrt(self.q1, L2, rays=rays)
        quadriga_lib.layout.export_rays_qrt(self.q2, L2)
        self.assertTrue(same(L2, before_layout))
        self.assertTrue(same(rays, before_rays))

    def test_errors(self):
        L2 = self.L2
        with self.assertRaisesRegex(TypeError, "incompatible function arguments"):  # rays is keyword-only
            quadriga_lib.layout.export_rays_qrt(self.q1, L2, [])
        with self.assertRaises(TypeError):
            quadriga_lib.layout.export_rays_qrt(self.q1)
        with self.assertRaisesRegex(TypeError, "Input 'fn' must be a str or path-like object"):
            quadriga_lib.layout.export_rays_qrt(1, L2)
        with self.assertRaises(TypeError):
            quadriga_lib.layout.export_rays_qrt(self.q1, 1)
        with self.assertRaisesRegex(ValueError, "Input 'rays' must be a list"):
            quadriga_lib.layout.export_rays_qrt(self.q1, L2, rays=5)
        with self.assertRaisesRegex(ValueError, "init_link_states"):  # C++: no links
            quadriga_lib.layout.export_rays_qrt(self.q1, self.L)
        two = dict(self.L, center_freq=np.array([2e9, 3.5e9]))
        two = quadriga_lib.layout.init_link_states(two, seed=5)
        with self.assertRaisesRegex(ValueError, "one frequency"):  # C++
            quadriga_lib.layout.export_rays_qrt(self.q1, two)


if __name__ == "__main__":
    unittest.main()
