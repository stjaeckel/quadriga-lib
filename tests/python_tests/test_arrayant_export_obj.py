# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python port of test_arrayant_export_obj_file.m.
# MEX-only cases are dropped: argument-count / nargout errors, the 1-based `i_element = 0`
# violation, and the 1-based `freq = 0` error (freq_ind = 0 is the valid default in Python).
# The multi-frequency struct array is replaced by a 4D pattern dict selected via `freq_ind`.

import sys
import os
import shutil
import tempfile
import unittest
import numpy as np
import numpy.testing as npt
from collections import namedtuple

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

# pybind11 maps std::invalid_argument -> ValueError and std::runtime_error -> RuntimeError;
# a non-dict arrayant fails the dict conversion with TypeError. Accept any for error cases.
ANY_ERR = (ValueError, TypeError, RuntimeError, OSError)

# All 11 colormaps supported by export_obj_file
COLORMAPS = (
    "jet",
    "parula",
    "winter",
    "hot",
    "turbo",
    "copper",
    "spring",
    "cool",
    "gray",
    "autumn",
    "summer",
)


def _copy_ant(ant):
    """Deep copy of an arrayant dict (numpy fields copied, scalars/strings kept)."""
    return {
        k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in ant.items()
    }


def _azimuth_taper(ant):
    """Return a copy with an azimuth-dependent gain taper — reshapes the directivity pattern."""
    out = _copy_ant(ant)
    az = np.asarray(ant["azimuth_grid"], dtype=float)
    w = (1.0 + 0.9 * np.cos(az))[None, :, None]  # front-directional weight
    for key in ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im"):
        out[key] = np.asarray(ant[key], dtype=float) * w
    return out


def _stack_two(ant_a, ant_b):
    """Stack two single-frequency arrayant dicts into one 2-frequency dict (4D pattern fields)."""
    mf = {}
    for key in ("e_theta_re", "e_theta_im", "e_phi_re", "e_phi_im"):
        mf[key] = np.stack(
            [np.asarray(ant_a[key], dtype=float), np.asarray(ant_b[key], dtype=float)],
            axis=3,
        )
    for key in (
        "azimuth_grid",
        "elevation_grid",
        "element_pos",
        "coupling_re",
        "coupling_im",
    ):
        if key in ant_a:
            mf[key] = np.asarray(ant_a[key], dtype=float)
    mf["center_freq"] = np.array([1.0e9, 2.0e9])
    return mf


class test_case(unittest.TestCase):

    def setUp(self):
        # Export into a private temp directory so nothing leaks into the working dir.
        self.tmpdir = tempfile.mkdtemp()
        self._counter = 0

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _obj_path(self):
        self._counter += 1
        return os.path.join(self.tmpdir, "ant_%d.obj" % self._counter)

    def _export(self, ant, **kwargs):
        """Export an antenna to a fresh .obj path and return that path."""
        p = self._obj_path()
        quadriga_lib.arrayant.export_obj_file(p, ant, **kwargs)
        return p

    def _read(self, obj_path):
        ObjData = namedtuple('ObjData', 'mesh vert_list face_ind obj_ind obj_names '
                                        'mtl_ind mtl_names bsdf csv_ind csv_names csv_prop')
        return ObjData(*quadriga_lib.RTtools.obj_file_read(obj_path))

    # ---- Basic file creation and content ----

    def test_basic_file_creation(self):
        ant = quadriga_lib.arrayant.generate("xpol")  # 2 elements
        p = self._export(ant)

        # Both the .obj and the companion .mtl must be written.
        self.assertTrue(os.path.exists(p))
        self.assertTrue(os.path.exists(p[:-4] + ".mtl"))

        mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, *_ = (
            self._read(p)
        )

        n_tri = mesh.shape[0]
        self.assertGreater(n_tri, 0)
        self.assertEqual(mesh.shape[1], 9)  # {X1..Z3} per triangle
        self.assertEqual(vert_list.shape[1], 3)  # xyz columns
        self.assertEqual(face_ind.shape, (n_tri, 3))  # 3 vertex indices per face
        self.assertEqual(obj_ind.size, n_tri)
        self.assertEqual(mtl_ind.size, n_tri)

        # Face indices reference valid vertices (range check holds for 0- and 1-based indexing).
        n_vert = vert_list.shape[0]
        self.assertGreaterEqual(int(face_ind.min()), 0)
        self.assertLessEqual(int(face_ind.max()), n_vert)

        # xpol has 2 elements -> 2 distinct objects.
        self.assertEqual(np.unique(obj_ind).size, 2)
        self.assertEqual(len(obj_names), 2)

    # ---- icosphere_n_div ----

    def test_icosphere_subdivision(self):
        ant = quadriga_lib.arrayant.generate("3gpp", res=10)  # 1 element

        n_low = self._read(self._export(ant, icosphere_n_div=1))[0].shape[0]
        n_high = self._read(self._export(ant, icosphere_n_div=3))[0].shape[0]

        # Each subdivision step roughly quadruples the triangle count (1 -> 3 gives ~16x).
        self.assertGreater(n_high, n_low)
        self.assertGreater(n_high, 4 * n_low)

    # ---- object_radius ----

    def test_object_radius_scaling(self):
        ant = quadriga_lib.arrayant.generate("3gpp", res=10)

        v1 = self._read(self._export(ant, object_radius=1.0, icosphere_n_div=2))[1]
        v3 = self._read(self._export(ant, object_radius=3.0, icosphere_n_div=2))[1]

        # object_radius scales the mesh linearly.
        npt.assert_allclose(np.abs(v3).max() / np.abs(v1).max(), 3.0, rtol=0.05)

    # ---- element selection ----

    def test_element_selection(self):
        ant = quadriga_lib.arrayant.generate("xpol")  # 2 elements

        # element omitted / None -> export all elements.
        obj_all = self._read(self._export(ant, object_radius=1.0, icosphere_n_div=2))[3]
        obj_one = self._read(
            self._export(ant, object_radius=1.0, icosphere_n_div=2, element=[0])
        )[3]

        self.assertEqual(np.unique(obj_all).size, 2)
        self.assertEqual(np.unique(obj_one).size, 1)
        self.assertLess(obj_one.size, obj_all.size)

    # ---- colormap ----

    def test_colormap(self):
        ant = quadriga_lib.arrayant.generate("3gpp", res=10)
        for cmap in COLORMAPS:
            p = self._export(ant, colormap=cmap, object_radius=1.0, icosphere_n_div=2)
            self.assertTrue(os.path.exists(p))
            self.assertTrue(os.path.exists(p[:-4] + ".mtl"))

    # ---- directivity_range ----

    def test_directivity_range(self):
        ant = quadriga_lib.arrayant.generate("3gpp", res=10)
        for drange in (10.0, 30.0, 50.0):
            p = self._export(
                ant, directivity_range=drange, object_radius=1.0, icosphere_n_div=2
            )
            self.assertTrue(os.path.exists(p))

    # ---- multi-frequency dispatch via freq_ind ----

    def test_multifreq(self):
        base = quadriga_lib.arrayant.generate("3gpp", res=10)  # entry 0
        ent_b = _azimuth_taper(base)  # entry 1: different pattern shape
        mf = _stack_two(base, ent_b)

        kw = dict(object_radius=1.0, icosphere_n_div=2)

        # freq_ind selects which frequency entry is exported.
        v0 = self._read(self._export(mf, freq_ind=0, **kw))[1]
        v1 = self._read(self._export(mf, freq_ind=1, **kw))[1]
        self.assertFalse(np.array_equal(v0, v1))  # patterns differ -> geometry differs

        # Each multi-freq export matches its direct single-frequency counterpart.
        v_a = self._read(self._export(base, **kw))[1]
        v_b = self._read(self._export(ent_b, **kw))[1]
        npt.assert_allclose(v0, v_a, atol=1e-9, rtol=0)
        npt.assert_allclose(v1, v_b, atol=1e-9, rtol=0)

        # Omitted freq_ind defaults to 0.
        v_default = self._read(self._export(mf, **kw))[1]
        npt.assert_allclose(v_default, v0, atol=1e-9, rtol=0)

    # ---- error handling ----

    def test_errors(self):
        base = quadriga_lib.arrayant.generate("3gpp", res=10)
        mf = _stack_two(base, _azimuth_taper(base))

        # arrayant must be a dict.
        with self.assertRaises(ANY_ERR):
            quadriga_lib.arrayant.export_obj_file(self._obj_path(), 1.0)

        # freq_ind out of bound for a 2-entry frequency-dependent dict.
        with self.assertRaises(ANY_ERR):
            quadriga_lib.arrayant.export_obj_file(self._obj_path(), mf, freq_ind=5)

        # freq_ind out of bound for a single-frequency dict (only 0 is valid).
        with self.assertRaises(ANY_ERR):
            quadriga_lib.arrayant.export_obj_file(self._obj_path(), base, freq_ind=1)

        # Filename must end in .obj.
        with self.assertRaises(ANY_ERR):
            quadriga_lib.arrayant.export_obj_file(
                os.path.join(self.tmpdir, "ant.txt"), base
            )

        # Filename must not be empty.
        with self.assertRaises(ANY_ERR):
            quadriga_lib.arrayant.export_obj_file("", base)


if __name__ == "__main__":
    unittest.main()
