# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for the argument passing of all wrappers.
# Scope: optional arguments are keyword-only, except the few listed below that read clearly by position.
# A new optional argument goes after the "*" marker of its registration; this test fails if it lands before it.

import sys
import os
import unittest

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

# Optional arguments that may be passed by position: the data a call works on or its address
POSITIONAL_OPTIONAL = {
    "RTtools.cube": ("scale", "rotation", "location"),
    "RTtools.plane": ("scale", "rotation", "location"),
    "RTtools.mitsuba_xml_file_write": ("bsdf",),
    "RTtools.obj_file_read": ("fn", "fn_csv"),
    "RTtools.obj_file_write": ("fn", "mesh", "obj_ind", "mtl_ind", "obj_names", "mtl_names"),
    "RTtools.point_cloud_aabb": ("sub_cloud_ind",),
    "RTtools.point_inside_mesh": ("obj_ind",),
    "RTtools.ray_point_intersect": ("sub_cloud_ind",),
    "RTtools.ray_triangle_intersect": ("sub_mesh_ind", "aabb"),
    "RTtools.triangle_mesh_aabb": ("sub_mesh_ind",),
    "arrayant.generate": ("res",),
    "arrayant.generate_speaker": ("driver_type",),
    "arrayant.qdant_read": ("id",),
    "arrayant.qdant_write": ("id",),
    "channel.baseband_freq_response": ("coeff", "delay"),
    "channel.hdf5_create_file": ("nx", "ny", "nz", "nw"),
    "channel.hdf5_read_channel": ("ix", "iy", "iz", "iw"),
    "channel.hdf5_read_dset": ("ix", "iy", "iz", "iw"),
    "channel.hdf5_read_dset_names": ("ix", "iy", "iz", "iw"),
    "channel.hdf5_reshape_layout": ("ny", "nz", "nw"),
    "channel.hdf5_write_channel": ("par", "ix", "iy", "iz", "iw"),
    "channel.hdf5_write_dset": ("ix", "iy", "iz", "iw"),
    "channel.ieee_indoor": ("ap_array", "sta_array"),
    "channel.irs": ("center_freq",),
    "channel.planar": ("center_freq",),
    "channel.spherical": ("center_freq",),
    "channel.qrt_file_read": ("i_cir", "i_orig"),
    "gbsm.conf_file_write": ("plpar",),
    "gbsm.scenpar_validate": ("scenpar_in",),
    "sos.preset": ("dist_decorr",),
    "sos.val": ("pos_b",),
    "sos.rand": ("pos_b",),
    "sos.randn": ("pos_b",),
    "sos.randi": ("pos_b",),
    "sos.remove_bias": ("pos_b",),
    "sos.map": ("y", "z"),
    "sos.generate": ("n_dim",),
    "tools.acdf": ("bins",),
    "tools.fast_cart2geo": ("y", "z"),
    "tools.fast_geo2cart": ("el", "len"),
}

SUBMODULES = sorted(n for n in dir(quadriga_lib) if type(getattr(quadriga_lib, n)) is type(quadriga_lib))


def split_top_level(text):
    # Split at commas outside brackets and quotes
    parts, depth, quote, cur = [], 0, None, ""
    for ch in text:
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(cur.strip())
            cur = ""
            continue
        cur += ch
    if cur.strip():
        parts.append(cur.strip())
    return parts


def parameters(func):
    # (name, has_default, keyword_only) for each parameter, from the pybind11 signature line
    line = func.__doc__.strip().split("\n")[0]
    inner = line[line.index("(") + 1:line.rindex(") ->")]
    out, kw_only = [], False
    for p in split_top_level(inner):
        if p == "*":
            kw_only = True
            continue
        name = p.split(":")[0].split("=")[0].strip()
        out.append((name, "=" in p.split(":", 1)[-1], kw_only))
    return out


def all_functions():
    for sub in SUBMODULES:
        mod = getattr(quadriga_lib, sub)
        for name in sorted(dir(mod)):
            func = getattr(mod, name)
            if callable(func) and not name.startswith("_"):
                yield f"{sub}.{name}", func


class test_case(unittest.TestCase):

    def test_optional_arguments_are_keyword_only(self):
        n_func = 0
        for key, func in all_functions():
            n_func += 1
            with self.subTest(function=key):
                params = parameters(func)
                positional_optional = tuple(n for n, has_default, kw in params if has_default and not kw)
                self.assertEqual(positional_optional, POSITIONAL_OPTIONAL.get(key, ()))
        self.assertGreater(n_func, 0)

    def test_allow_list_names_registered_functions(self):
        names = dict(all_functions())
        for key in POSITIONAL_OPTIONAL:
            self.assertIn(key, names)

    def test_positional_optional_raises(self):
        # An optional argument passed by position is a TypeError naming the call
        with self.assertRaises(TypeError):
            quadriga_lib.tools.calc_angular_spread([[0.1]], [[0.0]], [[1.0]], True)
        with self.assertRaises(TypeError):
            quadriga_lib.arrayant.calc_directivity(quadriga_lib.arrayant.generate("omni"), [0])


if __name__ == "__main__":
    unittest.main()
