# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, "../../lib")
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

# obj_file_read output order:
#   mesh, vert_list, face_ind, obj_ind, obj_names,
#   mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop

_EM_DEFAULTS = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "att": 0.0,
                "attB": 0.0, "alpha": 0.0, "alphaB": 0.0, "fRef": 1.0}
_EM_ORDER = ["a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"]


def prop_at(csv_prop, key, idx):
    """Value of column 'key' for the 1-based material index 'idx', per-column default if absent.
    idx == 0 means 'no material' (outside) -> transparent defaults."""
    if idx == 0:
        return _EM_DEFAULTS.get(key, 0.0)
    arr_idx = idx - 1
    if key in csv_prop:
        return float(csv_prop[key][arr_idx])
    return _EM_DEFAULTS.get(key, 0.0)


def em_row(csv_prop, idx):
    """9-element EM row {a,b,c,d,att,attB,alpha,alphaB,fRef} for material index 'idx'."""
    return np.array([prop_at(csv_prop, k, idx) for k in _EM_ORDER], dtype=float)


def _rm(fn):
    if os.path.isfile(fn):
        os.remove(fn)

def read_lines(fn):
    """Read all lines of a text file, newlines stripped."""
    with open(fn) as f:
        return [line.rstrip("\n") for line in f]

def cube_vertices():
    """8 vertices of a unit cube."""
    return np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, -1.0, -1.0],
        ]
    )


def cube_faces():
    """12 triangular faces, 0-based indices into cube_vertices()."""
    return (
        np.array(
            [
                [5, 3, 1],
                [3, 8, 4],
                [7, 6, 8],
                [2, 8, 6],
                [1, 4, 2],
                [5, 2, 6],
                [5, 7, 3],
                [3, 7, 8],
                [7, 5, 6],
                [2, 4, 8],
                [1, 3, 4],
                [5, 1, 2],
            ],
            dtype=np.uint64,
        )
        - 1
    )


def make_mesh(V, F):
    """Assemble a (n_face, 9) mesh from a vertex list and 0-based face indices."""
    return np.hstack([V[F[:, 0], :], V[F[:, 1], :], V[F[:, 2], :]])


class test_case(unittest.TestCase):

    def test_mesh_roundtrip(self):
        fn = "cube.obj"
        mtl_fn = "cube.mtl"
        _rm(fn)
        _rm(mtl_fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)

        # Write from mesh; no objects, no materials
        vlo, fio = quadriga_lib.RTtools.obj_file_write(fn, mesh)

        assert vlo.shape == (8, 3)
        assert fio.shape == (12, 3)
        npt.assert_(vlo.dtype == np.float64)
        npt.assert_(fio.dtype == np.int64)
        npt.assert_almost_equal(make_mesh(vlo, fio), mesh, decimal=12)
        npt.assert_(not os.path.isfile(mtl_fn))  # no materials -> no .mtl

        (
            mesh_rd,
            vert_list_rd,
            face_ind_rd,
            obj_ind_rd,
            obj_names_rd,
            mtl_ind_rd,
            mtl_names_rd,
            _,
            csv_ind_rd,
            _,
            _,
        ) = quadriga_lib.RTtools.obj_file_read(fn)

        assert vert_list_rd.shape == (8, 3)
        npt.assert_almost_equal(mesh_rd, mesh, decimal=12)
        npt.assert_almost_equal(make_mesh(vert_list_rd, face_ind_rd), mesh, decimal=12)
        assert len(obj_names_rd) == 1
        npt.assert_equal(obj_names_rd[0], "object")
        # No usemtl written -> no materials on read-back (no synthetic "default")
        assert len(mtl_names_rd) == 0
        npt.assert_(np.all(obj_ind_rd == 0))
        npt.assert_(np.all(mtl_ind_rd == 0))
        npt.assert_(np.all(csv_ind_rd == 0))  # no material

        _rm(fn)

    def test_vertlist_faceind_roundtrip(self):
        fn = "cube.obj"
        _rm(fn)

        V = cube_vertices()
        F = cube_faces()  # 0-based
        mesh = make_mesh(V, F)

        # Write directly from vertex list + face indices (no mesh)
        vlo, fio = quadriga_lib.RTtools.obj_file_write(fn, vert_list=V, face_ind=F)

        # In this mode the outputs are exact copies of the inputs (0-based)
        npt.assert_almost_equal(vlo, V, decimal=14)
        npt.assert_array_equal(fio, F)

        _, vert_list_rd, face_ind_rd, _, _, _, _, _, _, _, _ = (
            quadriga_lib.RTtools.obj_file_read(fn)
        )
        npt.assert_almost_equal(
            make_mesh(vert_list_rd, face_ind_rd), mesh, decimal=12
        )
        _rm(fn)

    def test_materials_roundtrip(self):
        fn = "cube.obj"
        mtl_fn = "cube.mtl"
        _rm(fn)
        _rm(mtl_fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)
        obj_ind = np.zeros(12, dtype=np.uint64)        # 0-based, single object
        mtl_ind = np.ones(12, dtype=np.uint64)
        mtl_ind[4:] = 2  # faces 0-3 concrete (1), 4-11 wood (2)
        obj_names = ["Cube"]
        mtl_names = ["itu_concrete", "itu_wood"]

        quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names
        )
        npt.assert_(os.path.isfile(mtl_fn))

        # Resolve EM properties from the built-in default table (names are ITU materials)
        (_, _, _, _, _, mtl_ind_rd, mtl_names_rd, _, csv_ind_rd, _, csv_prop_rd) = (
            quadriga_lib.RTtools.obj_file_read(fn)
        )

        assert len(mtl_names_rd) == 2
        npt.assert_equal(mtl_names_rd[0], "itu_concrete")
        npt.assert_equal(mtl_names_rd[1], "itu_wood")
        npt.assert_almost_equal(
            em_row(csv_prop_rd, csv_ind_rd[0]),
            [5.24, 0.0, 0.0462, 0.7822, 0, 0, 0, 0, 1], decimal=3
        )
        npt.assert_almost_equal(
            em_row(csv_prop_rd, csv_ind_rd[4]),
            [1.99, 0.0, 0.0047, 1.0718, 0, 0, 0, 0, 1], decimal=3
        )
        npt.assert_array_equal(mtl_ind_rd, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])

        _rm(fn)
        _rm(mtl_fn)

    def test_custom_material_csv(self):
        fn = "cube.obj"
        mtl_fn = "cube.mtl"
        csv_fn = "custom_materials.csv"
        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)
        obj_ind = np.zeros(12, dtype=np.uint64)
        mtl_ind = np.ones(12, dtype=np.uint64)

        quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, mtl_ind, ["Cube"], ["glass"]
        )

        # EM properties come from a CSV, not from the OBJ/MTL
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att\n")
            f.write("air,1.0,0.0,0.0,0.0,0.0\n")
            f.write("glass,6.0,0.0,0.1,1.2,0.0\n")

        (_, _, _, _, _, mtl_ind_rd, mtl_names_rd, _, csv_ind_rd, _, csv_prop_rd) = (
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)
        )
        npt.assert_equal(mtl_names_rd[0], "glass")
        npt.assert_almost_equal(
            em_row(csv_prop_rd, csv_ind_rd[0]),
            [6.0, 0.0, 0.1, 1.2, 0, 0, 0, 0, 1], decimal=12
        )
        npt.assert_(np.all(mtl_ind_rd == 1))

        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_fn)

    def test_csv_table_roundtrip(self):
        fn = "cube.obj"
        mtl_fn = "cube.mtl"
        csv_obj_fn = "cube.csv"  # companion CSV written next to the .obj
        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_obj_fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)
        obj_ind = np.zeros(12, dtype=np.uint64)
        mtl_ind = np.ones(12, dtype=np.uint64)
        mtl_ind[4:] = 2  # faces 0-3 concrete (1), 4-11 wood (2)
        obj_names = ["Cube"]
        mtl_names = ["concrete", "wood"]  # usemtl names must match csv_names

        csv_ind = np.ones(12, dtype=np.uint64)
        csv_ind[4:] = 2  # 1-based, same split as mtl_ind
        csv_names = ["concrete", "wood"]
        csv_prop = {
            "a":    np.array([5.24, 1.99]),
            "c":    np.array([0.0462, 0.0047]),
            "d":    np.array([0.7822, 1.0718]),
            "fRef": np.array([1.0, 1.0]),
        }

        quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names,
            csv_ind=csv_ind, csv_names=csv_names, csv_prop=csv_prop,
            csv_write_defaults=False,
        )

        npt.assert_(os.path.isfile(fn))
        npt.assert_(os.path.isfile(mtl_fn))
        npt.assert_(os.path.isfile(csv_obj_fn))  # companion .csv named after the .obj

        (_, _, _, _, _, mtl_ind_rd, _, _, csv_ind_rd, csv_names_rd, csv_prop_rd) = (
            quadriga_lib.RTtools.obj_file_read(fn, csv_obj_fn)
        )

        assert len(csv_names_rd) == 2
        npt.assert_equal(csv_names_rd[0], "concrete")
        npt.assert_equal(csv_names_rd[1], "wood")

        # csv_ind is 1-based; prop_at maps it onto the table (helper handles the -1)
        npt.assert_almost_equal(prop_at(csv_prop_rd, "a", csv_ind_rd[0]), 5.24, decimal=2)
        npt.assert_almost_equal(prop_at(csv_prop_rd, "a", csv_ind_rd[4]), 1.99, decimal=2)
        npt.assert_almost_equal(prop_at(csv_prop_rd, "d", csv_ind_rd[0]), 0.7822, decimal=3)

        # Visual side round-trips 1-based
        npt.assert_array_equal(mtl_ind_rd, [1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2])

        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_obj_fn)

    def test_csv_columns_defaults_validation(self):
        fn = "cube.obj"
        mtl_fn = "cube.mtl"
        csv_obj_fn = "cube.csv"
        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_obj_fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)
        obj_ind = np.zeros(12, dtype=np.uint64)
        mtl_ind = np.ones(12, dtype=np.uint64)
        obj_names = ["Cube"]
        mtl_names = ["slab"]

        csv_names = ["slab"]
        csv_prop = {"c": np.array([0.05]), "tf": np.array([2.0]), "zzz": np.array([7.0])}

        # csv_write_defaults = False -> only present columns (canonical order, then extras)
        quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names,
            csv_names=csv_names, csv_prop=csv_prop, csv_write_defaults=False,
        )
        npt.assert_(os.path.isfile(csv_obj_fn))

        lines = read_lines(csv_obj_fn)
        npt.assert_equal(lines[0], "name,c,tf,zzz")
        npt.assert_equal(lines[1], "slab,0.05,2,7")
        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_obj_fn)

        # csv_write_defaults = True -> full canonical set with defaults (a, e, fRef = 1, else 0)
        quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names,
            csv_names=csv_names, csv_prop=csv_prop, csv_write_defaults=True,
        )
        lines = read_lines(csv_obj_fn)
        npt.assert_equal(
            lines[0],
            "name,a,b,c,d,e,f,g,h,att,attB,alpha,alphaB,fRef,m,resF,resQ,resS,coiF,coiQ,coiA,tf,tfB,zzz",
        )
        npt.assert_equal(
            lines[1],
            "slab,1,0,0.05,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,7",
        )
        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_obj_fn)

        # Validation: csv_prop column length must match len(csv_names)
        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_write(
                fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names,
                csv_names=csv_names, csv_prop={"a": np.array([1.0, 2.0])},
                csv_write_defaults=False,
            )

        # Validation: csv_ind out of range (only 1 material in csv_names)
        with self.assertRaises(ValueError):
            csv_ind_bad = np.ones(12, dtype=np.uint64)
            csv_ind_bad[0] = 5
            quadriga_lib.RTtools.obj_file_write(
                fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names,
                csv_ind=csv_ind_bad, csv_names=csv_names, csv_prop=csv_prop,
                csv_write_defaults=False,
            )

        # Validation: csv inputs without csv_names
        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_write(
                fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names,
                csv_prop=csv_prop, csv_write_defaults=False,
            )

        _rm(fn)
        _rm(mtl_fn)
        _rm(csv_obj_fn)

    def test_bsdf_roundtrip(self):
        fn = "cube.obj"
        mtl_fn = "cube.mtl"
        _rm(fn)
        _rm(mtl_fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)
        obj_ind = np.zeros(12, dtype=np.uint64)
        mtl_ind = np.ones(12, dtype=np.uint64)

        # Distinct non-default values; clamped fields inside [0, 1], ior in a sane range
        bsdf = np.array(
            [
                [
                    0.1, 0.2, 0.3,        # base color RGB
                    0.7,                  # transparency
                    0.4,                  # roughness
                    0.6,                  # metallic
                    1.7,                  # ior
                    0.8,                  # specular
                    0.05, 0.15, 0.25,     # emission RGB
                    0.3,                  # sheen
                    0.35,                 # clearcoat
                    0.45,                 # clearcoat roughness
                    0.55,                 # anisotropic
                    0.65,                 # anisotropic rotation
                    0.9,                  # transmission
                ]
            ]
        )

        quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, mtl_ind, ["Cube"], ["painted"], bsdf=bsdf
        )
        npt.assert_(os.path.isfile(mtl_fn))

        (_, _, _, _, _, _, mtl_names_rd, bsdf_rd, _, _, _) = (
            quadriga_lib.RTtools.obj_file_read(fn)
        )
        npt.assert_equal(mtl_names_rd[0], "painted")
        assert bsdf_rd.shape == (1, 17)
        npt.assert_almost_equal(bsdf_rd, bsdf, decimal=9)

        _rm(fn)
        _rm(mtl_fn)

    def test_multiple_objects(self):
        fn = "cube.obj"
        _rm(fn)

        V = cube_vertices()
        F = cube_faces()
        meshA = make_mesh(V, F)
        meshB = meshA.copy()
        meshB[:, [0, 3, 6]] += 10.0  # shift x of all corners -> disjoint cube
        mesh = np.vstack([meshA, meshB])  # (24, 9)
        obj_ind = np.concatenate(
            [np.zeros(12, dtype=np.uint64), np.ones(12, dtype=np.uint64)]
        )  # 0-based
        obj_names = ["CubeA", "CubeB"]

        vlo, _ = quadriga_lib.RTtools.obj_file_write(
            fn, mesh, obj_ind, obj_names=obj_names
        )
        assert vlo.shape == (16, 3)  # no cross-object merging -> 8 + 8

        (_, vert_list_rd, face_ind_rd, obj_ind_rd, obj_names_rd, _, _, _, _, _, _) = (
            quadriga_lib.RTtools.obj_file_read(fn)
        )

        assert vert_list_rd.shape == (16, 3)
        assert len(obj_names_rd) == 2
        npt.assert_equal(obj_names_rd[0], "CubeA")
        npt.assert_equal(obj_names_rd[1], "CubeB")
        npt.assert_array_equal(
            obj_ind_rd,
            np.concatenate(
                [np.zeros(12, dtype=np.int64), np.ones(12, dtype=np.int64)]
            ),
        )
        npt.assert_almost_equal(
            make_mesh(vert_list_rd, face_ind_rd), mesh, decimal=12
        )
        _rm(fn)

    def test_outputs_only(self):
        # Empty filename: derive vert_list / face_ind from mesh, write no file
        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)

        vlo, fio = quadriga_lib.RTtools.obj_file_write("", mesh)
        assert vlo.shape == (8, 3)
        assert fio.shape == (12, 3)
        npt.assert_almost_equal(make_mesh(vlo, fio), mesh, decimal=12)

    def test_errors(self):
        fn = "cube.obj"
        _rm(fn)

        V = cube_vertices()
        F = cube_faces()
        mesh = make_mesh(V, F)

        # Both mesh and vert_list / face_ind given
        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_write(fn, mesh, vert_list=V, face_ind=F)

        # Neither geometry source given
        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_write(fn)

        # File name does not end in .obj
        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_write("cube.txt", mesh)

        # Non-contiguous obj_ind: {0,0,1,1,0,...}
        with self.assertRaises(ValueError):
            obj_bad = np.zeros(12, dtype=np.uint64)
            obj_bad[2] = 1
            obj_bad[3] = 1
            quadriga_lib.RTtools.obj_file_write(
                fn, mesh, obj_bad, obj_names=["A", "B"]
            )

        # obj_names too short for obj_ind (0-based: max index 1 needs 2 names)
        with self.assertRaises(ValueError):
            obj_ind = np.concatenate(
                [np.zeros(6, dtype=np.uint64), np.ones(6, dtype=np.uint64)]
            )
            quadriga_lib.RTtools.obj_file_write(
                fn, mesh, obj_ind, obj_names=["OnlyOne"]
            )

        # mtl_names too short for mtl_ind (1-based: max index 2 needs 2 names)
        with self.assertRaises(ValueError):
            mtl_ind = np.concatenate( [np.ones(6, dtype=np.uint64), 2 * np.ones(6, dtype=np.uint64)] )
            quadriga_lib.RTtools.obj_file_write( fn, mesh, mtl_ind=mtl_ind, mtl_names=["OnlyOne"] )

        # bsdf given without materials
        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_write(fn, mesh, bsdf=np.zeros((1, 17)))

        # None of the error cases should have produced a file
        npt.assert_(not os.path.isfile(fn))
        npt.assert_(not os.path.isfile("cube.txt"))