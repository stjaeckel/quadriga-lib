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

# Per-column defaults for absent EM columns (mirrors the consumer-side defaults).
_EM_DEFAULTS = {"a": 1.0, "b": 0.0, "c": 0.0, "d": 0.0, "att": 0.0,
                "attB": 0.0, "alpha": 0.0, "alphaB": 0.0, "fRef": 1.0}
_EM_ORDER = ["a", "b", "c", "d", "att", "attB", "alpha", "alphaB", "fRef"]


def prop_at(csv_prop, key, idx):
    """Read the value of column 'key' for material index 'idx' from the csv_prop dict,
    applying the documented per-column default when the column is absent."""
    if key in csv_prop:
        return float(csv_prop[key][idx])
    return _EM_DEFAULTS.get(key, 0.0)


def em_row(csv_prop, idx):
    """Assemble the standard 9-element EM row {a,b,c,d,att,attB,alpha,alphaB,fRef}
    for material index 'idx', filling absent columns with their defaults."""
    return np.array([prop_at(csv_prop, k, idx) for k in _EM_ORDER], dtype=float)


def create_cube_with_materials(obj_file, mtl1, mtl2=None):
    """Helper to create a cube OBJ file with specified materials"""
    with open(obj_file, "w") as f:
        f.write("o Cube\n")
        f.write("v 1.0 1.0 1.0\n")
        f.write("v 1.0 1.0 -1.0\n")
        f.write("v 1.0 -1.0 1.0\n")
        f.write("v 1.0 -1.0 -1.0\n")
        f.write("v -1.0 1.0 1.0\n")
        f.write("v -1.0 1.0 -1.0\n")
        f.write("v -1.0 -1.0 1.0\n")
        f.write("v -1.0 -1.0 -1.0\n")
        f.write(f"usemtl {mtl1}\n")
        f.write("f 5 3 1\n")
        f.write("f 3 8 4\n")
        f.write("f 7 6 8\n")
        f.write("f 2 8 6\n")
        if mtl2:
            f.write(f"usemtl {mtl2}\n")
        f.write("f 1 4 2\n")
        f.write("f 5 2 6\n")
        f.write("f 5 7 3\n")
        f.write("f 3 7 8\n")
        f.write("f 7 5 6\n")
        f.write("f 2 4 8\n")
        f.write("f 1 3 4\n")
        f.write("f 5 1 2\n")


def cleanup(fn, csv_fn):
    if os.path.isfile(fn):
        os.remove(fn)
    if os.path.isfile(csv_fn):
        os.remove(csv_fn)


class test_case(unittest.TestCase):

    def test(self):

        fn = "cube.obj"
        if os.path.isfile(fn):
            os.remove(fn)

        with open(fn, "w") as f:
            f.write("# A very nice, but useless comment ;-)\n")
            f.write("o Cube\n")
            f.write("v 1.0 1.0 1.0\n")
            f.write("v 1.0 1.0 -1.0\n")
            f.write("v 1.0 -1.0 1.0\n")
            f.write("v 1.0 -1.0 -1.0\n")
            f.write("v -1.0 1.0 1.0\n")
            f.write("v -1.0 1.0 -1.0\n")
            f.write("v -1.0 -1.0 1.0\n")
            f.write("v -1.0 -1.0 -1.0\n")
            f.write("s 0\n")
            f.write("f 5 3 1\n")
            f.write("f 3 8 4\n")
            f.write("f 7 6 8\n")
            f.write("f 2 8 6\n")
            f.write("f 1 4 2\n")
            f.write("f 5 2 6\n")
            f.write("f 5 7 3\n")
            f.write("f 3 7 8\n")
            f.write("f 7 5 6\n")
            f.write("f 2 4 8\n")
            f.write("f 1 3 4\n")
            f.write("f 5 1 2\n")

        vert_list_correct = np.array(
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

        face_ind_correct = (
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
                dtype=int,
            )
            - 1
        )  # zero-based indexing

        mesh_correct = np.hstack(
            [
                vert_list_correct[face_ind_correct[:, 0], :],
                vert_list_correct[face_ind_correct[:, 1], :],
                vert_list_correct[face_ind_correct[:, 2], :],
            ]
        )

        # No output should be fine
        quadriga_lib.RTtools.obj_file_read(fn)

        # Read all
        (
            mesh,
            vert_list,
            face_ind,
            obj_ind,
            obj_names,
            mtl_ind,
            mtl_names,
            bsdf,
            csv_ind,
            csv_names,
            csv_prop,
        ) = quadriga_lib.RTtools.obj_file_read(fn)

        assert mesh.shape == (12, 9)
        assert vert_list.shape == (8, 3)
        assert face_ind.shape == (12, 3)
        assert obj_ind.shape == (12,)
        assert mtl_ind.shape == (12,)
        assert csv_ind.shape == (12,)
        assert len(obj_names) == 1
        # No usemtl -> faces get the synthetic "default" material on the .mtl side
        assert len(mtl_names) == 1
        assert mtl_names[0] == "default"
        assert bsdf.shape == (0, 17)
        # csv side is the full default table with row 0 = air
        assert len(csv_names) > 1
        assert csv_names[0] == "air"
        assert isinstance(csv_prop, dict)

        npt.assert_(mesh.dtype == np.float64)
        npt.assert_(vert_list.dtype == np.float64)
        npt.assert_(face_ind.dtype == np.int64)
        npt.assert_(obj_ind.dtype == np.int64)
        npt.assert_(mtl_ind.dtype == np.int64)
        npt.assert_(csv_ind.dtype == np.int64)

        npt.assert_almost_equal(vert_list, vert_list_correct, decimal=14)
        npt.assert_array_equal(face_ind, face_ind_correct)
        npt.assert_almost_equal(mesh, mesh_correct, decimal=14)

        # 0-based indices; "default" not in table -> air fallback (row 0)
        npt.assert_(np.all(obj_ind == 0))
        npt.assert_(np.all(mtl_ind == 0))
        npt.assert_(np.all(csv_ind == 0))
        npt.assert_equal(obj_names[0], "Cube")

        # Air at csv row 0 is transparent (a = 1)
        npt.assert_almost_equal(em_row(csv_prop, 0),
                                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], decimal=14)

        os.remove(fn)

        # Two planes, second uses a named ITU material
        with open(fn, "w") as f:
            f.write("o Plane\n")
            f.write("v -1.000000 -1.000000 0.000000\n")
            f.write("v 1.000000 -1.000000 0.000000\n")
            f.write("v -1.000000 1.000000 0.000000\n")
            f.write("v 1.000000 1.000000 0.000000\n")
            f.write("vt 1.000000 0.000000\n")
            f.write("vt 0.000000 1.000000\n")
            f.write("vt 0.000000 0.000000\n")
            f.write("vt 1.000000 1.000000\n")
            f.write("f 2/1 3/2 1/3\n")
            f.write("f 2/1 4/4 3/2\n")
            f.write("o Plane.001\n")
            f.write("v -1.000000 -1.000000 1.26\n")
            f.write("v 1.000000 -1.000000 1.26\n")
            f.write("v -1.000000 1.000000 1.26\n")
            f.write("v 1.000000 1.000000 1.26\n")
            f.write("vt 1.000000 0.000000\n")
            f.write("vt 0.000000 1.000000\n")
            f.write("vt 0.000000 0.000000\n")
            f.write("vt 1.000000 1.000000\n")
            f.write("usemtl itu_wood\n")
            f.write("f 6/5 7/6 5/7\n")
            f.write("f 6/5 8/8 7/6\n")

        (mesh, vert_list, face_ind, obj_ind, obj_names,
         mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop) = \
            quadriga_lib.RTtools.obj_file_read(fn)

        assert len(obj_names) == 2
        # Faces 0-1 have no usemtl -> synthetic "default" -> air (a = 1)
        # Faces 2-3 are itu_wood (a = 1.99)
        npt.assert_almost_equal(prop_at(csv_prop, "a", csv_ind[0]), 1.0, decimal=14)
        npt.assert_almost_equal(prop_at(csv_prop, "a", csv_ind[1]), 1.0, decimal=14)
        npt.assert_(prop_at(csv_prop, "a", csv_ind[2]) > 1.5)
        npt.assert_(prop_at(csv_prop, "a", csv_ind[3]) > 1.5)

        expected_face_ind = np.array([[2, 3, 1], [2, 4, 3], [6, 7, 5], [6, 8, 7]]) - 1
        npt.assert_array_equal(face_ind, expected_face_ind)

        npt.assert_array_equal(obj_ind, [0, 0, 1, 1])
        # mtl_names: faces 0-1 -> "default", faces 2-3 -> "itu_wood" (first-appearance order)
        npt.assert_array_equal(mtl_ind, [0, 0, 1, 1])
        npt.assert_equal(obj_names[0], "Plane")
        npt.assert_equal(obj_names[1], "Plane.001")
        npt.assert_equal(mtl_names[0], "default")
        npt.assert_equal(mtl_names[1], "itu_wood")

        # Missing file raises
        with self.assertRaises(ValueError) as context:
            quadriga_lib.RTtools.obj_file_read("bla.obj")
        self.assertEqual(
            str(context.exception), "Error opening file: 'bla.obj' does not exist."
        )

        os.remove(fn)

    def test_custom_materials_csv(self):
        """Test OBJ File Read - Custom Materials CSV"""

        fn = "cube.obj"
        csv_fn = "custom_materials.csv"

        # Basic custom materials (row 0 = air fallback)
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att\n")
            f.write("air,1.0,0.0,0.0,0.0,0.0\n")
            f.write("custom_material_1,2.5,0.0,0.001,0.5,5.0\n")
            f.write("custom_material_2,4.0,-0.1,0.05,1.2,10.0\n")

        create_cube_with_materials(fn, "custom_material_1", "custom_material_2")

        (mesh, vert_list, face_ind, obj_ind, obj_names,
         mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop) = \
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)

        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[0]),
            [2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[0], "custom_material_1")

        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[4]),
            [4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[1], "custom_material_2")
        npt.assert_(csv_ind[0] != csv_ind[4])

        # Jumbled column order (keyed by name, so order is irrelevant)
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("att,d,c,b,a,name\n")
            f.write("0.0,0.0,0.0,0.0,1.0,air\n")
            f.write("5.0,0.5,0.001,0.0,2.5,custom_material_1\n")
            f.write("10.0,1.2,0.05,-0.1,4.0,custom_material_2\n")

        create_cube_with_materials(fn, "custom_material_1", "custom_material_2")

        (mesh, vert_list, face_ind, obj_ind, obj_names,
         mtl_ind, mtl_names, bsdf, csv_ind, csv_names, csv_prop) = \
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)

        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[0]),
            [2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[0], "custom_material_1")

        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[4]),
            [4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[1], "custom_material_2")

        # Duplicate material names -> error
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att\n")
            f.write("custom_material_1,2.5,0.0,0.001,0.5,5.0\n")
            f.write("custom_material_1,4.0,-0.1,0.05,1.2,10.0\n")

        create_cube_with_materials(fn, "custom_material_1")

        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)

        # Non-existent CSV file -> error
        cleanup(fn, csv_fn)
        create_cube_with_materials(fn, "custom_material_1")

        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_read(fn, "nonexistent.csv")

        cleanup(fn, csv_fn)

        # CSV with frequency-dependent columns populated.
        # Jumbled order: fRef before alphaB.
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att,attB,alpha,fRef,alphaB\n")
            f.write("air,1.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0,0.0\n")
            f.write("lossy_wall,4.5,0.1,0.02,0.8,3.0,0.2,0.5,2.4,0.15\n")

        create_cube_with_materials(fn, "lossy_wall")

        (_, _, _, _, _, _, mtl_names, _, csv_ind, _, csv_prop) = \
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)
        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[0]),
            [4.5, 0.1, 0.02, 0.8, 3.0, 0.2, 0.5, 0.15, 2.4], decimal=14
        )
        npt.assert_equal(mtl_names[0], "lossy_wall")

        # CSV with subset of optional columns; unspecified ones default.
        # fRef given but attB/alpha/alphaB absent -> defaults.
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,c,fRef\n")
            f.write("air,1.0,0.0,1.0\n")
            f.write("partial,3.0,0.01,5.0\n")

        create_cube_with_materials(fn, "partial")

        (_, _, _, _, _, _, _, _, csv_ind, _, csv_prop) = \
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)
        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[0]),
            [3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], decimal=14
        )
        # Columns absent from the CSV are not keys of the dict
        npt.assert_("b" not in csv_prop)

        # Empty cells in optional columns parse as 0.
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att,attB,alpha,alphaB,fRef\n")
            f.write("air,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.0\n")
            f.write("sparse,2.0,,0.005,,1.5,,,,\n")

        create_cube_with_materials(fn, "sparse")

        (_, _, _, _, _, _, _, _, csv_ind, _, csv_prop) = \
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)
        # Empty cells are 0; the fRef cell is present-but-blank here -> 0
        npt.assert_almost_equal(
            em_row(csv_prop, csv_ind[0]),
            [2.0, 0.0, 0.005, 0.0, 1.5, 0.0, 0.0, 0.0, 0.0], decimal=14
        )

        cleanup(fn, csv_fn)

    # Built-in ITU materials
    def test_itu_air(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "air")
        try:
            (_, _, _, _, _, mtl_ind, mtl_names, _, csv_ind, _, csv_prop) = \
                quadriga_lib.RTtools.obj_file_read(fn)
            npt.assert_almost_equal(
                em_row(csv_prop, csv_ind[0]),
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=14,
            )
            npt.assert_equal(mtl_names[0], "air")
            npt.assert_(np.all(mtl_ind == 0))
        finally:
            if os.path.isfile(fn):
                os.remove(fn)

    def test_itu_concrete_and_wood(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "itu_concrete", "itu_wood")
        try:
            (_, _, _, _, _, mtl_ind, mtl_names, _, csv_ind, _, csv_prop) = \
                quadriga_lib.RTtools.obj_file_read(fn)
            npt.assert_almost_equal(
                em_row(csv_prop, csv_ind[0]),
                [5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=3,
            )
            npt.assert_equal(mtl_names[0], "itu_concrete")
            npt.assert_equal(mtl_ind[0], 0)
            npt.assert_almost_equal(
                em_row(csv_prop, csv_ind[4]),
                [1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=3,
            )
            npt.assert_equal(mtl_names[1], "itu_wood")
            npt.assert_equal(mtl_ind[4], 1)
        finally:
            if os.path.isfile(fn):
                os.remove(fn)

    # Material name with dot-suffix (Blender exports these)
    def test_material_dot_suffix(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "itu_brick.001", "itu_metal.shiny.001")
        try:
            (_, _, _, _, _, _, mtl_names, _, csv_ind, _, csv_prop) = \
                quadriga_lib.RTtools.obj_file_read(fn)
            # itu_brick.001 -> itu_brick (base name = everything before the first dot)
            npt.assert_almost_equal(
                em_row(csv_prop, csv_ind[0]),
                [3.91, 0.0, 0.0238, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=3,
            )
            npt.assert_equal(mtl_names[0], "itu_brick.001")  # raw name preserved
            # itu_metal.shiny.001 -> itu_metal (everything before the first dot)
            npt.assert_almost_equal(
                em_row(csv_prop, csv_ind[4]),
                [1.0, 0.0, 1.0e7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=3,
            )
            npt.assert_equal(mtl_names[1], "itu_metal.shiny.001")
        finally:
            if os.path.isfile(fn):
                os.remove(fn)

    # Unknown material: non-strict falls back to air; strict raises
    def test_strict_flag(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "not_a_real_material")
        try:
            # non-strict (default): resolves to air (row 0); request csv_ind so resolution runs
            (_, _, _, _, _, _, mtl_names, _, csv_ind, _, _) = \
                quadriga_lib.RTtools.obj_file_read(fn, "", False)
            npt.assert_equal(mtl_names[0], "not_a_real_material")  # raw name kept
            npt.assert_(np.all(csv_ind == 0))                      # resolved to air

            # strict: raises because the material is absent from the table
            with self.assertRaises(ValueError):
                quadriga_lib.RTtools.obj_file_read(fn, "", True)
        finally:
            if os.path.isfile(fn):
                os.remove(fn)


if __name__ == "__main__":
    unittest.main()