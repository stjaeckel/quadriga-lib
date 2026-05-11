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
        )  # zero‐based indexing

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
            mtl_prop,
            vert_list,
            face_ind,
            obj_ind,
            mtl_ind,
            obj_names,
            mtl_names,
            bsdf,
        ) = quadriga_lib.RTtools.obj_file_read(fn)

        assert mesh.shape == (12, 9)
        assert mtl_prop.shape == (12, 9)
        assert vert_list.shape == (8, 3)
        assert face_ind.shape == (12, 3)
        assert obj_ind.shape == (12,)
        assert mtl_ind.shape == (12,)
        assert len(obj_names) == 1
        assert len(mtl_names) == 0
        assert bsdf.shape == (0, 0)

        npt.assert_(mesh.dtype == np.float64)
        npt.assert_(mtl_prop.dtype == np.float64)
        npt.assert_(vert_list.dtype == np.float64)
        npt.assert_(face_ind.dtype == np.int64)
        npt.assert_(obj_ind.dtype == np.int64)
        npt.assert_(mtl_ind.dtype == np.int64)

        npt.assert_(np.all(mtl_prop[:, 0] == 1.0))
        npt.assert_(np.all(mtl_prop[:, 1:8] == 0.0))
        npt.assert_(np.all(mtl_prop[:, 8] == 1.0))  # fRef defaults to 1 GHz

        npt.assert_almost_equal(vert_list, vert_list_correct, decimal=14)
        npt.assert_array_equal(face_ind, face_ind_correct)
        npt.assert_almost_equal(mesh, mesh_correct, decimal=14)

        npt.assert_(np.all(obj_ind == 1))
        npt.assert_(np.all(mtl_ind == 0))
        npt.assert_equal(obj_names[0], "Cube")

        os.remove(fn)

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

        data = quadriga_lib.RTtools.obj_file_read(fn)

        assert len(data[6]) == 2
        assert len(data[7]) == 1

        npt.assert_(np.all(data[1][[0, 1], 0] == 1.0))
        npt.assert_(np.all(data[1][[2, 3], 0] > 1.5))

        expected_face_ind = np.array([[2, 3, 1], [2, 4, 3], [6, 7, 5], [6, 8, 7]]) - 1
        npt.assert_array_equal(data[3], expected_face_ind)

        npt.assert_array_equal(data[4], [1, 1, 2, 2])
        npt.assert_array_equal(data[5], [0, 0, 1, 1])
        npt.assert_equal(data[6][0], "Plane")
        npt.assert_equal(data[6][1], "Plane.001")
        npt.assert_equal(data[7][0], "itu_wood")

        os.remove(fn)

        with open(fn, "w") as f:
            f.write("o Plane\n")
            f.write("v -1.000000 -1.000000 0.000000\n")
            f.write("v 1.000000 -1.000000 0.000000\n")
            f.write("v -1.000000 1.000000 0.000000\n")
            f.write("v 1.000000 1.000000 0.000000\n")
            f.write("vn -0.0000 -0.0000 1.0000\n")
            f.write("vt 1.000000 0.000000\n")
            f.write("vt 0.000000 1.000000\n")
            f.write("vt 0.000000 0.000000\n")
            f.write("vt 1.000000 1.000000\n")
            f.write("usemtl Cst::1.1:1.2:1.3:1.4:10\n")
            f.write("f 2/1/1 3/2/1 1/3/1\n")
            f.write("usemtl Cst::2.1:2.2:2.3:2.4:20\n")
            f.write("f 2/1/1 4/4/1 3/2/1\n")

        (
            mesh,
            mtl_prop,
            vert_list,
            face_ind,
            obj_ind,
            mtl_ind,
            obj_names,
            mtl_names,
            bsdf,
        ) = quadriga_lib.RTtools.obj_file_read(fn)

        npt.assert_almost_equal(
            mtl_prop[0, :], [1.1, 1.2, 1.3, 1.4, 10, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_almost_equal(
            mtl_prop[1, :], [2.1, 2.2, 2.3, 2.4, 20, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[0], "Cst::1.1:1.2:1.3:1.4:10")
        npt.assert_equal(mtl_names[1], "Cst::2.1:2.2:2.3:2.4:20")
        npt.assert_array_equal(obj_ind, [1, 1])
        npt.assert_array_equal(mtl_ind, [1, 2])

        with self.assertRaises(TypeError) as context:
            quadriga_lib.RTtools.obj_file_read()

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

        # Basic custom materials
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att\n")
            f.write("custom_material_1,2.5,0.0,0.001,0.5,5.0\n")
            f.write("custom_material_2,4.0,-0.1,0.05,1.2,10.0\n")

        create_cube_with_materials(fn, "custom_material_1", "custom_material_2")

        (
            mesh,
            mtl_prop,
            vert_list,
            face_ind,
            obj_ind,
            mtl_ind,
            obj_names,
            mtl_names,
            bsdf,
        ) = quadriga_lib.RTtools.obj_file_read(fn, csv_fn)

        npt.assert_almost_equal(
            mtl_prop[0, :], [2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[0], "custom_material_1")
        npt.assert_equal(mtl_ind[0], 1)

        npt.assert_almost_equal(
            mtl_prop[4, :], [4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[1], "custom_material_2")
        npt.assert_equal(mtl_ind[4], 2)

        # Jumbled column order
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("att,d,c,b,a,name\n")
            f.write("5.0,0.5,0.001,0.0,2.5,custom_material_1\n")
            f.write("10.0,1.2,0.05,-0.1,4.0,custom_material_2\n")

        create_cube_with_materials(fn, "custom_material_1", "custom_material_2")

        (
            mesh,
            mtl_prop,
            vert_list,
            face_ind,
            obj_ind,
            mtl_ind,
            obj_names,
            mtl_names,
            bsdf,
        ) = quadriga_lib.RTtools.obj_file_read(fn, csv_fn)

        npt.assert_almost_equal(
            mtl_prop[0, :], [2.5, 0.0, 0.001, 0.5, 5.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[0], "custom_material_1")

        npt.assert_almost_equal(
            mtl_prop[4, :], [4.0, -0.1, 0.05, 1.2, 10.0, 0.0, 0.0, 0.0, 1.0], decimal=14
        )
        npt.assert_equal(mtl_names[1], "custom_material_2")

        # Duplicate material names
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att\n")
            f.write("custom_material_1,2.5,0.0,0.001,0.5,5.0\n")
            f.write("custom_material_1,4.0,-0.1,0.05,1.2,10.0\n")

        create_cube_with_materials(fn, "custom_material_1")

        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_read(fn, csv_fn)

        # Non-existent CSV file
        cleanup(fn, csv_fn)
        create_cube_with_materials(fn, "custom_material_1")

        with self.assertRaises(ValueError):
            quadriga_lib.RTtools.obj_file_read(fn, "nonexistent.csv")

        cleanup(fn, csv_fn)

        # CSV with new frequency-dependent columns populated.
        # Note jumbled order: fRef before alphaB → exercises the column reorder code.
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att,attB,alpha,fRef,alphaB\n")
            f.write("lossy_wall,4.5,0.1,0.02,0.8,3.0,0.2,0.5,2.4,0.15\n")

        create_cube_with_materials(fn, "lossy_wall")

        _, mtl_prop, _, _, _, _, _, mtl_names, _ = quadriga_lib.RTtools.obj_file_read(
            fn, csv_fn
        )
        npt.assert_almost_equal(
            mtl_prop[0, :], [4.5, 0.1, 0.02, 0.8, 3.0, 0.2, 0.5, 0.15, 2.4], decimal=14
        )
        npt.assert_equal(mtl_names[0], "lossy_wall")

        # CSV with subset of optional columns; unspecified ones default.
        # fRef given but attB/alpha/alphaB absent -> defaults 0.
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,c,fRef\n")
            f.write("partial,3.0,0.01,5.0\n")

        create_cube_with_materials(fn, "partial")

        _, mtl_prop, _, _, _, _, _, _, _ = quadriga_lib.RTtools.obj_file_read(
            fn, csv_fn
        )
        npt.assert_almost_equal(
            mtl_prop[0, :], [3.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0], decimal=14
        )

        # Empty cells in optional columns fall back to defaults.
        cleanup(fn, csv_fn)
        with open(csv_fn, "w") as f:
            f.write("name,a,b,c,d,att,attB,alpha,alphaB,fRef\n")
            f.write("sparse,2.0,,0.005,,1.5,,,,\n")

        create_cube_with_materials(fn, "sparse")

        _, mtl_prop, _, _, _, _, _, _, _ = quadriga_lib.RTtools.obj_file_read(
            fn, csv_fn
        )
        npt.assert_almost_equal(
            mtl_prop[0, :], [2.0, 0.0, 0.005, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0], decimal=14
        )

        cleanup(fn, csv_fn)

    # Built-in ITU materials
    def test_itu_air(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "air")
        try:
            _, mtl_prop, _, _, _, mtl_ind, _, mtl_names, _ = (
                quadriga_lib.RTtools.obj_file_read(fn)
            )
            npt.assert_almost_equal(
                mtl_prop[0, :],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=14,
            )
            npt.assert_equal(mtl_names[0], "air")
            npt.assert_(np.all(mtl_ind == 1))
        finally:
            if os.path.isfile(fn):
                os.remove(fn)

    def test_itu_concrete_and_wood(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "itu_concrete", "itu_wood")
        try:
            _, mtl_prop, _, _, _, mtl_ind, _, mtl_names, _ = (
                quadriga_lib.RTtools.obj_file_read(fn)
            )
            npt.assert_almost_equal(
                mtl_prop[0, :],
                [5.24, 0.0, 0.0462, 0.7822, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=14,
            )
            npt.assert_equal(mtl_names[0], "itu_concrete")
            npt.assert_equal(mtl_ind[0], 1)
            npt.assert_almost_equal(
                mtl_prop[4, :],
                [1.99, 0.0, 0.0047, 1.0718, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=14,
            )
            npt.assert_equal(mtl_names[1], "itu_wood")
            npt.assert_equal(mtl_ind[4], 2)
        finally:
            if os.path.isfile(fn):
                os.remove(fn)

    # Material name with dot-suffix
    # Verifies the .001 / .shiny.001 suffix handling, which is what Blender exports.
    def test_material_dot_suffix(self):
        fn = "cube.obj"
        create_cube_with_materials(fn, "itu_brick.001", "itu_metal.shiny.001")
        try:
            _, mtl_prop, _, _, _, _, _, mtl_names, _ = (
                quadriga_lib.RTtools.obj_file_read(fn)
            )
            npt.assert_almost_equal(
                mtl_prop[0, :],
                [3.91, 0.0, 0.0238, 0.16, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=14,
            )
            npt.assert_equal(mtl_names[0], "itu_brick.001")
            npt.assert_almost_equal(
                mtl_prop[4, :],
                [1.0, 0.0, 1.0e7, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                decimal=14,
            )
            npt.assert_equal(mtl_names[1], "itu_metal.shiny.001")
        finally:
            if os.path.isfile(fn):
                os.remove(fn)

    #  Inline custom syntax — 7 and 9 value forms
    # Tests progressive truncation: 9-value form fully populated, partial form (alpha-only here) defaults alphaB=0, fRef=1.
    def test_inline_custom_full_and_partial(self):
        fn = "cube.obj"
        create_cube_with_materials(
            fn,
            "full_inline::2.2:0.05:0.01:0.7:4.0:0.3:0.8:0.2:3.5",  # all 9 values
            "partial_inline::6.0:0:0:0:0:0:0.1",
        )  # only alpha set
        try:
            _, mtl_prop, _, _, _, _, _, _, _ = quadriga_lib.RTtools.obj_file_read(fn)
            npt.assert_almost_equal(
                mtl_prop[0, :],
                [2.2, 0.05, 0.01, 0.7, 4.0, 0.3, 0.8, 0.2, 3.5],
                decimal=14,
            )
            npt.assert_almost_equal(
                mtl_prop[4, :],
                [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 1.0],
                decimal=14,
            )
        finally:
            if os.path.isfile(fn):
                os.remove(fn)
