import sys
import os
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib

class test_case(unittest.TestCase):

    def test(self):
        # Test

        # -----------------------------------------------------------------
        # 1. Read reference OBJ
        # -----------------------------------------------------------------
        obj_file = os.path.join(current_dir, '../data/test_scene_pbr.obj')

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
        ) = quadriga_lib.RTtools.obj_file_read(obj_file)

        # Minimal sanity check
        self.assertTrue(vert_list.size, "No vertices loaded from OBJ")

        # -----------------------------------------------------------------
        # 2. Write the Mitsuba XML scene
        # -----------------------------------------------------------------
        xml_file = os.path.join(current_dir, 'test_scene_x.xml')

        quadriga_lib.RTtools.mitsuba_xml_file_write(
            xml_file,
            vert_list,
            face_ind,
            obj_ind,
            mtl_ind,
            obj_names,
            mtl_names,
            bsdf,
            True        # map_to_itu_materials
        )

        xml_stem = os.path.splitext(os.path.basename(xml_file))[0]
        mesh_folder = os.path.join(current_dir, f'{xml_stem}_meshes')

        # -----------------------------------------------------------------
        # 3. Assertions
        # -----------------------------------------------------------------
        self.assertTrue(os.path.exists(xml_file),       "XML file was not created")
        self.assertTrue(os.path.isdir(mesh_folder),     "Mesh folder does not exist")
        self.assertTrue(os.listdir(mesh_folder),        "Mesh folder is empty")

        # Optionally, check that the number of mesh files equals objects
        self.assertEqual(
            16, 
            len([f for f in os.listdir(mesh_folder) if os.path.isfile(os.path.join(mesh_folder, f))]),
            "Unexpected number of mesh files",
        )

        # -----------------------------------------------------------------
        # 4. Clean-up (delete XML + folder)
        # -----------------------------------------------------------------
        # Recursively remove everything in mesh_folder without shutil
        if os.path.isdir(mesh_folder):
            for root, dirs, files in os.walk(mesh_folder, topdown=False):
                for name in files:
                    try:
                        os.remove(os.path.join(root, name))
                    except OSError:
                        pass
                for name in dirs:
                    try:
                        os.rmdir(os.path.join(root, name))
                    except OSError:
                        pass
            try:
                os.rmdir(mesh_folder)
            except OSError:
                pass

        # Remove the XML file
        try:
            os.remove(xml_file)
        except OSError:
            pass
        

