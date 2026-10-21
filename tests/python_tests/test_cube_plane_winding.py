# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import RTtools


def normals(mesh):
    # Face normals (v2 - v1) x (v3 - v1)
    return np.cross(mesh[:, 3:6] - mesh[:, 0:3], mesh[:, 6:9] - mesh[:, 0:3])


class test_cube_plane_winding(unittest.TestCase):

    def test_cube_negative_scale_keeps_outward_normals(self):
        # A negative product of the scales swaps the winding, so the normals stay outward
        for scale in ([-1.0], [1.0, 1.0, -2.0], [-1.0, -1.0, 1.0], [-1.0, -2.0, -0.5]):
            with self.subTest(scale=scale):
                m = RTtools.cube(np.array(scale), None, np.array([3.0, -1.0, 2.0]))
                centroid = np.array([3.0, -1.0, 2.0])
                face_center = (m[:, 0:3] + m[:, 3:6] + m[:, 6:9]) / 3.0
                outward = np.sum(normals(m) * (face_center - centroid), axis=1)
                self.assertTrue(np.all(outward > 0.0))

    def test_plane_negative_scale_keeps_up_normals(self):
        # A negative product of the x and y scales swaps the winding, the normals stay +z
        for scale in ([-1.0, 1.0], [2.0, -0.5], [-1.0], [-1.0, -3.0], [-1.0, 1.0, 5.0]):
            with self.subTest(scale=scale):
                m = RTtools.plane(np.array(scale))
                n = normals(m)
                self.assertTrue(np.all(n[:, 2] > 0.0))
                np.testing.assert_allclose(n[:, 0:2], 0.0, atol=1e-12)


if __name__ == '__main__':
    unittest.main()
