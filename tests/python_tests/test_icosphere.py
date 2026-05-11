# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import unittest
import numpy as np

# Append the directory containing your package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

# Now you can import your package
from quadriga_lib import RTtools

class test_case(unittest.TestCase):

    def test(self):

        center, length, vert, direction = RTtools.icosphere()

        self.assertEqual(center.shape, (20,3))
        self.assertEqual(length.shape, (20,))
        self.assertEqual(vert.shape, (20,9))
        self.assertEqual(direction.shape, (20,6))
       
if __name__ == '__main__':
    unittest.main()
