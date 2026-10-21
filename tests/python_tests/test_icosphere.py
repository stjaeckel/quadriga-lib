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

    def test_invalid_radius(self):
        # The radius must be finite and positive
        for r in (0.0, -1.0, float('nan'), float('inf')):
            with self.subTest(radius=r):
                with self.assertRaises(ValueError):
                    RTtools.icosphere(1, r)
       
class test_n_div_too_large(unittest.TestCase):

    # Checked before the outputs are allocated: 20 * n_div^2 wraps to a huge but representable count for 2^31 + 1
    def test(self):
        for n_div in (2**31 + 1, 2**30, 2**32):
            with self.assertRaisesRegex(ValueError, "Input 'n_div' is too large."):
                RTtools.icosphere(n_div)


if __name__ == '__main__':
    unittest.main()
