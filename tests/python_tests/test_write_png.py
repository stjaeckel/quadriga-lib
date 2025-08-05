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

        N = 50                 # size along one edge
        max_val = 10.0          # value at the upper‑right corner

        # Create a left‑to‑right gradient (columns vary, rows repeat)
        gradient = np.tile(np.linspace(0.0, max_val, N, endpoint=False), (25, 1))
        gradient[0,:] = 0.0
        gradient[-1,:] = 10.0

        # File name to write
        out_file = "test.png"

        # quadriga_lib.tools.write_png(fn, data, colormap, min_val, max_val, log_transform)
        quadriga_lib.tools.write_png(
            out_file,
            gradient,
            "jet",          # any valid Matplotlib‑style colormap string works
            0.0,                # min_val  (matches gradient.min())
            max_val,            # max_val  (matches gradient.max())
            False               # log_transform
        )

        # Check that the file was created
        self.assertTrue(os.path.exists(out_file))

        # Clean up
        os.remove(out_file)