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
import quadriga_lib

class test_cart2geo(unittest.TestCase):

    def test(self):

        # This should work fine
        e = np.random.rand(3, 6, 2)
        x = quadriga_lib.tools.cart2geo(e)

        # We need 3 dimensions
        e = np.random.rand(3, 6, 5, 2)
        with self.assertRaises(ValueError) as context:
            x = quadriga_lib.tools.cart2geo(e)
        self.assertEqual(str(context.exception), "Expected 1D, 2D or 3D array, got 4D")

if __name__ == '__main__':
    unittest.main()
