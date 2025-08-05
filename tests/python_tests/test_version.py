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

class test_case(unittest.TestCase):

    def test(self):

        v = quadriga_lib.version()
       
if __name__ == '__main__':
    unittest.main()
