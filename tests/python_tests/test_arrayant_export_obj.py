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

class test_version(unittest.TestCase):

    def test(self):

        # Omni antenna, 10 deg
        data = quadriga_lib.arrayant.generate("3gpp", 10, M=2)

        quadriga_lib.arrayant.export_obj_file("test_py.obj", data)

        assert os.path.exists("test_py.obj")
        assert os.path.exists("test_py.mtl")

        os.remove("test_py.obj")
        os.remove("test_py.mtl")
            
if __name__ == '__main__':
    unittest.main()
