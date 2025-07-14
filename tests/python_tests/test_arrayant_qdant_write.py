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

from quadriga_lib import arrayant

class test_version(unittest.TestCase):

    def test(self):

        if os.path.exists("test_py.qdant"):
            os.remove("test_py.qdant")

        ant = { 'azimuth_grid': np.array([-1.5, 0, 1.5, 2]) * np.pi/2,
                'elevation_grid': np.array([-0.9, 0, 0.9]) * np.pi/2,
                'e_theta_re': np.reshape(np.arange(1, 13), (3, -1)) / 2,
                'e_theta_im': -np.reshape(np.arange(1, 13), (3, -1)) * 0.002,
                'e_phi_re': -np.reshape(np.arange(1, 13), (3, -1)),
                'e_phi_im': -np.reshape(np.arange(1, 13), (3, -1)) * 0.001, 
                'element_pos': np.array([1, 2, 4]),
                'coupling_re': 1,
                'coupling_im': 0.1,
                'center_freq': 2e9,
                'name': 'name' }
        
        id_file = arrayant.qdant_write("test_py.qdant", ant)
        assert id_file == 1

        data = arrayant.qdant_read('test_py.qdant')
        npt.assert_almost_equal(data["azimuth_grid"], ant["azimuth_grid"], decimal=6)
        npt.assert_almost_equal(data["elevation_grid"], ant["elevation_grid"], decimal=6)
        npt.assert_almost_equal(data["e_theta_re"][:,:,0], ant["e_theta_re"], decimal=4)
        npt.assert_almost_equal(data["e_theta_im"][:,:,0], ant["e_theta_im"], decimal=4)
        npt.assert_almost_equal(data["e_phi_re"][:,:,0], ant["e_phi_re"], decimal=4)
        npt.assert_almost_equal(data["e_phi_im"][:,:,0], ant["e_phi_im"], decimal=4)
        npt.assert_almost_equal(data["element_pos"][:,0], ant["element_pos"], decimal=6)
        npt.assert_almost_equal(data["coupling_re"], ant["coupling_re"], decimal=4)
        npt.assert_almost_equal(data["coupling_im"], ant["coupling_im"], decimal=4)
        assert data["center_freq"] == ant["center_freq"]
        assert data["name"] == ant["name"]

        os.remove("test_py.qdant")

        


