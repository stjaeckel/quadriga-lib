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

        ant = quadriga_lib.arrayant_generate("3gpp", res=10, N=2, spacing=0.0)
        ant["e_theta_re"][:,:,1] = 2*ant["e_theta_re"][:,:,1]
        ant["coupling_re"] = [1,1]
        ant["coupling_im"] = [0,0]
        ant["name"] = "buy_bitcoin"

        out = quadriga_lib.arrayant_combine_pattern(ant)
        npt.assert_almost_equal(out["e_theta_re"][:,:,0], 3*ant["e_theta_re"][:,:,0], decimal=14)
        npt.assert_almost_equal(out["coupling_re"], 1, decimal=14)
        npt.assert_almost_equal(out["coupling_im"], 0, decimal=14)
        npt.assert_almost_equal(out["element_pos"][:,0], [0,0,0], decimal=14)

        az_grid = np.linspace(-np.pi, np.pi, 73)
        out = quadriga_lib.arrayant_combine_pattern(ant, 3.0e9, az_grid)
        npt.assert_almost_equal(out["e_theta_re"][:,::2,0], 3*ant["e_theta_re"][:,:,0], decimal=14)
        assert out["center_freq"] == 3.0e9

        el_grid = np.linspace(-np.pi/2, np.pi/2, 37)
        out = quadriga_lib.arrayant_combine_pattern(ant, elevation_grid=el_grid)
        npt.assert_almost_equal(out["e_theta_re"][::2,:,0], 3*ant["e_theta_re"][:,:,0], decimal=14)
        assert out["name"] == "buy_bitcoin"

        directivity_ant = quadriga_lib.arrayant_calc_directivity(ant,0)
        directivity_out = quadriga_lib.arrayant_calc_directivity(out)
        npt.assert_almost_equal(directivity_ant, directivity_out, decimal=1)

        ant["element_pos"][1,:] = [-0.25,0.25]
        out = quadriga_lib.arrayant_combine_pattern(ant)
        directivity_out = quadriga_lib.arrayant_calc_directivity(out)
        assert directivity_out > directivity_ant + 1.2
        
if __name__ == '__main__':
    unittest.main()
