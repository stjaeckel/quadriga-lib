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

        ant = quadriga_lib.arrayant_generate('custom', az_3dB=5, el_3dB=20 )
        out = quadriga_lib.arrayant_rotate_pattern(ant, 0, 0, 90)

        flat_idx = np.argmax(out["e_theta_re"])
        s0, s1, s2 = np.unravel_index(flat_idx, out["e_theta_re"].shape)
        assert s0 == 90
        assert s1 == 270

        out = quadriga_lib.arrayant_rotate_pattern(ant, 0, -45, 0)

        flat_idx = np.argmax(out["e_theta_re"])
        s0, s1, s2 = np.unravel_index(flat_idx, out["e_theta_re"].shape)
        assert s0 == 135
        assert s1 == 180

        v = ant["e_theta_re"][90, 180, 0]
        out = quadriga_lib.arrayant_rotate_pattern(ant, 45, 0, 0)
        npt.assert_almost_equal(out["e_theta_re"][90,180,0] * np.sqrt(2), v, decimal=14)
        npt.assert_almost_equal(out["e_phi_re"][90,180,0] * np.sqrt(2), v, decimal=14)

        out = quadriga_lib.arrayant_rotate_pattern(ant, 180, 180, 180)
        npt.assert_almost_equal(out["e_theta_re"], ant["e_theta_re"], decimal=13)

        out = quadriga_lib.arrayant_rotate_pattern(ant, 90, 0, 0, 1)
        npt.assert_almost_equal(out["e_theta_re"][90,180,0], v, decimal=14)

        # Single element rotation
        with self.assertRaises(ValueError) as context:
            quadriga_lib.arrayant_rotate_pattern(ant, 0, 0, 90, 1, 3)
        self.assertEqual(str(context.exception), "Input parameter 'element' out of bound.")

        ant = quadriga_lib.arrayant_copy_element(ant,0,[1,2])
        ant = quadriga_lib.arrayant_copy_element(ant,[0,1],[1,2])
        out = quadriga_lib.arrayant_rotate_pattern(ant, 0, 0, 90, element=[1,2])

        flat_idx = np.argmax(out["e_theta_re"][:,:,0])
        s0, s1 = np.unravel_index(flat_idx, out["e_theta_re"][:,:,0].shape)
        assert s0 == 90
        assert s1 == 180

        flat_idx = np.argmax(out["e_theta_re"][:,:,1])
        s0, s1 = np.unravel_index(flat_idx, out["e_theta_re"][:,:,1].shape)
        assert s0 == 90
        assert s1 == 270

        flat_idx = np.argmax(out["e_theta_re"][:,:,2])
        s0, s1 = np.unravel_index(flat_idx, out["e_theta_re"][:,:,2].shape)
        assert s0 == 90
        assert s1 == 270

        with self.assertRaises(TypeError) as context:
            quadriga_lib.arrayant_rotate_pattern(ant["e_theta_re"])
        
