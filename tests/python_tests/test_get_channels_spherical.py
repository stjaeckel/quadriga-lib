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

class test_case(unittest.TestCase):

    def test(self):

        ant = arrayant.generate('omni')  
        ant = arrayant.copy_element(ant,0,1)
        ant['e_theta_re'][:, :, 1] = 2
        ant['e_theta_im'][:, :, 1] = 0
        ant['e_phi_re'][:, :, 1] = 0
        ant['e_phi_im'][:, :, 1] = 0
        ant['element_pos'] = np.array([[0, 0],[1, -1],[0, 0]])
        ant['coupling_re'] = np.eye(2)
        ant['coupling_im'] = np.zeros((2, 2))

        fbs_pos = np.array([[10, 0], [0, 10], [1, 11]])
        path_gain = np.array([1, 0.25])
        path_length = np.zeros(2)
        M = np.array([[1,1],[0,0],[0,0],[0,0],[0,0],[0,0],[-1,-1],[0,0]])

        coeff_re, coeff_im, delay, aod, eod, aoa, eoa = arrayant.get_channels_spherical( ant, ant, 
            fbs_pos, fbs_pos, path_gain, path_length, M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1, 0, angles=1 )

        # Convert to degrees
        aod = np.degrees(aod)
        eod = np.degrees(eod)
        aoa = np.degrees(aoa)
        eoa = np.degrees(eoa)

        # AOD assertions
        alpha = np.degrees(np.arctan(2.0 / 20.0))
        beta = 90.0
        npt.assert_almost_equal(aod[:, :, 0], [[0, alpha], [-alpha, 0]], decimal=14)
        npt.assert_almost_equal(aod[:, :, 1], [[beta, beta], [beta, beta]], decimal=14)

        # AOA cosd check
        alpha = 180 - alpha
        beta = 180
        npt.assert_almost_equal(np.cos(np.radians(aoa[:, :, 0])), 
                                np.cos(np.radians([[beta, -alpha], [alpha, beta]])), decimal=13)

        # AOA slice 2
        alpha = 180 - np.degrees(np.arctan(9.0 / 20.0))
        beta = 180 - np.degrees(np.arctan(11.0 / 20.0))
        npt.assert_almost_equal(aoa[:, :, 1], [[alpha, alpha], [beta, beta]], decimal=13)

        # EOD assertions
        alpha = np.degrees(np.arctan(10.0 / 11.0))
        beta = np.degrees(np.arctan(10.0 / 9.0))
        npt.assert_almost_equal(eod[:, :, 0], [[0, 0], [0, 0]], decimal=14)
        npt.assert_almost_equal(eod[:, :, 1], [[beta, alpha], [beta, alpha]], decimal=13)

        # EOA assertions
        alpha = np.degrees(np.arctan(10.0 / np.sqrt(9.0**2 + 20.0**2)))
        beta = np.degrees(np.arctan(10.0 / np.sqrt(11.0**2 + 20.0**2)))
        npt.assert_almost_equal(eoa[:, :, 0], [[0, 0], [0, 0]], decimal=14)
        npt.assert_almost_equal(eoa[:, :, 1], [[alpha, alpha], [beta, beta]], decimal=13)

        # Amplitude
        amp = coeff_re**2 + coeff_im**2
        npt.assert_almost_equal(amp[:, :, 0], [[1, 4], [4, 16]], decimal=13)
        npt.assert_almost_equal(amp[:, :, 1], [[0.25, 1], [1, 4]], decimal=13)

        # Delays
        C = 299792458.0
        d0 = 20.0
        d1 = np.hypot(20.0, 2.0)
        e0 = np.hypot(9.0, 10.0) + np.sqrt(9.0**2 + 20.0**2 + 10.0**2)
        e1 = np.hypot(9.0, 10.0) + np.sqrt(11.0**2 + 20.0**2 + 10.0**2)
        e2 = np.hypot(11.0, 10.0) + np.sqrt(9.0**2 + 20.0**2 + 10.0**2)
        e3 = np.hypot(11.0, 10.0) + np.sqrt(11.0**2 + 20.0**2 + 10.0**2)

        npt.assert_almost_equal(delay[:, :, 0], np.array([[d0, d1], [d1, d0]]) / C, decimal=13)
        npt.assert_almost_equal(delay[:, :, 1], np.array([[e0, e2], [e1, e3]]) / C, decimal=13)

        # Exception handling
        ant.pop('element_pos', None)
        A, B, C = arrayant.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, 
                                                 M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        
        ant.pop('coupling_re', None)
        with self.assertRaises(ValueError) as context:
            arrayant.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, 
                                              M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        self.assertEqual(str(context.exception), "Transmit antenna: Imaginary part of coupling matrix (phase component) defined without real part (absolute component)")

        ant.pop('coupling_im', None)
        data = arrayant.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, 
                                                 M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)

        # Mismatching n_path
        with self.assertRaises(ValueError) as context:
            arrayant.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain[[0, 1, 1]], path_length, 
                                                 M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        self.assertEqual(str(context.exception), "Inputs 'fbs_pos', 'lbs_pos', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).")

        # Wrong size of TX position
        with self.assertRaises(ValueError) as context:
            arrayant.get_channels_spherical( ant, ant, fbs_pos, fbs_pos, path_gain, path_length, 
                                              M, [0,0,1,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        self.assertEqual(str(context.exception), "Input 'tx_pos' has incorrect number of elements.")