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

        ant = arrayant.generate('omni')  
        ant = arrayant.copy_element(ant,0,1)
        ant['e_theta_re'][:, :, 1] = 2
        ant['e_theta_im'][:, :, 1] = 0
        ant['e_phi_re'][:, :, 1] = 0
        ant['e_phi_im'][:, :, 1] = 0
        ant['element_pos'] = np.array([[0, 0],[1, -1],[0, 0]])
        ant['coupling_re'] = np.eye(2)
        ant['coupling_im'] = np.zeros((2, 2))

        pi = np.pi
        aod = np.radians([0, 90])
        eod = np.radians([0, 45])
        aoa = np.radians([180, 180 - np.degrees(np.arctan(10 / 20))])
        eoa = np.radians([0, np.degrees(np.arctan(10 / np.sqrt(10**2 + 20**2)))])
        path_gain = np.array([1, 0.25])
        path_length = np.array([20, np.hypot(10, 10) + np.sqrt(10**2 + 20**2 + 10**2)])
        M = np.array([[1,1],[0,0],[0,0],[0,0],[0,0],[0,0],[-1,-1],[0,0]])

        coeff_re, coeff_im, delay, rx_Doppler = arrayant.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, 
                                                                                 M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        
        amp = coeff_re**2 + coeff_im**2
        npt.assert_almost_equal(amp[:, :, 0], np.array([[1, 4], [4, 16]]), decimal=13)
        npt.assert_almost_equal(amp[:, :, 1], np.array([[0.25, 1], [1, 4]]), decimal=13)
        
        C = 299792458.0
        d0 = 20.0
        d1 = 20.0
        e0 = np.sqrt(9**2 + 10**2) + np.sqrt(9**2 + 20**2 + 10**2)
        e1 = np.sqrt(9**2 + 10**2) + np.sqrt(11**2 + 20**2 + 10**2)
        e2 = np.sqrt(11**2 + 10**2) + np.sqrt(9**2 + 20**2 + 10**2)
        e3 = np.sqrt(11**2 + 10**2) + np.sqrt(11**2 + 20**2 + 10**2)

        npt.assert_almost_equal(delay[:, :, 0], np.array([[d0, d1], [d1, d0]]) / C, decimal=13)
        npt.assert_almost_equal(delay[:, :, 1], np.array([[e0, e2], [e1, e3]]) / C, decimal=10)

        Doppler = np.cos(aoa[1]) * np.cos(eoa[1])
        npt.assert_almost_equal(rx_Doppler, np.array([-1, Doppler]), decimal=13)

        # Exception handling
        ant.pop('element_pos', None)
        data = arrayant.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, 
                                                 M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        
        ant.pop('coupling_re', None)
        with self.assertRaises(ValueError) as context:
            arrayant.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, 
                                              M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        self.assertEqual(str(context.exception), "Transmit antenna: Imaginary part of coupling matrix (phase component) defined without real part (absolute component)")

        ant.pop('coupling_im', None)
        data = arrayant.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, 
                                                 M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)

        # Mismatching n_path
        with self.assertRaises(ValueError) as context:
            arrayant.get_channels_planar( ant, ant, aod[[0, 1, 1]], eod, aoa, eoa, path_gain, path_length, 
                                              M, [0,0,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        self.assertEqual(str(context.exception), "Inputs 'aod', 'eod', 'aoa', 'eoa', 'path_gain', 'path_length', and 'M' must have the same number of columns (n_paths).")

        # Wrong size of TX position
        with self.assertRaises(ValueError) as context:
            arrayant.get_channels_planar( ant, ant, aod, eod, aoa, eoa, path_gain, path_length, 
                                              M, [0,0,1,1], [0,0,0], [20,0,1], [0,0,0], 2997924580.0, 1)
        self.assertEqual(str(context.exception), "Input 'tx_pos' has incorrect number of elements.")


