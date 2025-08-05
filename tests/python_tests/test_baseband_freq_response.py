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

from quadriga_lib import channel

class test_case(unittest.TestCase):

    def test(self):

        # Create a 4x3x2 array of zeros
        coeff = np.zeros((4, 3, 2), dtype=complex)

        # Fill in the values for the first slice of the third dimension
        coeff[:, 0, 0] = np.arange(0.25, 1.25, 0.25)
        coeff[:, 1, 0] = np.arange(1, 5)
        coeff[:, 2, 0] = 1j * np.arange(1, 5)

        # Fill in the values for the second slice of the third dimension
        coeff[:, :, 1] = -coeff[:, :, 0]

        # Create list of 3 snapshots
        coeff_list = [coeff, 2 * coeff, 3 * coeff]

        # Speed of light
        fc = 299792458.0

        # Create a 1x1x2 array of zeros
        delay = np.zeros((1, 1, 2), dtype=float)

        # Fill in the values for the delay
        delay[0, 0, 0] = 1 / fc
        delay[0, 0, 1] = 1.5 / fc

        # Create additional dimensions for the array and fill in the values
        delay_list = [delay, delay, delay]

        # Pilots array
        pilots = np.arange(0, 2.1, 0.1)

        hmat = channel.baseband_freq_response( coeff_list, delay_list, fc, carriers=11 )
        hmat_re = np.real(hmat)
        hmat_im = np.imag(hmat)

        T = np.zeros((4, 3))
        npt.assert_almost_equal( hmat_re[:, :, 0, 0], T, decimal=5 )
        npt.assert_almost_equal( hmat_im[:, :, 0, 0], T, decimal=5 )
        
        npt.assert_almost_equal( hmat_re[:, 0, :, :]*4.0, hmat_re[:, 1, :, :], decimal=6 )

        T = np.array([ [0.5, 2, 0], [1, 4, 0], [1.5, 6, 0], [2, 8, 0]])
        npt.assert_almost_equal( hmat_re[:, :, 10, 0], T, decimal=6 )

        npt.assert_almost_equal( hmat_re[:, :, :, 0]*2.0, hmat_re[:, :, :, 1], decimal=6 )
        npt.assert_almost_equal( hmat_re[:, :, :, 0]*3.0, hmat_re[:, :, :, 2], decimal=5 )

        hmat = channel.baseband_freq_response( coeff_list, delay_list, fc, pilot_grid=pilots, snap=[1,2,1,0] )
        hmat_re = np.real(hmat)
        hmat_im = np.imag(hmat)

        T = np.zeros((4, 3))
        npt.assert_almost_equal( hmat_re[:, :, 20, 3], T, decimal=5 )
        npt.assert_almost_equal( hmat_im[:, :, 20, 3], T, decimal=5 )

        T = np.array([ [0.25, 1, -1], [0.5, 2, -2], [0.75, 3, -3], [1, 4, -4]])
        npt.assert_almost_equal( hmat_im[:, :, 15, 3], T, decimal=5 )

        npt.assert_almost_equal( hmat_re[:, :, :, 3]*2.0, hmat_re[:, :, :, 0], decimal=5 )
        npt.assert_almost_equal( hmat_re[:, :, :, 3]*3.0, hmat_re[:, :, :, 1], decimal=5 )
        npt.assert_almost_equal( hmat_re[:, :, :, 3]*2.0, hmat_re[:, :, :, 2], decimal=5 )


if __name__ == '__main__':
    unittest.main()
