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

        # Simple interpolation in az-direction
        ant = { "e_theta_re" : [[-2,2]], "e_theta_im" : [[-1,1]],
                "e_phi_re" : [[3,1]], "e_phi_im" : [[6,2]],
                "azimuth_grid" : [0,np.pi], "elevation_grid" : 0 }
        
        az = [[0, np.pi/4, np.pi/2, 3*np.pi/4]]
        el = [[-0.5,0,0,0.5]]

        vr,vi,hr,hi = arrayant.interpolate(ant, az, el)

        npt.assert_almost_equal(vr, [[-2, -1, 0, 1]], decimal=14)
        npt.assert_almost_equal(vi, [[-1, -0.5, 0, 0.5]], decimal=14)
        npt.assert_almost_equal(hr, [[3, 2.5, 2, 1.5]], decimal=14)
        npt.assert_almost_equal(hi, [[6, 5, 4, 3]], decimal=14)

        v,h = arrayant.interpolate(ant, az, el, complex=1)

        npt.assert_almost_equal(v.real, [[-2, -1, 0, 1]], decimal=14)
        npt.assert_almost_equal(v.imag, [[-1, -0.5, 0, 0.5]], decimal=14)
        npt.assert_almost_equal(h.real, [[3, 2.5, 2, 1.5]], decimal=14)
        npt.assert_almost_equal(h.imag, [[6, 5, 4, 3]], decimal=14)

        res = arrayant.interpolate(ant, az, el, dist=1, local_angles=1)

        npt.assert_almost_equal(res[4], [[0, 0, 0, 0]], decimal=14)
        npt.assert_almost_equal(res[5], az, decimal=14)
        npt.assert_almost_equal(res[6], el, decimal=14)

        # Simple interpolation in el-direction
        ant = { "e_theta_re" : [-2,2], "e_theta_im" : [-1,1],
                "e_phi_re" : [3,1], "e_phi_im" : [6,2],
                "azimuth_grid" : 0, "elevation_grid" : [0,np.pi/2] }
        
        az = [[-0.5,0,0,0.5]]
        el = [[0, np.pi/8, np.pi/4, 3*np.pi/8]]

        res = arrayant.interpolate(ant, az, el, dist=1, local_angles=1)

        npt.assert_almost_equal(res[0], [[-2, -1, 0, 1]], decimal=14)
        npt.assert_almost_equal(res[1], [[-1, -0.5, 0, 0.5]], decimal=14)
        npt.assert_almost_equal(res[2], [[3, 2.5, 2, 1.5]], decimal=14)
        npt.assert_almost_equal(res[3], [[6, 5, 4, 3]], decimal=14)
        npt.assert_almost_equal(res[4], [[0, 0, 0, 0]], decimal=14)
        npt.assert_almost_equal(res[5], az, decimal=14)
        npt.assert_almost_equal(res[6], el, decimal=14)

        # Spheric interpolation in az-direction with z-rotation
        ant = { "e_theta_re" : [[1,0]], "e_theta_im" : [[0,1]],
                "e_phi_re" : [[-2,0]], "e_phi_im" : [[0,-1]],
                "azimuth_grid" : [0,np.pi], "elevation_grid" : 0 }
        
        az = [[0, np.pi/4, np.pi/2, 3*np.pi/4]]
        el = [[0,0,0,0]]
        ori = [[0,0],[0,0],[-np.pi/8,np.pi/8]]

        res = arrayant.interpolate(ant, az, el, [0,0], ori, dist=1, local_angles=1)
        
        npt.assert_almost_equal(res[0][0,:], np.cos(np.array([0,1,2,3]) * np.pi/8 + np.pi/16), decimal=14)
        npt.assert_almost_equal(res[1][0,:], np.sin(np.array([0,1,2,3]) * np.pi/8 + np.pi/16), decimal=14)
        npt.assert_almost_equal(res[2][0,:], -np.cos(np.array([0,1,2,3]) * np.pi/8 + np.pi/16) * np.array([1.875, 1.625, 1.375, 1.125]), decimal=14)
        npt.assert_almost_equal(res[3][0,:], -np.sin(np.array([0,1,2,3]) * np.pi/8 + np.pi/16) * np.array([1.875, 1.625, 1.375, 1.125]), decimal=14)
        npt.assert_almost_equal(res[5][0,:], np.array([0,1,2,3]) * np.pi/4 + np.pi/8, decimal=14)
        npt.assert_almost_equal(res[5][1,:], np.array([0,1,2,3]) * np.pi/4 - np.pi/8, decimal=14)

        # Polarization rotation using x-rotation
        ant = { "e_theta_re" : [[1,0]], "e_theta_im" : [[1,0]],
                "e_phi_re" : [[0,0]], "e_phi_im" : [[0,0]],
                "azimuth_grid" : [0,np.pi], "elevation_grid" : 0 }
        
        ori = [[np.pi/4,-np.pi/4],[0,0],[0,0]]
        vr,vi,hr,hi = arrayant.interpolate(ant, 0, 0, [0,0], ori)
        
        npt.assert_almost_equal(vr[:,0], [1/np.sqrt(2),1/np.sqrt(2)], decimal=14)
        npt.assert_almost_equal(vi[:,0], [1/np.sqrt(2),1/np.sqrt(2)], decimal=14)
        npt.assert_almost_equal(hr[:,0], [1/np.sqrt(2),-1/np.sqrt(2)], decimal=14)
        npt.assert_almost_equal(hi[:,0], [1/np.sqrt(2),-1/np.sqrt(2)], decimal=14)
        
        # Test projected distance
        ant = { "e_theta_re" : 1, "e_theta_im" : 0,
                "e_phi_re" : 0, "e_phi_im" : 0,
                "azimuth_grid" : 0, "elevation_grid" : 0 }
        
        vr,vi,hr,hi,ds = arrayant.interpolate(ant, 0, 0, [0,0,0], element_pos=np.eye(3), dist=1)
        npt.assert_almost_equal(ds[:,0], [-1,0,0], decimal=14)
        
        vr,vi,hr,hi,ds = arrayant.interpolate(ant, 3*np.pi/4, 0, [0,0,0], element_pos=np.eye(3), dist=1)
        npt.assert_almost_equal(ds[:,0], [1/np.sqrt(2),-1/np.sqrt(2),0], decimal=14)

        vr,vi,hr,hi,ds = arrayant.interpolate(ant, 0, -np.pi/4, [0,0,0], element_pos=np.eye(3), dist=1)
        npt.assert_almost_equal(ds[:,0], [-1/np.sqrt(2),0,1/np.sqrt(2)], decimal=14)

        vr,vi,hr,hi,ds = arrayant.interpolate(ant, [[-np.pi,-np.pi/2,0]], [[0,0,-np.pi/2]], [0,0,0], element_pos=-np.eye(3), dist=1)
        npt.assert_almost_equal(ds, -np.eye(3), decimal=14)

        # Error handling
        ant = { "e_theta_re" : np.random.random((5,10,3)), 
                "e_theta_im" : np.random.random((5,10,3)), 
                "e_phi_re" : np.random.random((5,10,3)), 
                "e_phi_im" : np.random.random((5,10,3)), 
                "azimuth_grid" : np.linspace(-np.pi, np.pi, 10), 
                "elevation_grid" : np.linspace(-np.pi/2, np.pi/2, 5) }

        # Minimal working example
        arrayant.interpolate(ant, 0, 0)

        with self.assertRaises(KeyError) as context: # No arrayant data
            arrayant.interpolate()

        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant)
        self.assertEqual(str(context.exception), "Azimuth angles cannot be empty.")

        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant,0)
        self.assertEqual(str(context.exception), "Sizes of 'azimuth' and 'elevation' do not match.")

        ant["e_theta_re"] = np.random.random((2,10,3))
        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant, 0, 0)
        self.assertEqual(str(context.exception), "Sizes of 'e_theta_re' and 'e_theta_im' do not match.")

        ant["e_theta_im"] = np.random.random((2,10,3))
        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant, 0, 0)
        self.assertEqual(str(context.exception), "Sizes of 'e_theta_re' and 'e_phi_re' do not match.")

        ant["e_phi_re"] = np.random.random((2,10,3))
        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant, 0, 0)
        self.assertEqual(str(context.exception), "Sizes of 'e_theta_re' and 'e_phi_im' do not match.")

        ant["e_phi_im"] = np.random.random((2,10,3))
        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant, 0, 0)
        self.assertEqual(str(context.exception), "Number of elements in 'elevation_grid' does not match number of rows in pattern data.")

        ant["elevation_grid"] = [-np.pi/4, np.pi/4]
        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant, 0, 0, element=3)
        self.assertEqual(str(context.exception), "Element indices 'i_element' cannot exceed the array antenna size.")

        with self.assertRaises(ValueError) as context:
            arrayant.interpolate(ant, 0, 0, fast_access=1)
        self.assertEqual(str(context.exception), "Could not obtain memory view, possibly due to mismatching strides.")

        # Interpolate real data (fast access OK)
        ant = arrayant.generate("3gpp", az_3dB=90, el_3dB=90, res=10, N=2)
        arrayant.interpolate(ant, 0, 0, fast_access=1)
            
if __name__ == '__main__':
    unittest.main()
