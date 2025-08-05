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

        # Minimal test
        with open('test.qdant', 'w') as f:
            f.write('<qdant><arrayant>\n')
            f.write('<name>bla</name>\n')
            f.write('<ElevationGrid>-90 -45 0 45 90</ElevationGrid>\n')
            f.write('<AzimuthGrid>-180 0 90</AzimuthGrid>\n')
            f.write('<EthetaMag>' + ' '.join(map(str, range(1, 16))) + '</EthetaMag>\n')
            f.write('</arrayant></qdant>\n')

        with self.assertRaises(ValueError) as context:
            data = arrayant.qdant_read( 'testx.qdant' )
        self.assertEqual(str(context.exception), "File was not found")

        data = arrayant.qdant_read( 'test.qdant' )

        A = 20 * np.log10(data["e_theta_re"])
        B = np.reshape(np.arange(1, 16), (5, 3, 1))
        npt.assert_almost_equal(A, B, decimal=14)

        npt.assert_almost_equal(data["e_theta_im"], np.zeros((5, 3, 1)), decimal=14)
        npt.assert_almost_equal(data["e_phi_im"], np.zeros((5, 3,1)), decimal=14)
        npt.assert_almost_equal(data["azimuth_grid"], np.array([-np.pi, 0, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["elevation_grid"], np.array([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["element_pos"][:,0], np.array([0, 0, 0]), decimal=13)
        npt.assert_almost_equal(data["coupling_re"], np.array([[1]]), decimal=13)
        npt.assert_almost_equal(data["center_freq"], 299792448, decimal=13)
        assert data["name"] == 'bla'
        assert data["layout"] == np.uint32(1)
        
        # More complex test
        with open('test.qdant', 'w') as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?><qdant xmlns:xx="test">\n')
            f.write('<xx:layout>1,1 1,1 1,1</xx:layout>\n')
            f.write('<xx:arrayant id="1">\n')
            f.write('<xx:AzimuthGrid>-90 -45 0 45 90</xx:AzimuthGrid>\n')
            f.write('<xx:ElevationGrid>-90 0 90</xx:ElevationGrid>\n')
            f.write('<xx:EphiMag>' + ' '.join(map(str, np.zeros(15))) + '</xx:EphiMag>\n')
            f.write('<xx:EphiPhase>' + ' '.join(map(str, np.ones(15) * 90)) + '</xx:EphiPhase>\n')
            f.write('<xx:EthetaMag>' + ' '.join(map(str, np.ones(15) * 3)) + '</xx:EthetaMag>\n')
            f.write('<xx:EthetaPhase>' + ' '.join(map(str, np.ones(15) * -90)) + '</xx:EthetaPhase>\n')
            f.write('<xx:ElementPosition>1,2,3</xx:ElementPosition>\n')
            f.write('<xx:CouplingAbs>1</xx:CouplingAbs>\n')
            f.write('<xx:CouplingPhase>45</xx:CouplingPhase>\n')
            f.write('<xx:CenterFrequency>3e9</xx:CenterFrequency>\n')
            f.write('</xx:arrayant></qdant>\n')

        data = arrayant.qdant_read( 'test.qdant' )

        npt.assert_almost_equal(data["azimuth_grid"], np.array([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["elevation_grid"], np.array([-np.pi / 2, 0, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["e_theta_re"], np.zeros((3, 5, 1)), decimal=13)
        npt.assert_almost_equal(data["e_theta_im"], -np.sqrt(10 ** 0.3 * np.ones((3, 5, 1))), decimal=13)
        npt.assert_almost_equal(data["e_phi_re"], np.zeros((3, 5, 1)), decimal=13)
        npt.assert_almost_equal(data["e_phi_im"], np.ones((3, 5, 1)), decimal=13)
        npt.assert_almost_equal(data["element_pos"][:,0], np.array([1, 2, 3]), decimal=13)
        npt.assert_almost_equal(data["coupling_re"], 1 / np.sqrt(2), decimal=13)
        npt.assert_almost_equal(data["coupling_im"], 1 / np.sqrt(2), decimal=13)
        npt.assert_almost_equal(data["center_freq"], 3e9, decimal=13)
        assert data["name"] == 'unknown'
        assert np.array_equal(data["layout"], np.ones((2, 3), dtype=np.uint32))

        # Two array antennas with uncommon formats
        with open('test.qdant', 'w') as f:
            f.write('<qdant><arrayant id="1">\n')
            f.write('<ElevationGrid> -45 45</ElevationGrid>\n')
            f.write('<AzimuthGrid>-90 0 90</AzimuthGrid>\n')
            f.write('<EthetaMag>\n\t1 1 1\n\t2 2 3\n</EthetaMag>\n')
            f.write('</arrayant><arrayant id="3">\n')
            f.write('<ElevationGrid> -45 45</ElevationGrid>')
            f.write('<NoElements>2</NoElements>')
            f.write('<AzimuthGrid>-90 0 90</AzimuthGrid>\n')
            f.write('<EphiMag el="2">\n\t1 2 3\n\t-1 -2 -3\n</EphiMag>\n')
            f.write('<EphiPhase el="2">90 90 90 -90 -90 -90</EphiPhase>\n')
            f.write('<CouplingAbs>1 2\n 3 4</CouplingAbs>\n')
            f.write('<CouplingPhase>45 -90\n 0 0</CouplingPhase>\n')
            f.write('<name>xxx</name>\n')
            f.write('</arrayant></qdant>\n')

        data = arrayant.qdant_read( 'test.qdant', 1 )

        B = np.sqrt(np.array([[1.26, 1.26, 1.26], [1.58, 1.58, 2]]))
        assert np.array_equal(data["layout"], np.array([[1, 3]], dtype=np.uint32))
        npt.assert_almost_equal(data["e_theta_re"][:,:,0], B, decimal=2)

        data = arrayant.qdant_read( 'test.qdant', 3 )

        B = np.array([[1], [-1]]) * np.sqrt(10 ** (np.array([[1, 2, 3], [-1, -2, -3]]) / 10))
        npt.assert_almost_equal(data["e_theta_re"], np.zeros((2, 3, 2)), decimal=13)
        npt.assert_almost_equal(data["e_phi_re"], np.zeros((2, 3, 2)), decimal=13)
        npt.assert_almost_equal(data["e_phi_im"][:,:,1], B, decimal=13)
        npt.assert_almost_equal(data["coupling_re"], np.array([[1 / np.sqrt(2), 3],[0,4]]), decimal=13)
        npt.assert_almost_equal(data["coupling_im"], np.array([[1 / np.sqrt(2), 0],[-2,0]]), decimal=13)

        os.remove('test.qdant')
            
if __name__ == '__main__':
    unittest.main()
