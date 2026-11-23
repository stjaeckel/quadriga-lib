# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.arrayant.qdant_read.

import sys
import os
import tempfile
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

        with self.assertRaises(RuntimeError):
            arrayant.qdant_read( 'testx.qdant' )

        data, layout = arrayant.qdant_read( 'test.qdant' )

        A = 20 * np.log10(data["e_theta_re"])
        B = np.reshape(np.arange(1, 16), (5, 3, 1))
        npt.assert_almost_equal(A, B, decimal=14)

        npt.assert_almost_equal(data["e_theta_im"], np.zeros((5, 3, 1)), decimal=14)
        npt.assert_almost_equal(data["e_phi_im"], np.zeros((5, 3,1)), decimal=14)
        npt.assert_almost_equal(data["azimuth_grid"], np.array([-np.pi, 0, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["elevation_grid"], np.array([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["element_pos"][:,0], np.array([0, 0, 0]), decimal=13)
        npt.assert_almost_equal(data["port_weights_re"], np.array([[1]]), decimal=13)
        npt.assert_almost_equal(data["center_freq"], 299792458, decimal=13)
        assert data["name"] == 'bla'
        assert layout == 1
        
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

        data, layout = arrayant.qdant_read( 'test.qdant' )

        npt.assert_almost_equal(data["azimuth_grid"], np.array([-np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["elevation_grid"], np.array([-np.pi / 2, 0, np.pi / 2]), decimal=13)
        npt.assert_almost_equal(data["e_theta_re"], np.zeros((3, 5, 1)), decimal=13)
        npt.assert_almost_equal(data["e_theta_im"], -np.sqrt(10 ** 0.3 * np.ones((3, 5, 1))), decimal=13)
        npt.assert_almost_equal(data["e_phi_re"], np.zeros((3, 5, 1)), decimal=13)
        npt.assert_almost_equal(data["e_phi_im"], np.ones((3, 5, 1)), decimal=13)
        npt.assert_almost_equal(data["element_pos"][:,0], np.array([1, 2, 3]), decimal=13)
        npt.assert_almost_equal(data["port_weights_re"], 1 / np.sqrt(2), decimal=13)
        npt.assert_almost_equal(data["port_weights_im"], 1 / np.sqrt(2), decimal=13)
        npt.assert_almost_equal(data["center_freq"], 3e9, decimal=13)
        assert data["name"] == 'unknown'
        assert np.array_equal(layout, np.ones((2, 3), dtype=int))

        # Two antenna arrays with uncommon formats
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

        data, layout = arrayant.qdant_read( 'test.qdant', 1 )

        B = np.sqrt(np.array([[1.26, 1.26, 1.26], [1.58, 1.58, 2]]))
        assert np.array_equal(layout, np.array([[1, 3]]))
        npt.assert_almost_equal(data["e_theta_re"][:,:,0], B, decimal=2)

        data, _ = arrayant.qdant_read( 'test.qdant', 3 )

        B = np.array([[1], [-1]]) * np.sqrt(10 ** (np.array([[1, 2, 3], [-1, -2, -3]]) / 10))
        npt.assert_almost_equal(data["e_theta_re"], np.zeros((2, 3, 2)), decimal=13)
        npt.assert_almost_equal(data["e_phi_re"], np.zeros((2, 3, 2)), decimal=13)
        npt.assert_almost_equal(data["e_phi_im"][:,:,1], B, decimal=13)
        npt.assert_almost_equal(data["port_weights_re"], np.array([[1 / np.sqrt(2), 3],[0,4]]), decimal=13)
        npt.assert_almost_equal(data["port_weights_im"], np.array([[1 / np.sqrt(2), 0],[-2,0]]), decimal=13)

        os.remove('test.qdant')

    def test_layout_dtype(self):
        """layout is returned as int64."""
        import tempfile, shutil
        d = tempfile.mkdtemp()
        try:
            fn = os.path.join(d, 'a.qdant')
            arrayant.qdant_write(fn, arrayant.generate_antenna('omni'))
            out = arrayant.qdant_read(fn)
            self.assertIsInstance(out, tuple)
            self.assertEqual(len(out), 2)
            r, layout = out
            self.assertEqual(layout.dtype, np.int64)
            npt.assert_array_equal(layout, [[1]])
        finally:
            shutil.rmtree(d, ignore_errors=True)


class test_default_id(unittest.TestCase):

    # Without 'ids' (or with ids = None) every entry is read, as in MATLAB; the entries must form a frequency set
    def test(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'freq.qdant')
            arrayant.qdant_write(fn, arrayant.generate_antenna('omni', center_freq=1e9), 1)
            arrayant.qdant_write(fn, arrayant.generate_antenna('omni', center_freq=2e9), 2)
            for r, _ in (arrayant.qdant_read(fn), arrayant.qdant_read(fn, None)):
                self.assertEqual(r['e_theta_re'].ndim, 4)
                npt.assert_array_equal(r['center_freq'], [1e9, 2e9])
            r, _ = arrayant.qdant_read(fn, 2)
            self.assertEqual(r['e_theta_re'].ndim, 3)
            self.assertEqual(r['center_freq'], 2e9)
            # IDs are 1-based: 0 is not an ID and does not mean "all"
            with self.assertRaisesRegex(ValueError, "'ids' must be >= 1"):
                arrayant.qdant_read(fn, 0)
            with self.assertRaisesRegex(TypeError, "'ids' must be an integer, a list of integers or None"):
                arrayant.qdant_read(fn, 1.5)
            npt.assert_array_equal(arrayant.qdant_read(fn, np.int64(2))[0]['center_freq'], 2e9)
            fn = os.path.join(tmp, 'two.qdant')
            arrayant.qdant_write(fn, arrayant.generate_antenna('omni'), 1)
            arrayant.qdant_write(fn, arrayant.generate_antenna('xpol'), 2)
            self.assertEqual(arrayant.qdant_read(fn, 1)[0]['e_theta_re'].shape[2], 1)
            self.assertEqual(arrayant.qdant_read(fn, 2)[0]['e_theta_re'].shape[2], 2)
            with self.assertRaisesRegex(ValueError, "do not form a frequency set.*pass 'ids'"):
                arrayant.qdant_read(fn)
            # A list or array of IDs gives a list of dicts in the given order, duplicates included
            fn = os.path.join(tmp, 'three.qdant')
            for i in (1, 2, 3):
                arrayant.qdant_write(fn, arrayant.generate_antenna('omni', center_freq=i * 1e9), i)
            r, _ = arrayant.qdant_read(fn, [3, 1, 3])
            self.assertIsInstance(r, list)
            self.assertEqual([a['center_freq'] for a in r], [3e9, 1e9, 3e9])
            r, _ = arrayant.qdant_read(fn, ids=np.array([2, 2]))
            self.assertEqual([a['center_freq'] for a in r], [2e9, 2e9])
            r, layout = arrayant.qdant_read(fn, [])  # An empty list reads no entries, the layout is still returned
            self.assertEqual(r, [])
            npt.assert_array_equal(layout, arrayant.qdant_read(fn, [1])[1])
            for bad in (0, 4, [1, 4], [1, 0]):
                with self.assertRaises(ValueError):
                    arrayant.qdant_read(fn, bad)
            fn = os.path.join(tmp, 'grids.qdant')
            arrayant.qdant_write(fn, arrayant.generate_antenna('omni'), 1)
            arrayant.qdant_write(fn, arrayant.generate_antenna('omni', res=5.0), 2)
            with self.assertRaisesRegex(ValueError, "do not form a frequency set"):
                arrayant.qdant_read(fn)


if __name__ == '__main__':
    unittest.main()
