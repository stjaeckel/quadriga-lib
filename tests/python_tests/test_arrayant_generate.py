# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for quadriga_lib.arrayant.generate.

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

        # Omni antenna, 10 deg
        data = arrayant.generate("omni", 10)

        npt.assert_almost_equal(data["e_theta_re"], np.ones((19, 37, 1)), decimal=14)
        npt.assert_almost_equal(data["e_theta_im"], np.zeros((19, 37, 1)), decimal=14)
        npt.assert_almost_equal(data["e_phi_re"], np.zeros((19, 37, 1)), decimal=14)
        npt.assert_almost_equal(data["e_phi_im"], np.zeros((19, 37, 1)), decimal=14)
        npt.assert_almost_equal(data["center_freq"], 299792458.0, decimal=14)
        npt.assert_almost_equal(data["azimuth_grid"], np.linspace(-np.pi, np.pi, 37), decimal=14)
        npt.assert_almost_equal(data["elevation_grid"], np.linspace(-np.pi/2, np.pi/2, 19), decimal=14)
        npt.assert_almost_equal(data["coupling_re"], 1.0, decimal=14)
        npt.assert_almost_equal(data["coupling_im"], 0.0, decimal=14)
        assert data["name"] == 'omni'

        # Xpol antenna
        data = arrayant.generate("xpol", 400.0, freq=3.0e9)
        npt.assert_almost_equal(data["e_theta_re"][:,:,0], np.ones((3, 5)), decimal=14)
        npt.assert_almost_equal(data["e_theta_re"][:,:,1], np.zeros((3, 5)), decimal=14)
        npt.assert_almost_equal(data["e_phi_re"][:,:,1], np.ones((3, 5)), decimal=14)
        npt.assert_almost_equal(data["e_phi_re"][:,:,0], np.zeros((3, 5)), decimal=14)
        npt.assert_almost_equal(data["e_theta_im"], np.zeros((3, 5, 2)), decimal=14)
        npt.assert_almost_equal(data["e_phi_im"], np.zeros((3, 5, 2)), decimal=14)
        npt.assert_almost_equal(data["center_freq"], 3.0e9, decimal=14)
        npt.assert_almost_equal(data["azimuth_grid"], np.linspace(-np.pi, np.pi, 5), decimal=14)
        npt.assert_almost_equal(data["elevation_grid"], np.linspace(-np.pi/2, np.pi/2, 3), decimal=14)
        assert data["name"] == 'xpol'

        # Custom antenna
        data = arrayant.generate("custom", 10, freq=90, az_3dB=90, el_3dB=0)
        assert data["e_theta_re"].shape == (19, 37, 1)
        assert data["e_theta_re"].flags.f_contiguous
        assert data["name"] == 'custom'

        # 3GPP default
        data = arrayant.generate("3gpp")
        assert data["e_theta_re"].shape == (181, 361, 1)
        npt.assert_almost_equal(data["e_theta_re"][0,0,0], 0.0794328, decimal=7)
        npt.assert_almost_equal(data["e_theta_re"][90,180,0], 2.51188, decimal=5)
        assert data["name"] == '3gpp'

        data = arrayant.generate("3gpp", pol=2)
        npt.assert_almost_equal(data["e_theta_re"][0,0,0], 0.0794328, decimal=7)
        npt.assert_almost_equal(data["e_theta_re"][90,180,0], 2.51188, decimal=5)
        npt.assert_almost_equal(data["e_phi_re"][0,0,1], -0.0794328, decimal=7)
        npt.assert_almost_equal(data["e_phi_re"][90,180,1], 2.51188, decimal=5)

        data = arrayant.generate("3gpp", pol=3)
        npt.assert_almost_equal(data["e_theta_re"][90,180,0], 1.776172, decimal=5)
        npt.assert_almost_equal(data["e_theta_re"][90,180,1], 1.776172, decimal=5)
        npt.assert_almost_equal(data["e_phi_re"][90,180,0], 1.776172, decimal=5)
        npt.assert_almost_equal(data["e_phi_re"][90,180,1], -1.776172, decimal=5)

        # Use custom pattern
        pat = arrayant.generate("custom", 10)

        data = arrayant.generate("3gpp", az_3dB=90, el_3dB=90, res=10, N=2)
        npt.assert_almost_equal(data["e_theta_re"][:,:,0], pat["e_theta_re"][:,:,0], decimal=14)
        npt.assert_almost_equal(data["e_theta_re"][:,:,1], pat["e_theta_re"][:,:,0], decimal=14)
        npt.assert_almost_equal(data["element_pos"], [[0.0,0.0],[-0.25,0.25],[0.0,0.0]], decimal=14)
    
        data = arrayant.generate("3gpp", pattern=pat, M=2)
        npt.assert_almost_equal(data["e_theta_re"][:,:,0], pat["e_theta_re"][:,:,0], decimal=14)
        npt.assert_almost_equal(data["e_theta_re"][:,:,1], pat["e_theta_re"][:,:,0], decimal=14)
        npt.assert_almost_equal(data["element_pos"], [[0.0,0.0],[0.0,0.0],[-0.25,0.25]], decimal=14)

        # Errors
        trash = {"e_theta_re":0.0}
        with self.assertRaises(KeyError) as context:
            arrayant.generate("3gpp", pattern=trash)
        self.assertEqual(str(context.exception), "'e_theta_im'")

        trash = {"e_theta_re":"Bla"}
        with self.assertRaises(ValueError) as context:
            arrayant.generate("3gpp", pattern=trash)
        self.assertEqual(str(context.exception), "could not convert string to float: 'Bla'")

        with self.assertRaises(ValueError) as context:
            arrayant.generate("bla")
        self.assertEqual(str(context.exception), "Array type not supported!")
            
    def test_none_and_flag_types(self):
        # The flags only act on 'multibeam', 'pattern' only on '3gpp' and 'ula'
        mb = dict(M=2, N=2, beam_az=[0.0, 40.0], beam_el=[0.0, 0.0], res=30)
        ref = arrayant.generate("multibeam", **mb)
        out = arrayant.generate("multibeam", **mb, separate_beams=None, apply_weights=None)
        for key in ("e_theta_re", "e_phi_re", "coupling_re"):
            npt.assert_array_equal(out[key], ref[key])
        ref = arrayant.generate("3gpp", res=30)
        for kw in ({"pattern": None}, {"pattern": {}}):
            npt.assert_array_equal(arrayant.generate("3gpp", res=30, **kw)["e_theta_re"], ref["e_theta_re"])

        for name in ("separate_beams", "apply_weights"):
            with self.assertRaises(TypeError) as context:
                arrayant.generate("multibeam", **mb, **{name: 2.5})
            self.assertIn("'" + name + "'", str(context.exception))

        with self.assertRaises(TypeError) as context:
            arrayant.generate("3gpp", pattern=[1.0])
        self.assertIn("Input 'pattern' must be a dict or None.", str(context.exception))

if __name__ == '__main__':
    unittest.main()
