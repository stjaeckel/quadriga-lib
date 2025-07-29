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
        # Helper constants
        deg2rad = np.pi / 180.0
        freq = 3.75e9

        ant = arrayant.generate(
            type="multibeam",
            M=6,
            N=6,
            beam_az=[20.0, 0.0],
            beam_el=[-7.0, 30.0],
            beam_weight=[2.0, 1.0],
            freq=freq,
            pol=1,
            spacing=0.4,
            az_3dB=120.0,
            el_3dB=120.0,
            rear_gain_lin=0.0,
            res=15.0,
            separate_beams=False,
            apply_weights=False,
        )

        self.assertEqual(ant["e_theta_re"].shape, (13, 25, 36))
        self.assertEqual(ant["coupling_re"].shape, (36, 1))

        azg = np.arange(-180.0, 181.0, 1.0, dtype=float) * deg2rad
        elg = np.arange(-90.0, 91.0, 1.0, dtype=float) * deg2rad

        comb = arrayant.combine_pattern(ant, azimuth_grid=azg, elevation_grid=elg)
        self.assertEqual(comb["e_theta_re"].shape, (181, 361, 1))

        a = comb["e_theta_re"][83, 199, 0]
        b = comb["e_theta_im"][83, 199, 0]
        gain = 10.0 * np.log10(a * a + b * b)

        self.assertTrue(abs(gain - 35.6992) < 1e-4)
        self.assertTrue(abs(a + 34.7096) < 0.01)
        self.assertTrue(abs(b + 50.0993) < 0.01)

        # --- Generate the same pattern from 3 unweighted beams ---
        ant2 = arrayant.generate(
            type="multibeam",
            M=6,
            N=6,
            beam_az=[20.0, 0.0, 20.0],
            beam_el=[-7.0, 30.0, -7.0],
            freq=freq,
            pol=1,
            spacing=0.4,
            az_3dB=120.0,
            el_3dB=120.0,
            rear_gain_lin=0.0,
            res=15.0,
            separate_beams=False,
            apply_weights=False,
        )
        comb2 = arrayant.combine_pattern(ant2, azimuth_grid=azg, elevation_grid=elg)

        npt.assert_allclose(ant["coupling_re"], ant2["coupling_re"], atol=1e-6, rtol=0)
        npt.assert_allclose(ant["coupling_im"], ant2["coupling_im"], atol=1e-6, rtol=0)

        npt.assert_allclose(comb["e_theta_re"], comb2["e_theta_re"], atol=1e-6, rtol=0)
        npt.assert_allclose(comb["e_theta_im"], comb2["e_theta_im"], atol=1e-6, rtol=0)

        # --- Generate combined pattern directly at 1° resolution ---
        comb3 = arrayant.generate(
            type="multibeam",
            M=6,
            N=6,
            beam_az=[20.0, 0.0, 20.0],
            beam_el=[-7.0, 30.0, -7.0],
            freq=freq,
            pol=1,
            spacing=0.4,
            az_3dB=120.0,
            el_3dB=120.0,
            rear_gain_lin=0.0,
            res=1.0,
            separate_beams=False,
            apply_weights=True,   # directly apply/compile beams on the 1° grid
        )

        self.assertEqual(comb3["e_theta_re"].shape, (181, 361, 1))

        a3 = comb3["e_theta_re"][83, 199, 0]
        b3 = comb3["e_theta_im"][83, 199, 0]
        gain3 = 10.0 * np.log10(a3 * a3 + b3 * b3)

        self.assertTrue(abs(gain3 - 35.6992) < 0.1)
        self.assertTrue(abs(a3 - (-34.7096)) < 0.5)
        self.assertTrue(abs(b3 - (-50.0993)) < 0.5)

        # --- Test generation of separate beams (expect 3 ports) ---
        ant4 = arrayant.generate(
            type="multibeam",
            M=6,
            N=6,
            beam_az=[20.0, 0.0, 20.0],
            beam_el=[-7.0, 30.0, -7.0],
            freq=freq,
            pol=1,
            spacing=0.4,
            az_3dB=120.0,
            el_3dB=120.0,
            rear_gain_lin=0.0,
            res=15.0,
            separate_beams=True,
            apply_weights=False,
        )

        self.assertEqual(ant4["e_theta_re"].shape, (13, 25, 36))
        self.assertEqual(ant4["coupling_re"].shape, (36, 3))

        # --- H/V Polarization (pol = 2) ---
        ant5 = arrayant.generate(
            type="multibeam",
            M=6,
            N=6,
            beam_az=[20.0, 0.0, 20.0],
            beam_el=[-7.0, 30.0, -7.0],
            freq=freq,
            pol=2,                # H/V
            spacing=0.4,
            az_3dB=120.0,
            el_3dB=120.0,
            rear_gain_lin=0.0,
            res=10.0,
            separate_beams=False,
            apply_weights=False,
        )

        self.assertEqual(ant5["e_theta_re"].shape, (19, 37, 72))
        self.assertEqual(ant5["coupling_re"].shape, (72, 2))

        # element_pos: first 36 and second 36 elements should match
        x = ant5["element_pos"][:, 0:36]
        y = ant5["element_pos"][:, 36:72]
        npt.assert_allclose(x, y, atol=1e-14, rtol=0)

        # Coupling: port 0 (elements 0..35) equals port 1 (elements 36..71)
        y1 = ant5["coupling_re"][36:72, 1]
        x1 = ant5["coupling_re"][0:36, 0]
        npt.assert_allclose(x1, y1, atol=1e-14, rtol=0)

        # Cross-polar coupling should be ~0
        zeros = np.zeros((36,), dtype=float)
        npt.assert_allclose(ant5["coupling_re"][0:36, 1], zeros, atol=1e-14, rtol=0)
        npt.assert_allclose(ant5["coupling_re"][36:72, 0], zeros, atol=1e-14, rtol=0)


if __name__ == "__main__":
    unittest.main()
