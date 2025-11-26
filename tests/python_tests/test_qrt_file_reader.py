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

# Adjusted according to the Python API
from quadriga_lib import channel


class test_case(unittest.TestCase):

    def test(self):
        # Path to QRT file (mirrors C++: "tests/data/test.qrt")
        fn = os.path.join(current_dir, '../data/test.qrt')

        # --- qrt_file_parse ---

        no_cir, no_orig, no_dest, cir_offset, orig_names, dest_names = channel.qrt_file_parse(fn)

        self.assertEqual(no_orig, 3)
        self.assertEqual(no_dest, 2)
        self.assertEqual(no_cir, 7)

        self.assertEqual(len(cir_offset), no_dest)
        self.assertEqual(cir_offset[0], 0)
        self.assertEqual(cir_offset[1], 1)

        self.assertEqual(len(orig_names), 3)
        self.assertEqual(len(dest_names), 2)

        self.assertEqual(orig_names[0], "TX1")
        self.assertEqual(orig_names[1], "TX2")
        self.assertEqual(orig_names[2], "TX3")

        self.assertEqual(dest_names[0], "RX1")
        self.assertEqual(dest_names[1], "RX2")

        # --- qrt_file_read: first link (downlink, i_cir=0, i_orig=0) ---

        data_0 = channel.qrt_file_read(fn, 0, 0, True)

        self.assertEqual(data_0["center_freq"], 3.75e9)

        T = np.array([-12.9607, 59.6906, 2.0])
        npt.assert_allclose(np.asarray(data_0["tx_pos"]), T, atol=1.5e-4, rtol=0)

        T = np.array([0.0, 0.0, 0.0])
        npt.assert_allclose(np.asarray(data_0["tx_orientation"]), T, atol=1.5e-4, rtol=0)

        T = np.array([-8.83498, 57.1893, 1.0])
        npt.assert_allclose(np.asarray(data_0["rx_pos"]), T, atol=1.5e-4, rtol=0)

        # --- qrt_file_read: second link (downlink, i_cir=1, i_orig=1) ---

        data_dl = channel.qrt_file_read(fn, 1, 1, True)

        T = np.array([-2.67888, 60.257, 2.0])
        npt.assert_allclose(np.asarray(data_dl["tx_pos"]), T, atol=1.5e-4, rtol=0)

        T = np.array([0.0, 0.0, np.pi])
        npt.assert_allclose(np.asarray(data_dl["tx_orientation"]), T, atol=1.5e-4, rtol=0)

        T = np.array([-5.86144, 53.8124, 1.0])
        npt.assert_allclose(np.asarray(data_dl["rx_pos"]), T, atol=1.5e-4, rtol=0)

        T = np.array([0.0, 0.0, 1.2753])
        npt.assert_allclose(np.asarray(data_dl["rx_orientation"]), T, atol=1.5e-4, rtol=0)

        # --- qrt_file_read: uplink (i_cir=1, i_orig=1, downlink=False) ---

        data_ul = channel.qrt_file_read(fn, 1, 1, False)

        # TX / RX swap between downlink and uplink
        T = np.asarray(data_dl["rx_pos"])
        npt.assert_allclose(np.asarray(data_ul["tx_pos"]), T, atol=1.5e-4, rtol=0)

        T = np.asarray(data_dl["rx_orientation"])
        npt.assert_allclose(np.asarray(data_ul["tx_orientation"]), T, atol=1.5e-4, rtol=0)

        T = np.asarray(data_dl["tx_pos"])
        npt.assert_allclose(np.asarray(data_ul["rx_pos"]), T, atol=1.5e-4, rtol=0)

        T = np.asarray(data_dl["tx_orientation"])
        npt.assert_allclose(np.asarray(data_ul["rx_orientation"]), T, atol=1.5e-4, rtol=0)

        # FBS / LBS swap
        npt.assert_allclose(np.asarray(data_dl["fbs_pos"]),
                            np.asarray(data_ul["lbs_pos"]),
                            atol=1.5e-4, rtol=0)
        npt.assert_allclose(np.asarray(data_dl["lbs_pos"]),
                            np.asarray(data_ul["fbs_pos"]),
                            atol=1.5e-4, rtol=0)

        # AoD / AoA swap
        npt.assert_allclose(np.asarray(data_ul["aod"]),
                            np.asarray(data_dl["aoa"]),
                            atol=1.5e-4, rtol=0)
        npt.assert_allclose(np.asarray(data_ul["eod"]),
                            np.asarray(data_dl["eoa"]),
                            atol=1.5e-4, rtol=0)


if __name__ == '__main__':
    unittest.main()
