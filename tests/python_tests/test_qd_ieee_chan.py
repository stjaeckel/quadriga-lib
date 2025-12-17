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
from quadriga_lib import channel, arrayant

def _snap_list(ch, key):
    v = ch[key]
    if not isinstance(v, (list, tuple)):
        raise TypeError(f'ch["{key}"] must be a list/tuple of per-snapshot arrays')
    return v

def _coeff_at(ch, snap=0):
    return np.asarray(_snap_list(ch, "coeff")[snap], dtype=np.complex128)  # (n_rx,n_tx,n_path)

def _delay_at(ch, snap=0):
    return np.asarray(_snap_list(ch, "delay")[snap], dtype=np.float64)     # (n_rx,n_tx,n_path)

def _path_gain_at(ch, snap=0):
    pg = ch["path_gain"]
    if isinstance(pg, (list, tuple)):
        return np.asarray(pg[snap], dtype=np.float64).reshape(-1)          # (n_path,)
    pg = np.asarray(pg, dtype=np.float64)
    if pg.ndim == 2:
        return pg[:, snap]
    return pg.reshape(-1)

class test_case(unittest.TestCase):

    def test_ieee_chan_a_1user_xpol(self):
        ant = arrayant.generate("xpol", 30.0)
        chan = channel.get_ieee_indoor(ant, ant, "A")

        self.assertEqual(len(chan), 1)
        ch = chan[0]

        coeff_list = _snap_list(ch, "coeff")
        delay_list = _snap_list(ch, "delay")
        self.assertEqual(len(coeff_list), 1)
        self.assertEqual(len(delay_list), 1)

        coeff0 = _coeff_at(ch, 0)  # (n_rx,n_tx,n_path)
        n_rx, n_tx, n_path = coeff0.shape
        self.assertEqual(n_tx, 2)
        self.assertEqual(n_rx, 2)
        self.assertEqual(n_path, 2)

        path_gain0 = _path_gain_at(ch, 0)
        self.assertEqual(path_gain0.shape, (2,))

        # Both paths should have equal power at KF = 0 dB
        self.assertLess(abs(path_gain0[0] - path_gain0[1]), 1e-14)

        # Calculate path gain from coefficients and compare (C++: sum(|H|^2) * 0.5)
        vc = np.abs(coeff0) ** 2  # (2,2,2)
        vv = np.sum(vc.reshape(n_rx * n_tx, n_path), axis=0) * 0.5
        npt.assert_allclose(path_gain0, vv, rtol=0.0, atol=1e-12)

        # Tx should be at the origin, facing east
        tx_pos = np.asarray(ch["tx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        tx_ori = np.asarray(ch["tx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
        npt.assert_allclose(tx_pos, np.array([0.0, 0.0, 0.0]), rtol=0.0, atol=0.0)
        npt.assert_allclose(tx_ori, np.array([0.0, 0.0, 0.0]), rtol=0.0, atol=0.0)

        # Rx should be 4.99 meters east, facing west (yaw ~ pi)
        rx_pos = np.asarray(ch["rx_position"], dtype=np.float64).reshape(3, -1)[:, 0]
        rx_ori = np.asarray(ch["rx_orientation"], dtype=np.float64).reshape(3, -1)[:, 0]
        npt.assert_allclose(rx_pos, np.array([4.99, 0.0, 0.0]), rtol=0.0, atol=0.0)
        npt.assert_allclose(rx_ori, np.array([0.0, 0.0, np.pi]), rtol=0.0, atol=1e-12)

        # Off-diagonal elements of the LOS steering matrix should be 0 due to perfect Xpol isolation
        H0 = coeff0[:, :, 0]  # LOS slice, (2,2)
        npt.assert_allclose(H0[0, 1], 0.0 + 0.0j, rtol=0.0, atol=0.0)
        npt.assert_allclose(H0[1, 0], 0.0 + 0.0j, rtol=0.0, atol=0.0)

        # Main diagonal elements should have equal power
        p00 = np.abs(H0[0, 0]) ** 2
        p11 = np.abs(H0[1, 1]) ** 2
        npt.assert_allclose(p00, p11, rtol=0.0, atol=0.0)


if __name__ == '__main__':
    unittest.main()
