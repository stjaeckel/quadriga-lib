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

import quadriga_lib

class test_hdf_rw(unittest.TestCase):

    def test(self):

        fn = 'hdf_python.hdf5'

        # Delete file
        if os.path.isfile(fn):
            os.remove(fn)

        # Try creating the file
        quadriga_lib.hdf5_create_file(fn)

        # Read Layout
        storage_space, has_data = quadriga_lib.hdf5_read_layout(fn)

        tst = np.array([65536, 1, 1, 1], dtype=np.uint32)
        npt.assert_array_equal(storage_space, tst)
        np.testing.assert_array_equal(has_data, 0)
        self.assertEqual(has_data.shape, (65536, 1, 1, 1))

        # Trying this again should fail because file exists
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_create_file(fn)
        self.assertEqual(str(context.exception), "File already exists.")

        # Try creating a file with a custom storage layout
        os.remove(fn)
        quadriga_lib.hdf5_create_file(fn, 12, 12)
        storage_space, has_data = quadriga_lib.hdf5_read_layout(fn)
        tst = np.array([12, 12, 1, 1], dtype=np.uint32)
        npt.assert_array_equal(storage_space, tst)
        self.assertEqual(has_data.shape, (12, 12, 1, 1))

        # Reshape the storage layout
        quadriga_lib.hdf5_reshape_layout(fn, 1, 1, 18, 8);
        storage_space, has_data = quadriga_lib.hdf5_read_layout(fn)
        tst = np.array([1, 1, 18, 8], dtype=np.uint32)
        npt.assert_array_equal(storage_space, tst)
        self.assertEqual(has_data.shape, (1, 1, 18, 8))

        # There should be an error if number of elements dont match
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_reshape_layout(fn, 145);
        self.assertEqual(str(context.exception), "Mismatch in number of elements in storage index.")

        os.remove(fn)

        # Calling the reshape function on a non-exisitng file should cause error
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_reshape_layout(fn, 145)
        self.assertEqual(str(context.exception), "File does not exist.")

        # Test writing unstructured data
        par = {
            "string": 'Buy Bitcoin!',
            "double_default": 21.0e6,
            "double_np": np.double(21.0e6),
            "int64": int(21e6*100e6),
            "double_Col": np.arange(0, 1.1, 0.1).reshape(-1, 1),
            "single_Col": np.arange(0, 1.1, 0.1, dtype=np.float32).reshape(-1, 1),
            "uint32_Col": np.arange(14, 19, dtype=np.uint32).reshape(-1, 1),
            "int32_Col": -np.arange(14, 19, dtype=np.int32).reshape(-1, 1),
            "uint64_Col": np.arange(14, 19, dtype=np.uint64).reshape(-1, 1),
            "int64_Col": -np.arange(14, 19, dtype=np.int64).reshape(-1, 1),
            "double_Row": np.arange(0, 1.1, 0.1),
            "single_Row": np.arange(0, 1.1, 0.1, dtype=np.float32),
            "uint32_Row": np.arange(14, 19, dtype=np.uint32),
            "int32_Row": -np.arange(14, 19, dtype=np.int32),
            "uint64_Row": np.arange(14, 19, dtype=np.uint64),
            "int64_Row": -np.arange(14, 19, dtype=np.int64),
            "double_Mat": np.random.rand(4, 4),
            "single_Mat": -np.random.rand(5, 5).astype(np.float32),
            "uint32_Mat": np.random.randint(1, 11, size=(3, 3), dtype=np.uint32),
            "int32_Mat": -np.random.randint(1, 11, size=(4, 4), dtype=np.int32),
            "uint64_Mat": np.random.randint(1, 11, size=(5, 5), dtype=np.uint64),
            "int64_Mat": np.random.randint(1, 11, size=(6, 6), dtype=np.int64),
            "double_Cube": np.random.rand(4,3,2),
            "single_Cube": -np.random.rand(5,4,3).astype(np.float32),
            "uint32_Cube": np.random.randint(1, 11, size=(3,3,4), dtype=np.uint32),
            "int32_Cube": -np.random.randint(1, 11, size=(4,5,6), dtype=np.int32),
            "uint64_Cube": np.random.randint(1, 11, size=(5,6,7), dtype=np.uint64),
            "int64_Cube": np.random.randint(1, 11, size=(6,7,8), dtype=np.int64)
        }

        storage_space = quadriga_lib.hdf5_write_channel(fn,1,1,1,1,par)
        tst = np.array([128,8,8,8], dtype=np.uint32)
        npt.assert_array_equal(storage_space, tst)

        rx_pos = np.random.rand(3, 4)
        tx_pos = np.random.rand(3, 1)
        coeff = np.random.random((2, 3, 5, 4)) + 1j * np.random.random((2, 3, 5, 4))

        quadriga_lib.hdf5_write_channel(fn,ix=2,rx_pos=rx_pos, tx_pos=tx_pos, center_frequency=1e6)

        




if __name__ == '__main__':
    unittest.main()
