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
            quadriga_lib.hdf5_reshape_layout(fn, 145)
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
            "double_Mat": np.random.rand(4, 3),
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

        # Write dataset to file
        storage_space = quadriga_lib.hdf5_write_channel(fn,1,1,1,1,par)
        tst = np.array([128,8,8,8], dtype=np.uint32)
        npt.assert_array_equal(storage_space, tst)

        # Try writing empty par - should be OK
        storage_space = quadriga_lib.hdf5_write_channel(fn,2,1,1,1,{})
        npt.assert_array_equal(storage_space, tst)

        # Read the names of the par
        par_names = quadriga_lib.hdf5_read_dset_names(fn,1,1,1,1)

        # Read the data again and compare the results
        res = quadriga_lib.hdf5_read_channel(fn,1,1,1,1)
        npt.assert_equal(len(res), 1)
        npt.assert_("par" in res, f"No unstructured data found in file!")

        for key, value in par.items():
            npt.assert_(key in par_names, f"Key '{key}' not found in list of dataset names!")
            npt.assert_(key in res["par"], f"Dataset '{key}' not found in file!")

            value_read = res["par"][key]

            if key.endswith("_Col"):
                value_read = value_read.reshape(-1, 1)

            npt.assert_equal(value, value_read)

            # Add a copy of the data to a new storage location
            quadriga_lib.hdf5_write_dset(fn, iy = 1, name=key, data=value)

        # Check if number of datasets matches
        par_names = quadriga_lib.hdf5_read_dset_names(fn,0,1)
        for key, value in par.items():
            npt.assert_(key in par_names, f"Key '{key}' not found in list of dataset names!")

        # Overwriting an exisiting dataset should cause error
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_write_dset(fn, 0,1,0,0, 'string', 'Oh no, I bought Ethereum.')
        self.assertEqual(str(context.exception), "Dataset 'par_string' already exists.")

        # Missing dataset name should cause error
        with self.assertRaises(TypeError) as context:
            quadriga_lib.hdf5_write_dset(fn, 12, value=5)

        # Reading from an empty location
        par_names = quadriga_lib.hdf5_read_dset_names(fn,12)
        npt.assert_equal(len(par_names), 0)

        # Passing a snapshot range should work fine since there is no structured data
        quadriga_lib.hdf5_read_channel(fn,1,1,1,1,[1,2])

        # Trying to write a complex number should generate an error
        with self.assertRaises(ValueError) as context:
            parC = { "complex": 1+2j }
            quadriga_lib.hdf5_write_channel(fn,3, par=parC)
        self.assertEqual(str(context.exception), "Input 'complex' has an unsupported type.")

        rx_pos = np.random.rand(3, 1)
        tx_pos = np.random.rand(3, 1)

        # Only providing rx_pos should lead to an error
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_write_channel(fn,5,rx_pos=rx_pos)
        self.assertEqual(str(context.exception), "'tx_pos' is missing or ill-formatted (must have 3 rows).")

        # This should be OK, but useless
        quadriga_lib.hdf5_write_channel(fn,5,rx_pos=rx_pos, tx_pos=tx_pos)
        
        coeff_re = np.random.random((2, 3, 5, 4))
        coeff = np.random.random((2, 3, 5, 4)) + 1j*np.random.random((2, 3, 5, 4))
        delay_4d = np.random.random((2, 3, 5, 4))

        # Passing only coeff should cause error
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_write_channel(fn,5,rx_pos=rx_pos, tx_pos=tx_pos,coeff=coeff)
        self.assertEqual(str(context.exception), "Delays are missing or incomplete.")

        # This should work fine
        quadriga_lib.hdf5_write_channel(fn,5,rx_pos=rx_pos, tx_pos=tx_pos,coeff=coeff, delay=delay_4d)
        quadriga_lib.hdf5_write_channel(fn,6,rx_pos=rx_pos, tx_pos=tx_pos,coeff=coeff_re, delay=delay_4d)

        # Test if we can restore the data
        res = quadriga_lib.hdf5_read_channel(fn, 5)

        npt.assert_almost_equal( res["tx_position"], tx_pos )
        npt.assert_almost_equal( res["rx_position"], rx_pos )
        npt.assert_almost_equal( res["coeff"], coeff )
        npt.assert_almost_equal( res["delay"], delay_4d )

        # Test if we can restore the data in reverse order
        res = quadriga_lib.hdf5_read_channel(fn, 5, snap=(2,1,0))

        npt.assert_almost_equal( res["tx_position"], tx_pos )
        npt.assert_almost_equal( res["rx_position"], rx_pos )
        npt.assert_almost_equal( res["coeff"], coeff[:, :, :, [2, 1, 0]] )
        npt.assert_almost_equal( res["delay"], delay_4d[:, :, :, [2, 1, 0]] )

        # Test out-of-bound error
        with self.assertRaises(ValueError) as context:
            quadriga_lib.hdf5_read_channel(fn, 5, snap=4)
        self.assertEqual(str(context.exception), "Snapshot index out of bound.")
        
        # Test alternative delays
        delay_2d = np.random.random((5, 4))
        quadriga_lib.hdf5_write_channel(fn,7,rx_pos=rx_pos, tx_pos=tx_pos,coeff=coeff, delay=delay_2d)
        res = quadriga_lib.hdf5_read_channel(fn, 7)

        npt.assert_almost_equal( res["delay"][0,0,:,:], delay_2d )

        # Add optional parameters
        center_frequency = 21e9
        name = 'buy_more_bitcoin'
        path_gain = np.random.random((5,4))
        path_length = np.random.random((5,4))
        path_polarization = np.random.random((4,5,4)) + 1j * np.random.random((4,5,4))
        path_angles = np.random.random((5,4,4))
        fbs_pos = np.random.random((3,5,4))
        lbs_pos = np.random.random((3,5,4))
        no_interact = np.array([
            [1, 2, 3, 4, 5],
            [5, 4, 3, 2, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1]
        ])
        interact_coord = np.random.random((3,15,4))
        interact_coord[:, 5:, 2] = 0.0
        interact_coord[:, 5:, 3] = 0.0
        rx_orientation = np.random.random((3,4))
        tx_orientation = np.random.random((3,4))

        quadriga_lib.hdf5_write_channel(fn, 8, rx_pos = rx_pos, tx_pos = tx_pos, coeff = coeff, delay = delay_4d,
                                        center_frequency = center_frequency, name = name, path_gain = path_gain,
                                        path_length = path_length, path_polarization = path_polarization, 
                                        path_angles = path_angles, path_fbs_pos = fbs_pos, path_lbs_pos = lbs_pos,
                                        no_interact = no_interact.T, interact_coord = interact_coord,
                                        rx_orientation = rx_orientation, tx_orientation = tx_orientation )

        # Test if we can restore the data
        res = quadriga_lib.hdf5_read_channel(fn, 8)

        npt.assert_almost_equal( res["tx_position"], tx_pos )
        npt.assert_almost_equal( res["rx_position"], rx_pos )
        npt.assert_almost_equal( res["coeff"], coeff )
        npt.assert_almost_equal( res["delay"], delay_4d )
        npt.assert_allclose( res["center_freq"], center_frequency)
        self.assertEqual(res["name"], name)
        npt.assert_equal( res["initial_pos"], 0 )
        npt.assert_almost_equal( res["path_gain"], path_gain )
        npt.assert_almost_equal( res["path_length"], path_length )
        npt.assert_almost_equal( res["polarization"], path_polarization )
        npt.assert_almost_equal( res["path_angles"], path_angles )
        npt.assert_almost_equal( res["path_fbs_pos"], fbs_pos )
        npt.assert_almost_equal( res["path_lbs_pos"], lbs_pos )
        npt.assert_equal( res["no_interact"], no_interact.T )
        npt.assert_almost_equal( res["interact_coord"], interact_coord )
        npt.assert_almost_equal( res["tx_orientation"], tx_orientation )
        npt.assert_almost_equal( res["rx_orientation"], rx_orientation )

        # Test if we can restore a single snapsot
        res = quadriga_lib.hdf5_read_channel(fn, 8, snap = 2)

        npt.assert_almost_equal( res["tx_position"], tx_pos )
        npt.assert_almost_equal( res["rx_position"], rx_pos )
        npt.assert_almost_equal( res["coeff"][:,:,:,0], coeff[:,:,:,2] )
        npt.assert_almost_equal( res["delay"][:,:,:,0], delay_4d[:,:,:,2] )
        npt.assert_allclose( res["center_freq"], center_frequency)
        self.assertEqual(res["name"], name)
        npt.assert_equal( res["initial_pos"], 0 )
        npt.assert_almost_equal( res["path_gain"][:,0], path_gain[:,2] )
        npt.assert_almost_equal( res["path_length"][:,0], path_length[:,2] )
        npt.assert_almost_equal( res["polarization"][:,:,0], path_polarization[:,:,2] )
        npt.assert_almost_equal( res["path_angles"][:,:,0], path_angles[:,:,2] )
        npt.assert_almost_equal( res["path_fbs_pos"][:,:,0], fbs_pos[:,:,2] )
        npt.assert_almost_equal( res["path_lbs_pos"][:,:,0], lbs_pos[:,:,2] )
        npt.assert_equal( res["no_interact"][:,0], no_interact[2,:].T )
        npt.assert_almost_equal( res["interact_coord"][:,:,0], interact_coord[:,:,2] )
        npt.assert_almost_equal( res["tx_orientation"][:,0], tx_orientation[:,2] )
        npt.assert_almost_equal( res["rx_orientation"][:,0], rx_orientation[:,2] )



if __name__ == '__main__':
    unittest.main()
