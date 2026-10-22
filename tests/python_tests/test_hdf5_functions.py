# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import pickle
import shutil
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import channel


def minimal_channel():
    return {'name': 'c', 'tx_position': np.zeros((3, 1)), 'rx_position': np.ones((3, 1)),
            'center_frequency': np.array([1e9]), 'coeff': [np.ones((1, 1, 1), complex)],
            'delay': [np.zeros((1, 1, 1))]}


def strided_inputs():
    """Arrays that are not Fortran-contiguous: slices with steps, negative strides, columns, struct fields."""
    x = np.arange(1000.0)
    y = np.arange(12.0).reshape(6, 2)
    z = np.arange(60.0).reshape(3, 4, 5)
    rec = np.zeros(5, dtype=[('a', 'f8'), ('b', 'f4')])  # packed: 12 bytes per record
    rec['a'] = np.arange(5.0) + 0.5
    rec['b'] = np.arange(5.0) - 0.25
    return {
        'step2': x[::2],
        'reversed': x[::-1],
        'column': y[:, 1:2],                 # (6, 1), stride 16 bytes
        'c_order': y,                        # (6, 2), C-order
        'transposed': y.T[:, ::-1],          # (2, 6), negative column stride
        'cube_slice': z[:, ::2, 1:4],        # (3, 2, 3)
        'cube_c': z,                         # (3, 4, 5), C-order
        'field_a': rec['a'],                 # float64 with a 12-byte stride (not a multiple of 8)
        'field_b': rec['b'],                 # float32 with a 12-byte stride
        'int_step': np.arange(10, dtype=np.int32)[1::3],
        'uint64_rev': np.arange(7, dtype=np.uint64)[::-1],
    }


class test_case(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def fn(self, name):
        return os.path.join(self.dir, name)

    # ------------------------------------------------------------------
    # hdf5_create_file returns the storage layout as int64 (B-116)
    # ------------------------------------------------------------------
    def test_create_file_returns_int64(self):
        s = channel.hdf5_create_file(self.fn('a.hdf5'), 2, 3, 1, 4)
        self.assertEqual(s.dtype, np.int64)
        npt.assert_array_equal(s, [2, 3, 1, 4])

    # ------------------------------------------------------------------
    # Strided arrays are written with their values (B-005)
    # ------------------------------------------------------------------
    def test_write_dset_strided_arrays(self):
        fn = self.fn('b.hdf5')
        channel.hdf5_create_file(fn, 1)
        for name, v in strided_inputs().items():
            channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
            r = channel.hdf5_read_dset(fn, name=name)
            self.assertEqual(r.dtype, v.dtype, name)
            self.assertEqual(r.shape, v.shape, name)
            npt.assert_array_equal(r, v, err_msg=name)

    def test_write_channel_par_strided_arrays(self):
        fn = self.fn('c.hdf5')
        par = strided_inputs()
        channel.hdf5_write_channel(fn, minimal_channel(), par)
        _, par_read = channel.hdf5_read_channel(fn)
        for name, v in par.items():
            r = par_read[name]
            self.assertEqual(r.dtype, v.dtype, name)
            self.assertEqual(r.shape, v.shape, name)
            npt.assert_array_equal(r, v, err_msg=name)

    # ------------------------------------------------------------------
    # Scalars, numpy scalars, 0-d arrays and equal-but-distinct dtypes (B-097)
    # ------------------------------------------------------------------
    def test_write_dset_types(self):
        fn = self.fn('d.hdf5')
        channel.hdf5_create_file(fn, 1)
        cases = {  # name: (input, expected value, expected Python type or numpy dtype)
            'py_float': (2.5, 2.5, float),
            'py_int': (-7, -7, int),
            'py_bool': (True, 1, int),  # documented: a Python bool is stored as int64
            'py_str': ('abc', 'abc', str),
            'f32_scalar': (np.float32(1.5), 1.5, float),
            'f64_scalar': (np.float64(-2.25), -2.25, float),
            'i32_scalar': (np.int32(-3), -3, int),
            'u32_scalar': (np.uint32(4), 4, int),
            'i64_scalar': (np.int64(3), 3, int),
            'u64_scalar': (np.uint64(2**63 + 5), 2**63 + 5, int),
            'zero_d': (np.array(2.0), 2.0, float),
            'zero_d_i32': (np.array(9, dtype=np.int32), 9, int),
            'pickled_f64': (pickle.loads(pickle.dumps(np.arange(3.0))), np.arange(3.0), np.float64),
            'longlong': (np.arange(3, dtype=np.longlong), np.arange(3), np.int64),
            'ulonglong': (np.arange(3, dtype=np.ulonglong), np.arange(3), np.uint64),
            'big_endian': (np.arange(3.0).astype('>f8'), np.arange(3.0), np.float64),
            'big_endian_i4': (np.arange(4, dtype='>i4').reshape(2, 2), np.arange(4).reshape(2, 2), np.int32),
            'f32': (np.arange(3, dtype=np.float32), np.arange(3.0), np.float32),
            'u32': (np.arange(3, dtype=np.uint32), np.arange(3), np.uint32),
            'i32': (np.arange(3, dtype=np.int32), np.arange(3), np.int32),
            'u64': (np.arange(3, dtype=np.uint64), np.arange(3), np.uint64),
            'i64': (np.arange(3, dtype=np.int64), np.arange(3), np.int64),
        }
        for name, (v, expected, typ) in cases.items():
            channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
            r = channel.hdf5_read_dset(fn, name=name)
            if isinstance(typ, type) and not issubclass(typ, np.generic):
                self.assertIs(type(r), typ, name)
                self.assertEqual(r, expected, name)
            else:
                self.assertIsInstance(r, np.ndarray, name)
                self.assertEqual(r.dtype, typ, name)
                npt.assert_array_equal(r, expected, err_msg=name)

    def test_write_dset_unsupported_types(self):
        fn = self.fn('e.hdf5')
        channel.hdf5_create_file(fn, 1)
        for name, v in {'int16': np.arange(3, dtype=np.int16), 'np_bool': np.array([True, False]),
                        'complex': np.ones(2, complex), 'f16': np.ones(2, np.float16),
                        'four_d': np.ones((1, 2, 1, 2)), 'list': [1.0, 2.0]}.items():
            with self.assertRaises(ValueError, msg=name):
                channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)

    # ------------------------------------------------------------------
    # Shapes: a 1-element vector is an array of shape (1,), not a scalar;
    # 2-D arrays keep their shape, also (n, 1) and (1, 1)
    # ------------------------------------------------------------------
    def test_shapes_round_trip(self):
        fn = self.fn('f.hdf5')
        channel.hdf5_create_file(fn, 1)
        cases = {'v1': np.array([5.0]), 'v3': np.ones(3), 'm11': np.full((1, 1), 2.0), 'm31': np.ones((3, 1)),
                 'm13': np.ones((1, 3)), 'm32': np.arange(6.0).reshape(3, 2), 'c111': np.ones((1, 1, 1)),
                 'c311': np.ones((3, 1, 1)), 'c113': np.ones((1, 1, 3)), 'u1': np.array([7], dtype=np.uint32)}
        for name, v in cases.items():
            channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
            r = channel.hdf5_read_dset(fn, name=name)
            self.assertIsInstance(r, np.ndarray, name)
            self.assertEqual(r.shape, v.shape, name)
            npt.assert_array_equal(r, v, err_msg=name)

        fn2 = self.fn('g.hdf5')
        channel.hdf5_write_channel(fn2, minimal_channel(), cases)
        _, par = channel.hdf5_read_channel(fn2)
        for name, v in cases.items():
            self.assertIsInstance(par[name], np.ndarray, name)
            self.assertEqual(par[name].shape, v.shape, name)
            npt.assert_array_equal(par[name], v, err_msg=name)

    # The caller's arrays are not modified by the writers
    def test_inputs_unchanged(self):
        fn = self.fn('h.hdf5')
        channel.hdf5_create_file(fn, 1)
        x = np.asfortranarray(np.arange(12.0).reshape(3, 4))
        x0 = x.copy()
        x.flags.writeable = False  # read-only arrays are accepted
        channel.hdf5_write_dset(fn, 0, 0, 0, 0, 'x', x)
        npt.assert_array_equal(x, x0)
        npt.assert_array_equal(channel.hdf5_read_dset(fn, name='x'), x0)


class test_storage_dims_dtype(unittest.TestCase):

    # storage_dims is a copied integer output: int64 in all HDF5 functions
    def test(self):
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, 'd.hdf5')
            dims = channel.hdf5_create_file(fn, 2, 3)
            self.assertEqual(dims.dtype, np.int64)
            dims = channel.hdf5_write_channel(fn, minimal_channel())
            self.assertEqual(dims.dtype, np.int64)
            npt.assert_array_equal(dims, [2, 3, 1, 1])
            dims, has_data = channel.hdf5_read_layout(fn)
            self.assertEqual(dims.dtype, np.int64)
            npt.assert_array_equal(dims, [2, 3, 1, 1])


if __name__ == '__main__':
    unittest.main()
