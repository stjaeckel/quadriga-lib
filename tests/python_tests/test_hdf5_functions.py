# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Python interface test for the HDF5 functions in quadriga_lib.channel: hdf5_create_file, hdf5_read_layout,
# hdf5_reshape_layout, hdf5_write_channel, hdf5_read_channel, hdf5_write_dset, hdf5_read_dset, hdf5_read_dset_names.
# Scope: argument and output handling, defaults, accepted input types and shapes, error paths.
# The file format and the core behavior are tested in the Catch2 suite (test_hdf_functions.cpp).

import sys
import os
import copy
import pickle
import shutil
import tempfile
import unittest
import warnings
import numpy as np
import numpy.testing as npt

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

from quadriga_lib import channel


def minimal_channel(name='c'):
    return {'name': name, 'tx_position': np.zeros((3, 1)), 'rx_position': np.ones((3, 1)),
            'center_frequency': np.array([1e9]), 'coeff': [np.ones((1, 1, 1), complex)],
            'delay': [np.zeros((1, 1, 1))]}


def full_channel(n_path=(3, 2)):
    """All channel keys, list model, one snapshot per entry of n_path. The values are exact in single precision,
    the precision of the structured data in the file, so that they read back unchanged."""
    rng = np.random.default_rng(0)

    def q(*shape):
        return rng.standard_normal(shape).astype(np.float32).astype(np.float64)

    return {
        'name': 'full',
        'tx_position': q(3, len(n_path)),
        'rx_position': q(3, 1),
        'tx_orientation': q(3, 1),
        'rx_orientation': q(3, len(n_path)),
        'coeff': [q(2, 3, n) + 1j * q(2, 3, n) for n in n_path],
        'delay': [q(2, 3, n) for n in n_path],
        'path_gain': [q(n) for n in n_path],
        'path_length': [q(n) for n in n_path],
        'path_polarization': [q(4, n) + 1j * q(4, n) for n in n_path],
        'path_angles': [q(n, 4) for n in n_path],
        'fbs_pos': [q(3, n) for n in n_path],
        'lbs_pos': [q(3, n) for n in n_path],
        'no_interact': [np.arange(n, dtype=np.int64) for n in n_path],
        'interact_coord': [q(3, n * (n - 1) // 2) for n in n_path],
        'center_frequency': 1e9 * np.arange(1.0, len(n_path) + 1.0),
        'initial_position': 2,
    }


def stack_model(chan):
    """The same channel in the stack model: per-snapshot lists become one array with the snapshot on the last axis
    (requires the same number of paths in all snapshots)."""
    return {k: np.stack(v, axis=-1) if isinstance(v, list) else v for k, v in chan.items()}


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


def typed_values():
    """Unstructured data of all supported types: name -> (input, expected value, expected Python type or dtype)."""
    return {
        'py_float': (2.5, 2.5, float),
        'py_int': (-7, -7, int),
        'py_bool': (True, 1, int),  # documented: a Python bool is stored as int64
        'py_str': ('abc', 'abc', str),
        'py_str_empty': ('', '', str),
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
        'f64_3d': (np.arange(24.0).reshape(2, 3, 4), np.arange(24.0).reshape(2, 3, 4), np.float64),
    }


class test_case(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def fn(self, name):
        return os.path.join(self.dir, name)

    def assert_same(self, a, b, where=''):
        """Exact comparison of nested dicts / lists / arrays, including the array dtypes and shapes."""
        if isinstance(a, dict):
            self.assertIsInstance(b, dict, where)
            self.assertEqual(sorted(a.keys()), sorted(b.keys()), where)
            for k in a:
                self.assert_same(a[k], b[k], where + '/' + str(k))
        elif isinstance(a, list):
            self.assertIsInstance(b, list, where)
            self.assertEqual(len(a), len(b), where)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assert_same(x, y, where + '[' + str(i) + ']')
        elif isinstance(a, np.ndarray):
            self.assertIsInstance(b, np.ndarray, where)
            self.assertEqual(a.dtype, b.dtype, where)
            self.assertEqual(a.shape, b.shape, where)
            npt.assert_array_equal(a, b, err_msg=where)
        else:
            self.assertIs(type(a), type(b), where)
            self.assertEqual(a, b, where)

    def assert_value(self, r, expected, typ, name):
        if isinstance(typ, type) and not issubclass(typ, np.generic):
            self.assertIs(type(r), typ, name)
            self.assertEqual(r, expected, name)
        else:
            self.assertIsInstance(r, np.ndarray, name)
            self.assertEqual(r.dtype, typ, name)
            npt.assert_array_equal(r, expected, err_msg=name)

    # ------------------------------------------------------------------
    # hdf5_create_file
    # ------------------------------------------------------------------
    def test_create_file(self):
        fn = self.fn('create.hdf5')
        dims = channel.hdf5_create_file(fn)  # default layout
        self.assertIsInstance(dims, np.ndarray)
        self.assertEqual(dims.dtype, np.int64)
        npt.assert_array_equal(dims, [65536, 1, 1, 1])
        npt.assert_array_equal(channel.hdf5_read_layout(fn)[0], [65536, 1, 1, 1])

        # Each argument reaches its dimension, positionally and by keyword
        dims = channel.hdf5_create_file(self.fn('create_pos.hdf5'), 2, 3, 4, 5)
        npt.assert_array_equal(dims, [2, 3, 4, 5])
        npt.assert_array_equal(channel.hdf5_read_layout(self.fn('create_pos.hdf5'))[0], [2, 3, 4, 5])
        dims = channel.hdf5_create_file(fn=self.fn('create_kw.hdf5'), nw=5, nz=4, ny=3, nx=2)
        npt.assert_array_equal(dims, [2, 3, 4, 5])
        npt.assert_array_equal(channel.hdf5_read_layout(self.fn('create_kw.hdf5'))[0], [2, 3, 4, 5])

        # numpy integers are accepted; omitted trailing dimensions are 1
        dims = channel.hdf5_create_file(self.fn('create_np.hdf5'), np.uint32(3), np.int64(2))
        self.assertEqual(dims.dtype, np.int64)
        npt.assert_array_equal(dims, [3, 2, 1, 1])

    def test_create_file_errors(self):
        fn = self.fn('create_err.hdf5')
        channel.hdf5_create_file(fn, 2)
        with self.assertRaises(ValueError) as ctx:  # thrown by the wrapper
            channel.hdf5_create_file(fn, 2)
        self.assertIn('File already exists.', str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:  # thrown by C++
            channel.hdf5_create_file(self.fn('zero.hdf5'), 2, 0)
        self.assertIn('cannot be 0', str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            channel.hdf5_create_file(self.fn('big.hdf5'), 65536, 65536)
        self.assertIn('too many slots', str(ctx.exception))

        # Rejected by pybind11: negative and non-integer sizes, missing or unknown arguments
        for args, kwargs in [((self.fn('x.hdf5'), -1), {}), ((self.fn('x.hdf5'), 2.0), {}), ((), {}),
                             ((self.fn('x.hdf5'),), {'no_such_argument': 1})]:
            with self.assertRaises(TypeError):
                channel.hdf5_create_file(*args, **kwargs)

    # ------------------------------------------------------------------
    # hdf5_read_layout
    # ------------------------------------------------------------------
    def test_read_layout(self):
        fn = self.fn('layout.hdf5')

        # Missing file: zero dimensions, empty mask
        out = channel.hdf5_read_layout(fn)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        dims, has_data = out
        self.assertEqual(dims.dtype, np.int64)
        npt.assert_array_equal(dims, [0, 0, 0, 0])
        self.assertEqual(has_data.dtype, np.uint32)
        self.assertEqual(has_data.shape, (0, 0, 0, 0))

        # The occupancy mask has the layout shape and marks the written slot (non-symmetric index)
        channel.hdf5_create_file(fn, 3, 2, 2, 2)
        channel.hdf5_write_channel(fn, minimal_channel(), ix=2, iy=1, iz=0, iw=1)
        dims, has_data = channel.hdf5_read_layout(fn=fn)
        self.assertEqual(dims.dtype, np.int64)
        npt.assert_array_equal(dims, [3, 2, 2, 2])
        self.assertEqual(has_data.dtype, np.uint32)
        expected = np.zeros((3, 2, 2, 2), dtype=np.uint32)
        expected[2, 1, 0, 1] = 1
        npt.assert_array_equal(has_data, expected)

    def test_read_layout_errors(self):
        fn = self.fn('not_hdf5.txt')
        with open(fn, 'w') as f:
            f.write('This is not HDF5 data')
        with self.assertRaises(ValueError) as ctx:
            channel.hdf5_read_layout(fn)
        self.assertIn('Not an HDF5 file', str(ctx.exception))
        with self.assertRaises(TypeError):
            channel.hdf5_read_layout()

    # ------------------------------------------------------------------
    # hdf5_reshape_layout
    # ------------------------------------------------------------------
    def test_reshape_layout(self):
        fn = self.fn('reshape.hdf5')
        channel.hdf5_create_file(fn, 4, 3)
        channel.hdf5_write_channel(fn, minimal_channel(), ix=1, iy=2)  # linear slot 1 + 4 * 2 = 9

        self.assertIsNone(channel.hdf5_reshape_layout(fn, 2, 3, 1, 2))
        dims, has_data = channel.hdf5_read_layout(fn)
        npt.assert_array_equal(dims, [2, 3, 1, 2])
        self.assertEqual(has_data[1, 1, 0, 1], 1)  # slot 9 in the new layout; the data is not moved

        channel.hdf5_reshape_layout(fn=fn, nx=12)  # omitted dimensions are 1
        dims, has_data = channel.hdf5_read_layout(fn)
        npt.assert_array_equal(dims, [12, 1, 1, 1])
        self.assertEqual(has_data[9, 0, 0, 0], 1)
        self.assertEqual(channel.hdf5_read_channel(fn, 9)[0][0]['name'], 'c')

        # Default layout: 65536 slots in x
        fn = self.fn('reshape_default.hdf5')
        channel.hdf5_create_file(fn, 256, 256)
        channel.hdf5_reshape_layout(fn)
        npt.assert_array_equal(channel.hdf5_read_layout(fn)[0], [65536, 1, 1, 1])

    def test_reshape_layout_errors(self):
        fn = self.fn('reshape_err.hdf5')
        with self.assertRaises(ValueError) as ctx:
            channel.hdf5_reshape_layout(fn, 4)
        self.assertIn('File does not exist.', str(ctx.exception))
        channel.hdf5_create_file(fn, 4, 3)
        with self.assertRaises(ValueError) as ctx:
            channel.hdf5_reshape_layout(fn, 13)
        self.assertIn('Mismatch in number of elements', str(ctx.exception))
        with self.assertRaises(TypeError):
            channel.hdf5_reshape_layout(fn, -12)
        with self.assertRaises(TypeError):
            channel.hdf5_reshape_layout()

    # ------------------------------------------------------------------
    # hdf5_write_channel
    # ------------------------------------------------------------------
    def test_write_channel(self):
        # A single dict goes to slot 0 of a new file sized for it
        fn = self.fn('write_single.hdf5')
        dims = channel.hdf5_write_channel(fn, minimal_channel())
        self.assertEqual(dims.dtype, np.int64)
        npt.assert_array_equal(dims, [1, 1, 1, 1])

        # A list goes to the slots 0 ... n-1 in x
        fn = self.fn('write_list.hdf5')
        dims = channel.hdf5_write_channel(fn, [minimal_channel(n) for n in 'abc'])
        npt.assert_array_equal(dims, [3, 1, 1, 1])
        chan, par = channel.hdf5_read_channel(fn)
        self.assertEqual([c['name'] for c in chan], ['a', 'b', 'c'])
        self.assertEqual(par, [{}, {}, {}])

        # A new file is sized from the indices; vector indices of any integer type, scalars are broadcast
        fn = self.fn('write_idx.hdf5')
        dims = channel.hdf5_write_channel(fn, [minimal_channel('a'), minimal_channel('b')], None,
                                          [0, 3], 1, np.uint64(2), np.array([0, 1], dtype=np.int32))
        npt.assert_array_equal(dims, [4, 2, 3, 2])
        expected = np.zeros((4, 2, 3, 2), dtype=np.uint32)
        expected[0, 1, 2, 0] = expected[3, 1, 2, 1] = 1
        npt.assert_array_equal(channel.hdf5_read_layout(fn)[1], expected)

        # Keyword arguments; an existing file keeps its layout, which is returned
        fn = self.fn('write_kw.hdf5')
        channel.hdf5_create_file(fn, 2, 3)
        dims = channel.hdf5_write_channel(fn=fn, chan=minimal_channel('k'), par={'x': 1.0},
                                          ix=1, iy=np.array([2]), iz=0, iw=None)
        self.assertEqual(dims.dtype, np.int64)
        npt.assert_array_equal(dims, [2, 3, 1, 1])
        chan, par = channel.hdf5_read_channel(fn, 1, 2, 0, 0)
        self.assertEqual(chan[0]['name'], 'k')
        self.assertEqual(par, {'x': 1.0})

    def test_write_channel_par(self):
        fn = self.fn('write_par.hdf5')

        # One dict per channel; None values are skipped; the dicts may have different keys
        chans = [minimal_channel('a'), minimal_channel('b')]
        channel.hdf5_write_channel(fn, chans, [{'x': 1.0, 'skip': None}, {'y': 'text', 'z': np.arange(2.0)}])
        chan, par = channel.hdf5_read_channel(fn)
        self.assertEqual(len(par), 2)
        self.assert_same(par[0], {'x': 1.0})
        self.assert_same(par[1], {'y': 'text', 'z': np.arange(2.0)})
        self.assertEqual(channel.hdf5_read_dset_names(fn, 1), ['y', 'z'])

        # A par dict without structured data (a channel with only a name holds no data of its own)
        fn = self.fn('write_par_only.hdf5')
        channel.hdf5_write_channel(fn, {'name': 'par_only'}, {'v': np.int32(5)})
        chan, par = channel.hdf5_read_channel(fn)
        self.assertEqual(len(chan), 1)
        self.assert_same(par, {'v': 5})

        # All supported types, shapes and memory layouts pass through the par dict unchanged
        fn = self.fn('write_par_types.hdf5')
        values = typed_values()
        par_in = {name: v for name, (v, _, _) in values.items()}
        par_in.update(strided_inputs())
        channel.hdf5_write_channel(fn, minimal_channel(), par_in)
        _, par = channel.hdf5_read_channel(fn)
        self.assertEqual(list(par.keys()), list(par_in.keys()))  # creation order is kept
        for name, (_, expected, typ) in values.items():
            self.assert_value(par[name], expected, typ, name)
        for name, v in strided_inputs().items():
            self.assert_value(par[name], v, v.dtype, name)
            self.assertEqual(par[name].shape, v.shape, name)

    def test_write_channel_overwrite(self):
        fn = self.fn('overwrite.hdf5')
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            channel.hdf5_write_channel(fn, minimal_channel('old'), {'a': 1.0})
            channel.hdf5_write_dset(fn, name='b', data=2.0)
            self.assertEqual(len(w), 0)
            channel.hdf5_write_channel(fn, minimal_channel('new'))
            self.assertEqual(len(w), 1)
            self.assertIs(w[0].category, UserWarning)
            self.assertIn('overwriting existing dataset', str(w[0].message))

        # The slot is replaced completely, also the datasets written with hdf5_write_dset
        chan, par = channel.hdf5_read_channel(fn, 0, 0, 0, 0)
        self.assertEqual(chan[0]['name'], 'new')
        self.assertEqual(par, {})

    def test_write_channel_errors(self):
        fn = self.fn('write_err.hdf5')
        channel.hdf5_create_file(fn, 2, 2)
        c = minimal_channel()
        cases = [  # (chan, keyword arguments, expected message); all std::invalid_argument -> ValueError
            ([], {}, "Input 'chan' must contain at least one channel."),
            (None, {}, "Input 'chan' must contain at least one channel."),
            ('abc', {}, 'Channel input must be a dict or a list of dicts.'),
            (['abc'], {}, 'Each channel list entry must be a dict.'),
            (dict(c, coeff_re=[np.ones((1, 1, 1))]), {}, "provide either 'coeff' (complex) or 'coeff_re'/'coeff_im'"),
            (dict(c, tx_position=np.zeros((2, 1))), {}, "'tx_pos' is missing or ill-formatted"),  # from C++
            (dict(c, name='x' * 256), {}, 'Name can have at most 255 characters.'),
            (c, {'par': 'abc'}, "'par' must be a dict, a list of dicts, or None."),
            (c, {'par': [1]}, "Each 'par' list entry must be a dict."),
            (c, {'par': [{}, {}]}, "'par' must have the same number of entries as 'chan'."),
            (c, {'par': {'x': np.ones(2, complex)}}, "Input 'x' has an unsupported type."),
            (c, {'par': {'a-b': 1.0}}, 'Parameter name must only contain letters, numbers and the underscore'),
            ([c, c, c], {'ix': [0, 1]}, "Input 'ix' must be a scalar or a vector of length len(chan)."),
            ([c, c], {'ix': 0, 'iw': [0, 0, 0]}, "Input 'iw' must be a scalar or a vector of length len(chan)."),
            (c, {'iy': 2}, 'Index out of bound.'),
        ]
        for chan, kwargs, msg in cases:
            with self.assertRaises(ValueError, msg=msg) as ctx:
                channel.hdf5_write_channel(fn, chan, **kwargs)
            self.assertIn(msg, str(ctx.exception))
        self.assertEqual(channel.hdf5_read_layout(fn)[1].sum(), 0)  # nothing was written

        with self.assertRaises(TypeError):
            channel.hdf5_write_channel(fn)
        with self.assertRaises(TypeError):
            channel.hdf5_write_channel(fn, c, no_such_argument=1)

    # ------------------------------------------------------------------
    # Channel dict: all keys, both layouts, input types
    # ------------------------------------------------------------------
    def test_channel_dict_round_trip(self):
        fn = self.fn('dict.hdf5')
        chan_in = full_channel()
        chan_ref = copy.deepcopy(chan_in)
        channel.hdf5_write_channel(fn, chan_in)
        self.assert_same(chan_in, chan_ref)  # inputs are not modified

        out = channel.hdf5_read_channel(fn)
        self.assertIsInstance(out, tuple)
        self.assertEqual(len(out), 2)
        chan, par = out
        self.assertIsInstance(chan, list)
        self.assertEqual(len(chan), 1)
        self.assertEqual(par, {})  # single slot: one dict

        # Every key comes back with its documented type, dtype and shape and the written values
        self.assert_same(chan[0], chan_ref)

        # Stack model: one array per key with the snapshot on the last axis, zero-padded to the largest n_path
        chan, _ = channel.hdf5_read_channel(fn, stack=True)
        c = chan[0]
        shapes = {'coeff': (2, 3, 3, 2), 'delay': (2, 3, 3, 2), 'path_gain': (3, 2), 'path_length': (3, 2),
                  'path_polarization': (4, 3, 2), 'path_angles': (3, 4, 2), 'fbs_pos': (3, 3, 2),
                  'lbs_pos': (3, 3, 2), 'no_interact': (3, 2), 'interact_coord': (3, 3, 2)}
        for key, shape in shapes.items():
            self.assertEqual(c[key].shape, shape, key)
            self.assertEqual(c[key].dtype, chan_ref[key][0].dtype, key)
            npt.assert_array_equal(c[key][..., 0], chan_ref[key][0], err_msg=key)
            ref = np.zeros(shape[:-1], dtype=c[key].dtype)  # snapshot 1: 2 paths, 1 interaction, zero-padded
            ref[tuple(slice(0, n) for n in chan_ref[key][1].shape)] = chan_ref[key][1]
            npt.assert_array_equal(c[key][..., 1], ref, err_msg=key)
        for key in ['name', 'tx_position', 'rx_position', 'tx_orientation', 'rx_orientation', 'center_frequency',
                    'initial_position']:
            self.assert_same(c[key], chan_ref[key], key)

    def test_channel_dict_input_forms(self):
        fn = self.fn('forms.hdf5')
        ref = full_channel(n_path=(3, 3))
        f32 = {k: [x.astype(np.complex64 if np.iscomplexobj(x) else np.float32) for x in v]
               if isinstance(v, list) and k != 'no_interact' else v for k, v in ref.items()}
        f32.update({k: ref[k].astype(np.float32) for k in ['tx_position', 'rx_position', 'center_frequency']})
        variants = [
            ref,                                                                      # list model
            stack_model(ref),                                                         # stack model
            {k: np.asfortranarray(v) if isinstance(v, np.ndarray) else v              # Fortran order
             for k, v in stack_model(ref).items()},
            {k: np.repeat(v, 2, axis=-1)[..., ::2] if isinstance(v, np.ndarray) else v  # non-contiguous views
             for k, v in stack_model(ref).items()},
            f32,                                                                      # single precision
            dict({k: v for k, v in ref.items() if k != 'coeff'},                      # split coefficients
                 coeff_re=[x.real for x in ref['coeff']], coeff_im=[x.imag for x in ref['coeff']]),
            dict(ref, no_interact=[x.astype(np.uint32) for x in ref['no_interact']],  # other integer types
                 initial_position=np.int32(2)),
        ]
        channel.hdf5_write_channel(fn, variants)
        chan, par = channel.hdf5_read_channel(fn)
        self.assertEqual(len(chan), len(variants))
        self.assertEqual(par, [{}] * len(variants))  # several slots: list of dicts
        for i, c in enumerate(chan):
            self.assert_same(c, ref, 'variant %d' % i)

    # ------------------------------------------------------------------
    # hdf5_read_channel
    # ------------------------------------------------------------------
    def test_read_channel_selection(self):
        fn = self.fn('select.hdf5')
        channel.hdf5_create_file(fn, 3, 2, 1, 2)
        for name, idx in [('c', (1, 1, 0, 1)), ('b', (2, 0, 0, 0)), ('a', (0, 0, 0, 0))]:
            channel.hdf5_write_channel(fn, minimal_channel(name), {'p': name}, *idx)

        def names(*args, **kwargs):
            chan, par = channel.hdf5_read_channel(fn, *args, **kwargs)
            self.assertIsInstance(par, list)
            self.assertEqual([p['p'] for p in par], [c['name'] for c in chan])
            return [c['name'] for c in chan]

        self.assertEqual(names(), ['a', 'b', 'c'])                     # all slots in column-major order
        self.assertEqual(names(None, None, None, None), ['a', 'b', 'c'])
        self.assertEqual(names(2), ['b'])                               # other dimensions: full range
        self.assertEqual(names([2, 0], 0), ['b', 'a'])                  # the given order is kept
        self.assertEqual(names(np.array([1, 2]), np.uint64(1), iw=np.array([1], dtype=np.int32)), ['c'])
        self.assertEqual(names(ix=1, iy=1, iz=[0, 0], iw=1), ['c', 'c'])

        # A single slot gives one par dict; keywords reach their dimension
        chan, par = channel.hdf5_read_channel(fn=fn, ix=1, iy=1, iz=0, iw=1, snap=None, stack=False)
        self.assertEqual(chan[0]['name'], 'c')
        self.assertEqual(par, {'p': 'c'})

        # Empty slots are skipped
        self.assertEqual(channel.hdf5_read_channel(fn, 1, 0, 0, 0), ([], {}))
        self.assertEqual(channel.hdf5_read_channel(fn, 1, 0), ([], []))

    def test_read_channel_snap(self):
        fn = self.fn('snap.hdf5')
        ref = full_channel()
        channel.hdf5_write_channel(fn, ref)

        chan, _ = channel.hdf5_read_channel(fn, 0, 0, 0, 0, [1, 0])  # 0-based, reversed order
        c = chan[0]
        for key in ['coeff', 'delay', 'path_gain', 'path_polarization', 'no_interact', 'interact_coord']:
            self.assert_same(c[key], [ref[key][1], ref[key][0]], key)
        self.assert_same(c['tx_position'], ref['tx_position'][:, [1, 0]])  # per-snapshot columns follow
        self.assert_same(c['rx_position'], ref['rx_position'])             # a shared column stays
        self.assert_same(c['center_frequency'], ref['center_frequency'][[1, 0]])

        for snap in [1, [1], np.array([1], dtype=np.int32), np.uint64(1)]:
            chan, _ = channel.hdf5_read_channel(fn, snap=snap)
            self.assert_same(chan[0]['coeff'], [ref['coeff'][1]])

        chan, _ = channel.hdf5_read_channel(fn, snap=[1], stack=True)
        self.assertEqual(chan[0]['coeff'].shape, (2, 3, 2, 1))

    def test_read_channel_errors(self):
        fn = self.fn('read_err.hdf5')
        with self.assertRaises(RuntimeError) as ctx:  # thrown by the wrapper
            channel.hdf5_read_channel(fn)
        self.assertIn('HDF5 file does not exist or has no layout.', str(ctx.exception))

        channel.hdf5_create_file(fn, 3, 2, 1, 2)
        channel.hdf5_write_channel(fn, full_channel())
        for kwargs, msg in [({'ix': 3}, "Slot index 'ix' is out of bound."),
                            ({'iy': [0, 2]}, "Slot index 'iy' is out of bound."),
                            ({'iz': 1}, "Slot index 'iz' is out of bound."),
                            ({'iw': 2}, "Slot index 'iw' is out of bound."),
                            ({'snap': 0}, "'snap' is only allowed when a single slot is selected."),
                            ({'ix': 0, 'iy': 0, 'iw': 0, 'snap': 2}, 'Snapshot index out of bound.')]:
            with self.assertRaises(ValueError, msg=msg) as ctx:
                channel.hdf5_read_channel(fn, **kwargs)
            self.assertIn(msg, str(ctx.exception))

        with self.assertRaises(TypeError):
            channel.hdf5_read_channel()
        with self.assertRaises(TypeError):
            channel.hdf5_read_channel(fn, no_such_argument=1)

    # ------------------------------------------------------------------
    # hdf5_write_dset, hdf5_read_dset, hdf5_read_dset_names
    # ------------------------------------------------------------------
    def test_dset_functions(self):
        fn = self.fn('dset.hdf5')
        channel.hdf5_create_file(fn, 2, 2, 2, 2)

        # Default slot 0; positional and keyword arguments reach their dimension
        self.assertIsNone(channel.hdf5_write_dset(fn, name='zeta', data=1.0))
        channel.hdf5_write_dset(fn, 0, 0, 0, 0, 'alpha', 2.0)
        channel.hdf5_write_dset(fn, 1, 0, 1, 1, 'b', np.arange(3.0))
        channel.hdf5_write_dset(fn=fn, ix=0, iy=1, iz=0, iw=1, name='c', data='text')

        has_data = channel.hdf5_read_layout(fn)[1]
        expected = np.zeros((2, 2, 2, 2), dtype=np.uint32)
        expected[0, 0, 0, 0] = expected[1, 0, 1, 1] = expected[0, 1, 0, 1] = 1
        npt.assert_array_equal(has_data, expected)

        names = channel.hdf5_read_dset_names(fn)
        self.assertIsInstance(names, list)
        self.assertEqual(names, ['zeta', 'alpha'])  # creation order
        self.assertEqual(channel.hdf5_read_dset_names(fn, 0, 0, 0, 0), ['zeta', 'alpha'])
        self.assertEqual(channel.hdf5_read_dset_names(fn, 1, 0, 1, 1), ['b'])
        self.assertEqual(channel.hdf5_read_dset_names(fn=fn, ix=0, iy=1, iz=0, iw=1), ['c'])
        self.assertEqual(channel.hdf5_read_dset_names(fn, 1), [])     # empty slot
        self.assertEqual(channel.hdf5_read_dset_names(fn, 5), [])     # outside the layout

        self.assertEqual(channel.hdf5_read_dset(fn, name='zeta'), 1.0)
        self.assertEqual(channel.hdf5_read_dset(fn, 0, 0, 0, 0, 'alpha'), 2.0)
        self.assert_same(channel.hdf5_read_dset(fn, 1, 0, 1, 1, 'b'), np.arange(3.0))
        self.assertEqual(channel.hdf5_read_dset(fn=fn, ix=0, iy=1, iz=0, iw=1, name='c'), 'text')
        self.assertIsNone(channel.hdf5_read_dset(fn, name='missing'))  # missing dataset
        self.assertIsNone(channel.hdf5_read_dset(fn, 1, name='zeta'))  # empty slot
        self.assertIsNone(channel.hdf5_read_dset(fn, 5, name='zeta'))  # outside the layout

        # The dataset is added to a slot with channel data; both are kept
        channel.hdf5_write_channel(fn, minimal_channel('ch'), ix=1, iy=1)
        channel.hdf5_write_dset(fn, 1, 1, name='extra', data=np.int64(4))
        chan, par = channel.hdf5_read_channel(fn, 1, 1, 0, 0)
        self.assertEqual(chan[0]['name'], 'ch')
        self.assert_same(par, {'extra': 4})

    def test_dset_prefix(self):
        fn = self.fn('dset_prefix.hdf5')
        channel.hdf5_create_file(fn, 2)
        channel.hdf5_write_dset(fn, 0, 0, 0, 0, 'a', 1.0)
        channel.hdf5_write_dset(fn, 0, 0, 0, 0, 'b', np.arange(2.0), 'x_')        # positional
        channel.hdf5_write_dset(fn, name='c', data='text')
        channel.hdf5_write_dset(fn, name='d', data=np.int32(4), prefix='x_')      # keyword

        self.assertEqual(channel.hdf5_read_dset_names(fn), ['a', 'c'])
        self.assertEqual(channel.hdf5_read_dset_names(fn, 0, 0, 0, 0, 'x_'), ['b', 'd'])
        self.assertEqual(channel.hdf5_read_dset_names(fn, prefix=''), ['par_a', 'x_b', 'par_c', 'x_d'])
        self.assertEqual(channel.hdf5_read_dset_names(fn, 1, prefix='x_'), [])

        self.assert_same(channel.hdf5_read_dset(fn, name='b', prefix='x_'), np.arange(2.0))
        self.assertEqual(channel.hdf5_read_dset(fn, 0, 0, 0, 0, 'd', 'x_'), 4)
        self.assertEqual(channel.hdf5_read_dset(fn, name='x_d', prefix=''), 4)
        self.assertIsNone(channel.hdf5_read_dset(fn, name='b'))                    # looked up as 'par_b'
        self.assertEqual(channel.hdf5_read_dset(fn, name='a', prefix='par_'), 1.0)

        with self.assertRaises(ValueError) as ctx:
            channel.hdf5_write_dset(fn, name='e', data=1.0, prefix='x-')
        self.assertIn('Parameter name must only contain', str(ctx.exception))

        # hdf5_read_channel returns only the 'par_' datasets
        self.assertEqual(list(channel.hdf5_read_channel(fn, 0)[1]), ['a', 'c'])

    def test_dset_types(self):
        fn = self.fn('dset_types.hdf5')
        channel.hdf5_create_file(fn, 1)
        for name, (v, expected, typ) in typed_values().items():
            channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
            self.assert_value(channel.hdf5_read_dset(fn, name=name), expected, typ, name)

        # Strided arrays are written with their values
        for name, v in strided_inputs().items():
            channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
            r = channel.hdf5_read_dset(fn, name=name)
            self.assertEqual(r.shape, v.shape, name)
            self.assert_value(r, v, v.dtype, name)

        # Shapes: a 1-element vector is an array of shape (1,), not a scalar; 2-D and 3-D arrays keep their shape,
        # also (n, 1) and (1, 1)
        for name, v in {'v1': np.array([5.0]), 'v3': np.ones(3), 'm11': np.full((1, 1), 2.0), 'm31': np.ones((3, 1)),
                        'm13': np.ones((1, 3)), 'm32': np.arange(6.0).reshape(3, 2), 'c111': np.ones((1, 1, 1)),
                        'c311': np.ones((3, 1, 1)), 'c113': np.ones((1, 1, 3)),
                        'u1': np.array([7], dtype=np.uint32)}.items():
            channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)
            self.assert_same(channel.hdf5_read_dset(fn, name=name), v, name)

        # Unsupported types
        for name, v in {'int16': np.arange(3, dtype=np.int16), 'np_bool': np.array([True, False]),
                        'complex': np.ones(2, complex), 'f16': np.ones(2, np.float16),
                        'four_d': np.ones((1, 2, 1, 2)), 'list': [1.0, 2.0]}.items():
            with self.assertRaises(ValueError, msg=name):
                channel.hdf5_write_dset(fn, 0, 0, 0, 0, name, v)

        # The caller's arrays are not modified; read-only arrays are accepted
        x = np.asfortranarray(np.arange(12.0).reshape(3, 4))
        x0 = x.copy()
        x.flags.writeable = False
        channel.hdf5_write_dset(fn, 0, 0, 0, 0, 'x', x)
        npt.assert_array_equal(x, x0)
        npt.assert_array_equal(channel.hdf5_read_dset(fn, name='x'), x0)

    def test_dset_errors(self):
        fn = self.fn('dset_err.hdf5')
        for f, kwargs in [(channel.hdf5_write_dset, {'name': 'x', 'data': 1.0}),
                          (channel.hdf5_read_dset, {'name': 'x'}), (channel.hdf5_read_dset_names, {})]:
            with self.assertRaises(ValueError) as ctx:
                f(fn, **kwargs)
            self.assertIn('File does not exist.', str(ctx.exception))

        channel.hdf5_create_file(fn, 2, 2)
        channel.hdf5_write_dset(fn, name='x', data=1.0)
        for kwargs, msg in [({'name': 'y'}, 'Data not provided.'),                    # thrown by the wrapper
                            ({'name': '', 'data': 1.0}, 'Dataset name not provided.'),
                            ({'ix': 3, 'name': 'y', 'data': 1.0}, 'Location exceeds storage space in HDF file.'),
                            ({'ix': 2, 'name': 'y', 'data': 1.0}, 'Index out of bound.'),  # thrown by C++
                            ({'iy': 2, 'name': 'y', 'data': 1.0}, 'Index out of bound.'),
                            ({'name': 'x', 'data': 2.0}, "Dataset 'par_x' already exists."),
                            ({'name': 'a b', 'data': 1.0}, 'Parameter name must only contain letters'),
                            ({'name': 'y', 'data': np.ones(2, complex)}, "Input 'y' has an unsupported type.")]:
            with self.assertRaises(ValueError, msg=msg) as ctx:
                channel.hdf5_write_dset(fn, **kwargs)
            self.assertIn(msg, str(ctx.exception))
        self.assertEqual(channel.hdf5_read_dset_names(fn), ['x'])  # nothing else was written

        # Rejected by pybind11: 'name' has no default
        with self.assertRaises(TypeError):
            channel.hdf5_write_dset(fn, data=1.0)
        with self.assertRaises(TypeError):
            channel.hdf5_read_dset(fn)
        with self.assertRaises(TypeError):
            channel.hdf5_read_dset_names(fn, no_such_argument=1)


if __name__ == '__main__':
    unittest.main()
