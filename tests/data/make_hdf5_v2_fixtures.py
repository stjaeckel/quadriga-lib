#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Writes the channel files of version 2 that test tools/hdf5_convert_v2.py (tests/python_tests/test_hdf5_convert_v2.py)
# and the oracle: what quadriga-lib 0.12.4 returns for each of their slots. The files are committed; run this script
# only to rebuild them.
#
#   hdf5_v2_written.hdf5   written by 0.12.4 functions only
#   hdf5_v2_edited.hdf5    0.12.4 output changed with h5py into layouts of older writers and of other HDF5 tools,
#                          all read by 0.12.4 and convertible without loss
#   hdf5_v2_lossy.hdf5     items that version 3 cannot store (the converter skips them with a warning)
#   hdf5_v2_expected.npz   per file: '<file>/layout', '<file>/has_data' (hdf5_read_layout of 0.12.4); per occupied
#                          slot s (linear index): '<file>/<s>/chan/<key>' (array entries of hdf5_read_channel),
#                          '<file>/<s>/chan/<key>/<i>' (list entries), '<file>/<s>/par/<name>', the key order in
#                          '<file>/<s>/chan_keys' (absent if the reader returned no channel) and '<file>/<s>/par_keys',
#                          or '<file>/<s>/error' with the exception text if the 0.12.4 Python reader raised for the
#                          slot. Every value is stored as np.asarray(value) without pickle.
#
# Recipe: build the Python module of quadriga-lib 0.12.4 from a checkout of commit 26da9f93 (release 0.12.4):
#   cmake -S <src> -B <build> -DENABLE_PYTHON=ON -DPython3_EXECUTABLE=<python> -DCMAKE_CXX_COMPILER=g++-12 \
#         -DENABLE_MATLAB=OFF -DENABLE_OCTAVE=OFF -DENABLE_TESTS=OFF
#   cmake --build <build> --target quadriga_py
# then, in a Python environment with numpy and h5py:
#   PYTHONPATH=<folder with the built module> python tests/data/make_hdf5_v2_fixtures.py
# The oracle is reproducible; the HDF5 files differ in their timestamps on each run.

import os
import warnings
import numpy as np
import h5py
from quadriga_lib import channel as qc

out_dir = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(20261029)
randn = lambda *shape: rng.standard_normal(shape)
crandn = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape)


def set_name(group, raw):
    """Write the 'Name' attribute as 0.12.4 does (fixed-length string of 256 bytes), with the given raw bytes."""
    del group.attrs['Name']
    tid = h5py.h5t.C_S1.copy()
    tid.set_size(256)
    aid = h5py.h5a.create(group.id, b'Name', tid, h5py.h5s.create(h5py.h5s.SCALAR))
    aid.write(np.frombuffer(raw.ljust(256, b'\0'), dtype='S256').reshape(()), mtype=tid)  # raw: no NUL is forced


def multi_snapshot_channel(n_rx, n_tx, paths, delay_full=(0,)):
    """Every channel key; 'delay' is (n_rx, n_tx, n) for the snapshots in delay_full and (1, 1, n) otherwise."""
    c = {k: [] for k in ('coeff', 'delay', 'path_gain', 'path_length', 'path_polarization', 'path_angles', 'fbs_pos',
                         'lbs_pos', 'no_interact', 'interact_coord')}
    for s, n in enumerate(paths):
        c['coeff'].append(crandn(n_rx, n_tx, n))
        c['delay'].append(randn(n_rx, n_tx, n) if s in delay_full else randn(1, 1, n))
        c['path_gain'].append(randn(n))
        c['path_length'].append(randn(n))
        c['path_polarization'].append(crandn(4, n))
        c['path_angles'].append(randn(n, 4))
        c['fbs_pos'].append(randn(3, n))
        c['lbs_pos'].append(randn(3, n))
        c['no_interact'].append(np.arange(n) % 3)
        c['interact_coord'].append(randn(3, int(c['no_interact'][-1].sum())))
    return c


def positions(n_tx=1):
    return {'tx_position': randn(3, n_tx), 'rx_position': randn(3, 1)}


def written(fn):
    qc.hdf5_create_file(fn, 4, 2, 2, 1)
    c = multi_snapshot_channel(2, 2, (3, 0, 2))
    special = [-0.0, np.nan, np.inf, -np.inf]
    c['coeff'][0][0, 0, :] = [complex(-0.0, np.nan), complex(np.inf, -0.0), complex(np.nan, -np.inf)]
    c['path_polarization'][2][:, 0] = [complex(a, b) for a, b in zip(special, special[::-1])]
    c.update(name='München €', initial_position=2, tx_position=randn(3, 3), rx_position=randn(3, 1),
             tx_orientation=randn(3, 1), rx_orientation=randn(3, 3), center_frequency=np.array([1e9, 2.4e9, 5.8e9]))
    par = {'text': 'carrier', 'empty': '', 'inner_nul': 'a\0b', 'f32': np.float32(1.5), 'f64': 0.1,
           'u64': np.uint64(2**60 + 1), 'i64': np.int64(-2**40), 'u32': np.uint32(4000000000), 'i32': np.int32(-7),
           'vec': randn(4), 'vec1': np.array([3.0]), 'mat': randn(2, 3), 'col': np.arange(3, dtype=np.int64)[:, None],
           'mat03': np.zeros((0, 3)), 'cube': np.arange(24, dtype=np.int32).reshape(2, 3, 4)}
    qc.hdf5_write_channel(fn, c, par, 0, 0, 0, 0)
    los = {'name': 'n' * 255, 'tx_position': randn(3, 1).astype(np.float32),
           'rx_position': randn(3, 1).astype(np.float32), 'coeff': [crandn(1, 1, 2).astype(np.complex64)],
           'delay': [randn(1, 1, 2).astype(np.float32)], 'no_interact': [np.zeros(2, np.uint32)],
           'interact_coord': [np.zeros((3, 0), np.float32)]}
    qc.hdf5_write_channel(fn, los, None, 1, 0, 0, 0)
    qc.hdf5_write_dset(fn, 1, 0, 0, 0, 'added', np.arange(4.0))  # added to a filled slot
    qc.hdf5_write_channel(fn, dict(positions(), coeff=[np.zeros((1, 1, 0), complex)], delay=[np.zeros((1, 1, 0))]),
                          None, 2, 0, 0, 0)
    qc.hdf5_write_channel(fn, positions(2), None, 3, 0, 0, 0)
    qc.hdf5_write_channel(fn, {'name': 'start', 'initial_position': 3}, None, 0, 1, 0, 0)
    qc.hdf5_write_channel(fn, {}, {'only': np.arange(3, dtype=np.int32)}, 1, 1, 0, 0)
    qc.hdf5_write_dset(fn, 2, 1, 0, 0, 'first', 'one')  # creates the slot
    qc.hdf5_write_dset(fn, 2, 1, 0, 0, 'second', np.float32(3))
    qc.hdf5_write_channel(fn, los, None, 3, 1, 0, 0)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        qc.hdf5_write_channel(fn, dict(multi_snapshot_channel(1, 1, (1, 2, 0)), **positions()), None, 3, 1, 0, 0)
    qc.hdf5_reshape_layout(fn, 2, 4, 2, 1)


def edited(fn):
    qc.hdf5_create_file(fn, 12, 1, 1, 1)
    c = multi_snapshot_channel(2, 1, (2, 0, 3, 0), delay_full=())
    qc.hdf5_write_channel(fn, dict(c, **positions()), None, 0)
    qc.hdf5_write_channel(fn, dict(positions(3), path_gain=[randn(2), randn(1), randn(0)],
                                   delay=[randn(1, 1, 2), randn(1, 1, 1), randn(1, 1, 0)]), None, 1)
    qc.hdf5_write_channel(fn, dict(multi_snapshot_channel(2, 3, (2, 3), delay_full=(0, 1)), **positions(2)), None, 2)
    c = multi_snapshot_channel(1, 1, (2, 3))
    c['no_interact'][0], c['interact_coord'][0] = np.zeros(2, np.uint32), np.zeros((3, 0))  # LOS only
    qc.hdf5_write_channel(fn, dict(c, **positions(2)), None, 3)
    qc.hdf5_write_channel(fn, dict(positions(), coeff=[np.zeros((1, 1, 0), complex)], delay=[np.zeros((1, 1, 0))]),
                          None, 4)
    qc.hdf5_write_channel(fn, dict(positions(), name='abc'), {'first': 1.0}, 5)
    qc.hdf5_write_channel(fn, {}, {'ip': np.arange(2.0)}, 6)
    qc.hdf5_write_channel(fn, positions(), {'a': 1.0}, 7)
    qc.hdf5_write_dset(fn, 8, 0, 0, 0, 'tmp', 1.0)
    with h5py.File(fn, 'r+') as f:
        group = lambda s: f['channel_%d' % f['Order'][s]]
        del group(0)['Snap_1'], group(0)['Snap_3']        # older writers: no group for an empty snapshot
        del group(1)['Snap_2']                            # same, without coefficients
        for name, part in (('Snap_0', np.s_[:, :, :1]), ('Snap_1', np.s_[:, :1, :])):  # delays of 2023/24 writers:
            d = group(2)[name]['delay'][()][part]          # {np, 3, 1} and {np, 1, 2}, h5py order {np, n_tx, n_rx}
            del group(2)[name]['delay']
            group(2)[name]['delay'] = d
        del group(3)['Snap_0']['interact_coord']          # LOS-only snapshot: older writers stored no interact_coord
        for k in list(group(4)):                          # single snapshot, 0 paths: older writers stored no datasets
            del group(4)[k]
        set_name(group(5), b'abc\0garbage after the NUL')
        data = group(5)['Data']                           # rebuilt without creation-order tracking
        first = data['par_first'][()]
        del group(5)['Data']
        data = group(5).create_group('Data', track_order=False)
        data['par_zz'] = np.arange(3, dtype='>i4')        # big-endian
        data['par_first'] = first
        data.create_dataset('par_vs', data='vlen ä', dtype=h5py.string_dtype())
        data.create_dataset('par_gz', data=np.arange(12.0).reshape(3, 4), chunks=(1, 4), compression='gzip')
        group(6).attrs['Initial_position'] = np.int32(5)  # older writers: attributes with unstructured data
        data = group(7)['Data']                           # custom prefixes (MEX hdf5_write_dset with prefix)
        data['x_b'] = np.arange(2, dtype=np.uint32)
        data['par_c'] = 'c'
        data['y_d'] = np.float32(4)
        data = group(8)['Data']
        data['x_e'] = np.arange(3.0)
        del data['par_tmp']


def lossy(fn):
    qc.hdf5_create_file(fn, 6, 1, 1, 1)
    qc.hdf5_write_channel(fn, dict(multi_snapshot_channel(1, 2, (2,)), **positions()), {'good': 1.0}, 0)
    for s in range(1, 5):
        qc.hdf5_write_channel(fn, {}, {'good': np.arange(s, dtype=np.int64)}, s)
    qc.hdf5_write_channel(fn, {'initial_position': 1}, None, 5)
    with h5py.File(fn, 'r+') as f:
        group = lambda s: f['channel_%d' % f['Order'][s]]
        set_name(group(5), ('a' * 254 + 'é').encode())    # 256 bytes, no NUL; the last character has 2 bytes
        data = group(0)['Data']
        data['par_bad-name'] = 1.0
        data['par_i16'] = np.arange(3, dtype=np.int16)
        data['par_rank4'] = np.zeros((1, 2, 1, 2))
        data.create_dataset('par_enum', data=np.array([0, 1], 'i4'), dtype=h5py.enum_dtype({'A': 0, 'B': 1}, 'i4'))
        data['par_last'] = np.int32(9)
        del group(1)['Data']                              # occupied, but nothing left
        del group(2)['Data']
        set_name(group(2), b'name only')
        set_name(group(3), b'name with data')
        group(4)['Data']['par_latin1'] = np.bytes_(b'caf\xe9')  # not UTF-8
        group(4)['Data']['par_after'] = 2.0


def oracle(fn, key, out):
    dims, has_data = qc.hdf5_read_layout(fn)
    out[key + '/layout'], out[key + '/has_data'] = np.asarray(dims), np.asarray(has_data)
    nx, ny, nz = (int(n) for n in dims[:3])
    for s in np.flatnonzero(np.asarray(has_data).ravel(order='F')):
        slot = '%s/%d/' % (key, s)
        try:
            chan, par = qc.hdf5_read_channel(fn, s % nx, s // nx % ny, s // (nx * ny) % nz, s // (nx * ny * nz))
        except Exception as e:
            out[slot + 'error'] = np.asarray('%s: %s' % (type(e).__name__, e))
            continue
        if chan:  # no channel for a slot with neither channel data nor 'par_' datasets
            out[slot + 'chan_keys'] = np.array(list(chan[0]), dtype=str)
        out[slot + 'par_keys'] = np.array(list(par), dtype=str)
        for k, v in (chan[0] if chan else {}).items():
            for i, x in enumerate(v) if isinstance(v, list) else [(None, v)]:
                out[slot + 'chan/' + k + ('' if i is None else '/%d' % i)] = np.asarray(x)
        for k, v in par.items():
            out[slot + 'par/' + k] = np.asarray(v)


expected = {}
for name, make in (('hdf5_v2_written.hdf5', written), ('hdf5_v2_edited.hdf5', edited),
                   ('hdf5_v2_lossy.hdf5', lossy)):
    fn = os.path.join(out_dir, name)
    if os.path.exists(fn):
        os.remove(fn)
    make(fn)
    with h5py.File(fn, 'r') as f:
        if int(f['Version'][()]) != 2:
            raise SystemExit("'%s' is not of version 2: run this script with the quadriga_lib module of 0.12.4" % fn)
    oracle(fn, name, expected)
np.savez_compressed(os.path.join(out_dir, 'hdf5_v2_expected.npz'), **expected)
for name in sorted(os.listdir(out_dir)):
    if name.startswith('hdf5_v2_'):
        print('%-24s %7d bytes' % (name, os.path.getsize(os.path.join(out_dir, name))))
