# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import shutil
import tempfile
import unittest
import numpy as np
import numpy.testing as npt

current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib


def _significant_lines(path):
    """Return the geometry-relevant lines of an OBJ/MTL file.

    Comments, blank lines and the 'mtllib' reference (which embeds the file
    name) are dropped, so that two files written under different names can be
    compared for byte-identical geometry.
    """
    out = []
    with open(path, 'r') as fid:
        for line in fid:
            stripped = line.strip()
            if not stripped or stripped.startswith('#') or stripped.startswith('mtllib'):
                continue
            out.append(line.rstrip('\n'))
    return out


def _make_inputs(n_rx=3, n_tx=2, n_path=5, n_snap=3, seed=1):
    """Build a consistent set of nD inputs (one interaction per path, no padding)."""
    rng = np.random.default_rng(seed)
    return {
        'rx_pos': rng.random((3, 1)),
        'tx_pos': rng.random((3, 1)),
        'coeff_re': rng.random((n_rx, n_tx, n_path, n_snap)),
        'coeff_im': rng.random((n_rx, n_tx, n_path, n_snap)),
        # one interaction point per path -> sum(no_interact) == n_path per snapshot
        'no_interact': np.ones((n_path, n_snap), dtype=np.uint32),
        'interact_coord': rng.random((3, n_path, n_snap)),
        'center_freq': np.full(n_snap, 3.0e9),
    }


def _to_lists(data):
    """Convert the per-snapshot nD fields of _make_inputs() into ragged lists."""
    n_snap = data['no_interact'].shape[1]
    coeff = data['coeff_re'] + 1j * data['coeff_im']
    return {
        'no_interact': [np.ascontiguousarray(data['no_interact'][:, j]) for j in range(n_snap)],
        'interact_coord': [np.ascontiguousarray(data['interact_coord'][:, :, j]) for j in range(n_snap)],
        'coeff': [np.ascontiguousarray(coeff[:, :, :, j]) for j in range(n_snap)],
    }


class test_channel_export_obj_file(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _export(self, name, **kwargs):
        """Run the exporter and return the path to the written OBJ file."""
        fn = self._path(name)
        quadriga_lib.channel.channel_export_obj_file(fn, **kwargs)
        return fn

    # value tests
    def test_basic_ndarray(self):
        d = _make_inputs()
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        fn = self._export('basic.obj',
                          rx_pos=d['rx_pos'], tx_pos=d['tx_pos'],
                          no_interact=d['no_interact'], interact_coord=d['interact_coord'],
                          center_freq=d['center_freq'], coeff=coeff)

        self.assertTrue(os.path.isfile(fn))
        lines = _significant_lines(fn)
        self.assertGreater(len(lines), 0)
        # geometry must contain vertex definitions
        self.assertTrue(any(ln.startswith('v ') for ln in lines))
        # the companion material file is written next to the OBJ file
        self.assertTrue(os.path.isfile(self._path('basic.mtl')))

    def test_split_coeff_matches_complex(self):
        # 'coeff' (complex) and 'coeff_re'/'coeff_im' (split real) must be equivalent
        d = _make_inputs()
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        common = dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'],
                      no_interact=d['no_interact'], interact_coord=d['interact_coord'],
                      center_freq=d['center_freq'])

        fn_c = self._export('cplx.obj', coeff=coeff, **common)
        fn_s = self._export('split.obj', coeff_re=d['coeff_re'], coeff_im=d['coeff_im'], **common)

        self.assertEqual(_significant_lines(fn_c), _significant_lines(fn_s))
        self.assertEqual(_significant_lines(self._path('cplx.mtl')),
                         _significant_lines(self._path('split.mtl')))

    def test_list_matches_ndarray(self):
        # ragged list inputs must produce the same file as the nD inputs
        d = _make_inputs()
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        lst = _to_lists(d)
        common = dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'], center_freq=d['center_freq'])

        fn_nd = self._export('nd.obj', no_interact=d['no_interact'],
                             interact_coord=d['interact_coord'], coeff=coeff, **common)
        fn_li = self._export('list.obj', no_interact=lst['no_interact'],
                             interact_coord=lst['interact_coord'], coeff=lst['coeff'], **common)

        self.assertEqual(_significant_lines(fn_nd), _significant_lines(fn_li))

    def test_mixed_formats(self):
        # list / ndarray / split-real can be mixed freely across the three fields
        d = _make_inputs()
        lst = _to_lists(d)
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        common = dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'], center_freq=d['center_freq'])

        fn_ref = self._export('ref.obj', no_interact=d['no_interact'],
                              interact_coord=d['interact_coord'], coeff=coeff, **common)
        # no_interact as list, interact_coord as ndarray, coefficients as split real
        fn_mix = self._export('mix.obj', no_interact=lst['no_interact'],
                              interact_coord=d['interact_coord'],
                              coeff_re=d['coeff_re'], coeff_im=d['coeff_im'], **common)

        self.assertEqual(_significant_lines(fn_ref), _significant_lines(fn_mix))

    def test_padded_ndarray_trim(self):
        # Snapshots with different interaction counts: the nD interact_coord is padded
        # to max(sum(no_interact)); the wrapper must trim it back per snapshot. The
        # ragged list carries exactly the valid columns, so both must match.
        rng = np.random.default_rng(7)
        n_rx, n_tx, n_path, n_snap = 2, 2, 4, 2
        no_interact = np.array([[1, 1],
                                [2, 1],
                                [1, 1],
                                [1, 1]], dtype=np.uint32)   # column sums: snap0=5, snap1=4
        sums = no_interact.sum(axis=0)
        max_int = int(sums.max())

        interact_coord_nd = np.zeros((3, max_int, n_snap))
        ic_list = []
        for j in range(n_snap):
            valid = rng.random((3, int(sums[j])))
            interact_coord_nd[:, :int(sums[j]), j] = valid
            # poison the padding columns: a missing trim would change the geometry
            interact_coord_nd[:, int(sums[j]):, j] = 999.0
            ic_list.append(np.ascontiguousarray(valid))

        coeff_re = rng.random((n_rx, n_tx, n_path, n_snap))
        coeff_im = rng.random((n_rx, n_tx, n_path, n_snap))
        coeff = coeff_re + 1j * coeff_im
        ni_list = [np.ascontiguousarray(no_interact[:, j]) for j in range(n_snap)]
        cf_list = [np.ascontiguousarray(coeff[:, :, :, j]) for j in range(n_snap)]

        common = dict(rx_pos=rng.random((3, 1)), tx_pos=rng.random((3, 1)),
                      center_freq=np.full(n_snap, 3.0e9))

        fn_nd = self._export('pad_nd.obj', no_interact=no_interact,
                             interact_coord=interact_coord_nd, coeff=coeff, **common)
        fn_li = self._export('pad_li.obj', no_interact=ni_list,
                             interact_coord=ic_list, coeff=cf_list, **common)

        self.assertEqual(_significant_lines(fn_nd), _significant_lines(fn_li))

    def test_max_no_paths_limits(self):
        d = _make_inputs()
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        common = dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'],
                      no_interact=d['no_interact'], interact_coord=d['interact_coord'],
                      center_freq=d['center_freq'], coeff=coeff)

        fn_all = self._export('all.obj', max_no_paths=0, **common)
        fn_one = self._export('one.obj', max_no_paths=1, **common)

        self.assertLess(len(_significant_lines(fn_one)), len(_significant_lines(fn_all)))

    def test_i_snap_selection(self):
        d = _make_inputs()
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        common = dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'],
                      no_interact=d['no_interact'], interact_coord=d['interact_coord'],
                      center_freq=d['center_freq'], coeff=coeff)

        # i_snap is 0-based; a subset of snapshots yields fewer paths than the full set
        fn_full = self._export('isnap_full.obj', **common)
        fn_one = self._export('isnap_one.obj', i_snap=np.array([0], dtype=np.uint64), **common)
        self.assertLess(len(_significant_lines(fn_one)), len(_significant_lines(fn_full)))

        # duplicate indices are allowed
        fn = self._export('isnap_dup.obj', i_snap=np.array([0, 2], dtype=np.uint64), **common)
        self.assertGreater(len(_significant_lines(fn)), 0)

    def test_scalar_params(self):
        # mirror the happy-path scalar variations from the MATLAB test
        d = _make_inputs()
        coeff = d['coeff_re'] + 1j * d['coeff_im']
        common = dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'],
                      no_interact=d['no_interact'], interact_coord=d['interact_coord'],
                      center_freq=d['center_freq'], coeff=coeff)

        self.assertTrue(os.path.isfile(self._export('gain.obj', gain_max=10.0, gain_min=6.0, **common)))
        self.assertTrue(os.path.isfile(self._export('cmap.obj', colormap='parula', **common)))
        # zero radius -> paths without a volume
        self.assertTrue(os.path.isfile(self._export('flat.obj', radius_max=0.0, radius_min=0.0, **common)))

    # error tests
    def _valid_kwargs(self):
        d = _make_inputs()
        return dict(rx_pos=d['rx_pos'], tx_pos=d['tx_pos'],
                    no_interact=d['no_interact'], interact_coord=d['interact_coord'],
                    center_freq=d['center_freq'],
                    coeff=d['coeff_re'] + 1j * d['coeff_im']), d

    def test_error_bad_filename(self):
        kw, _ = self._valid_kwargs()
        with self.assertRaises(Exception):
            quadriga_lib.channel.channel_export_obj_file(self._path('bad.txt'), **kw)

    def test_error_bad_colormap(self):
        kw, _ = self._valid_kwargs()
        with self.assertRaises(Exception):
            self._export('cm.obj', colormap='not_a_colormap', **kw)

    def test_error_n_edges_too_small(self):
        kw, _ = self._valid_kwargs()
        with self.assertRaises(Exception):
            self._export('ne.obj', n_edges=1, **kw)

    def test_error_negative_radius(self):
        kw, _ = self._valid_kwargs()
        with self.assertRaises(Exception):
            self._export('nr.obj', radius_max=-1.0, **kw)

    def test_error_bad_rx_pos(self):
        kw, _ = self._valid_kwargs()
        kw['rx_pos'] = np.random.default_rng(0).random((2, 1))   # must have 3 rows
        with self.assertRaises(Exception):
            self._export('rp.obj', **kw)

    def test_error_interact_coord_snap_mismatch(self):
        kw, d = self._valid_kwargs()
        # interact_coord has fewer snapshots than the coefficients
        kw['interact_coord'] = d['interact_coord'][:, :, :2]
        with self.assertRaisesRegex(Exception, 'Number of snapshots in interact_coord must match coefficients.'):
            self._export('mm.obj', **kw)

    def test_error_i_snap_out_of_range(self):
        kw, _ = self._valid_kwargs()
        # valid 0-based indices are 0..2; index 3 is out of range
        with self.assertRaises(Exception):
            self._export('oor.obj', i_snap=np.array([3], dtype=np.uint64), **kw)

    def test_error_coeff_and_split(self):
        kw, d = self._valid_kwargs()
        with self.assertRaisesRegex(Exception, 'both'):
            self._export('cs.obj', coeff_re=d['coeff_re'], coeff_im=d['coeff_im'], **kw)

    def test_error_split_incomplete(self):
        kw, d = self._valid_kwargs()
        del kw['coeff']
        with self.assertRaisesRegex(Exception, 'must both be provided'):
            self._export('si.obj', coeff_re=d['coeff_re'], **kw)

    def test_error_no_coeff(self):
        kw, _ = self._valid_kwargs()
        del kw['coeff']
        with self.assertRaisesRegex(Exception, 'Must provide either'):
            self._export('nc.obj', **kw)


if __name__ == '__main__':
    unittest.main()