# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

import sys
import os
import pathlib
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


def _make_chan(n_rx=3, n_tx=2, n_path=5, n_snap=3, seed=1):
    """Channel dict in the stacked layout (one interaction per path, no padding)."""
    rng = np.random.default_rng(seed)
    return {
        'rx_pos': rng.random((3, 1)),
        'tx_pos': rng.random((3, 1)),
        'coeff': rng.random((n_rx, n_tx, n_path, n_snap)) + 1j * rng.random((n_rx, n_tx, n_path, n_snap)),
        'delay': np.zeros((n_rx, n_tx, n_path, n_snap)),
        # one interaction point per path -> sum(n_interact) == n_path per snapshot
        'n_interact': np.ones((n_path, n_snap), dtype=np.uint32),
        'interact_coord': rng.random((3, n_path, n_snap)),
        'center_freq': np.full(n_snap, 3.0e9),
    }


def _to_lists(chan):
    """The same channel dict in the list layout (one array per snapshot)."""
    n_snap = chan['n_interact'].shape[1]
    out = dict(chan)
    for key in ('coeff', 'delay', 'interact_coord'):
        out[key] = [np.ascontiguousarray(chan[key][..., j]) for j in range(n_snap)]
    out['n_interact'] = [np.ascontiguousarray(chan['n_interact'][:, j]) for j in range(n_snap)]
    return out


class test_channel_export_obj_file(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _path(self, name):
        return os.path.join(self.tmp, name)

    def _export(self, name, chan, **kwargs):
        """Run the exporter and return the path to the written OBJ file."""
        fn = self._path(name)
        quadriga_lib.channel.export_obj_file(fn, chan, **kwargs)
        return fn

    # value tests
    def test_basic(self):
        fn = self._export('basic.obj', _make_chan())
        lines = _significant_lines(fn)
        self.assertTrue(any(ln.startswith('v ') for ln in lines))
        # the companion material file is written next to the OBJ file
        self.assertTrue(os.path.isfile(self._path('basic.mtl')))

    def test_layouts_and_split_coeff_match(self):
        # list layout, stacked layout, split real coefficients and a mix give the same file
        chan = _make_chan()
        ref = _significant_lines(self._export('ref.obj', chan))
        lst = _to_lists(chan)
        self.assertEqual(_significant_lines(self._export('list.obj', lst)), ref)
        split = {k: v for k, v in chan.items() if k != 'coeff'}
        split['coeff_re'], split['coeff_im'] = chan['coeff'].real, chan['coeff'].imag
        self.assertEqual(_significant_lines(self._export('split.obj', split)), ref)
        mix = dict(chan, n_interact=lst['n_interact'], delay=lst['delay'])
        self.assertEqual(_significant_lines(self._export('mix.obj', mix)), ref)

    def test_padded_stacked_trim(self):
        # Snapshots with different interaction counts: the stacked interact_coord is padded to
        # max(sum(n_interact)); the padding is dropped per snapshot. The list carries exactly the
        # valid columns, so both must match.
        rng = np.random.default_rng(7)
        n_interact = np.array([[1, 1], [2, 1], [1, 1], [1, 1]], dtype=np.uint32)  # column sums 5 and 4
        sums = n_interact.sum(axis=0)
        ic = np.zeros((3, int(sums.max()), 2))
        ic_list = []
        for j in range(2):
            ic[:, :sums[j], j] = rng.random((3, sums[j]))
            ic[:, sums[j]:, j] = 999.0  # poison the padding: a missing trim would change the geometry
            ic_list.append(np.ascontiguousarray(ic[:, :sums[j], j]))
        chan = _make_chan(n_rx=2, n_tx=2, n_path=4, n_snap=2, seed=3)
        chan['n_interact'], chan['interact_coord'] = n_interact, ic
        lst = dict(_to_lists(chan), interact_coord=ic_list)
        self.assertEqual(_significant_lines(self._export('pad_nd.obj', chan)),
                         _significant_lines(self._export('pad_li.obj', lst)))

        # A channel read with hdf5_read_channel (both layouts) is exported and written back directly
        fn_h5 = self._path('pad.hdf5')
        quadriga_lib.channel.hdf5_write_channel(fn_h5, lst)
        c_li = quadriga_lib.channel.hdf5_read_channel(fn_h5)[0][0]
        c_st = quadriga_lib.channel.hdf5_read_channel(fn_h5, stack=True)[0][0]
        self.assertEqual(c_st['interact_coord'].shape, (3, 5, 2))
        self.assertEqual(_significant_lines(self._export('h5_li.obj', c_li)),
                         _significant_lines(self._export('h5_st.obj', c_st)))
        quadriga_lib.channel.hdf5_write_channel(self._path('pad2.hdf5'), c_st)

    def test_max_paths_limits(self):
        chan = _make_chan()
        fn_all = self._export('all.obj', chan, max_paths=0)
        fn_one = self._export('one.obj', chan, max_paths=1)
        self.assertLess(len(_significant_lines(fn_one)), len(_significant_lines(fn_all)))

    def test_i_snap_selection(self):
        chan = _make_chan()
        # i_snap is 0-based; one snapshot gives only that snapshot
        with open(self._export('isnap_one.obj', chan, i_snap=np.array([2], dtype=np.uint64))) as f:
            txt = f.read()
        self.assertEqual(txt.count('# Snapshot'), 1)
        self.assertEqual(txt.count('# Snapshot 2'), 1)
        # duplicate indices are allowed
        with open(self._export('isnap_dup.obj', chan, i_snap=[0, 0])) as f:
            self.assertEqual(f.read().count('# Snapshot 0'), 2)

    def test_none_defaults_and_path_fn(self):
        chan = _make_chan(n_snap=1)
        ref = _significant_lines(self._export('omit.obj', chan))
        self.assertEqual(_significant_lines(self._export('none.obj', chan, i_snap=None)), ref)
        fn = pathlib.Path(self._path('path.obj'))
        quadriga_lib.channel.export_obj_file(fn, chan=chan)
        self.assertEqual(_significant_lines(str(fn)), ref)

    def test_scalar_params(self):
        chan = _make_chan()
        self.assertTrue(os.path.isfile(self._export('gain.obj', chan, gain_max_dB=10.0, gain_min_dB=6.0)))
        self.assertTrue(os.path.isfile(self._export('cmap.obj', chan, colormap='parula')))
        # zero radius -> paths without a volume
        self.assertTrue(os.path.isfile(self._export('flat.obj', chan, radius_max=0.0, radius_min=0.0)))

    # error tests
    def test_error_chan_type(self):
        chan = _make_chan()
        with self.assertRaisesRegex(ValueError, "Input 'chan' must be a dict."):
            self._export('l.obj', [chan])
        with self.assertRaises(TypeError):
            self._export('n.obj', None)
        with self.assertRaises(TypeError):  # options are keyword-only
            quadriga_lib.channel.export_obj_file(self._path('p.obj'), chan, 1)

    def test_error_invalid_channel(self):
        chan = _make_chan()
        cases = [
            ({k: v for k, v in chan.items() if k != 'delay'}, 'Delays are missing or incomplete.'),
            (dict(_to_lists(chan), interact_coord=[np.zeros((3, 6))] * 3), "must match the sum of 'n_interact'."),
            (dict(chan, coeff_re=chan['coeff'].real), "provide either 'coeff'"),
        ]
        for c, msg in cases:
            with self.assertRaisesRegex(ValueError, msg):
                self._export('bad.obj', c)

    def test_error_options(self):
        chan = _make_chan()
        with self.assertRaisesRegex(ValueError, 'must end with .obj'):
            quadriga_lib.channel.export_obj_file(self._path('bad.txt'), chan)
        with self.assertRaisesRegex(ValueError, 'Colormap is not supported.'):
            self._export('cm.obj', chan, colormap='not_a_colormap')
        with self.assertRaises(ValueError):
            self._export('ne.obj', chan, n_edges=1)
        with self.assertRaisesRegex(ValueError, 'Radius cannot be negative.'):
            self._export('nr.obj', chan, radius_max=-1.0)
        with self.assertRaisesRegex(ValueError, "'i_snap' cannot exceed"):
            self._export('oor.obj', chan, i_snap=np.array([3], dtype=np.uint64))


if __name__ == '__main__':
    unittest.main()
