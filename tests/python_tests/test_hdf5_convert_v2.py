# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

# Test of tools/hdf5_convert_v2.py, the converter of channel files of version 2 (quadriga-lib 0.12.4 and earlier) to
# version 3. The script runs as a subprocess, so the command line, exit codes and messages are tested. The inputs are
# the version-2 files in tests/data (written by tests/data/make_hdf5_v2_fixtures.py with quadriga-lib 0.12.4); the
# oracle hdf5_v2_expected.npz holds what 0.12.4 returned for each of their slots. Every converted slot must read back
# with hdf5_read_channel as the oracle says; the unstructured datasets are also compared with h5py.

import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

# Append the directory containing the package to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
package_path = os.path.join(current_dir, '../../lib')
if package_path not in sys.path:
    sys.path.append(package_path)

import quadriga_lib
from quadriga_lib import channel

# The script runs with the module that this test imported, so that both always use the same one
module_dir = os.path.dirname(os.path.abspath(quadriga_lib.__file__))
script_env = dict(os.environ)
script_env['PYTHONPATH'] = os.pathsep.join([module_dir] + os.environ.get('PYTHONPATH', '').split(os.pathsep)).rstrip(
    os.pathsep)

script = os.path.join(current_dir, '../../tools/hdf5_convert_v2.py')
data_dir = os.path.join(current_dir, '../data')
fixtures = ('hdf5_v2_written.hdf5', 'hdf5_v2_edited.hdf5', 'hdf5_v2_lossy.hdf5')


def slot_index(s, dims):
    nx, ny, nz = (int(n) for n in dims[:3])
    return s % nx, s // nx % ny, s // (nx * ny) % nz, s // (nx * ny * nz)


def link_names(group):
    """Link names in creation order, or in name order if the group does not track it (as quadriga-lib 0.12.4)."""
    names = []
    try:
        group.id.links.iterate(names.append, idx_type=h5py.h5.INDEX_CRT_ORDER)
    except RuntimeError:
        names = []
        group.id.links.iterate(names.append, idx_type=h5py.h5.INDEX_NAME)
    return [n.decode() for n in names]


def stored_value(dset):
    """Value of a dataset as stored: native-order array, or the bytes of a string without trailing NULs."""
    t = dset.id.get_type()
    if isinstance(t, h5py.h5t.TypeStringID):
        if t.is_variable_str():
            return bytes(np.asarray(dset[()], dtype=object).reshape(-1)[0]).split(b'\0', 1)[0]
        buf = np.empty(dset.shape, dtype='S%d' % t.get_size())
        dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=t)
        return buf.tobytes().rstrip(b'\0')
    a = np.asarray(dset[()])
    return a.astype(a.dtype.newbyteorder('='))


@unittest.skipIf(h5py is None, "h5py is not installed (needed to read version-2 files)")
class test_case(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as d:
            fn = os.path.join(d, 'version.hdf5')
            channel.hdf5_create_file(fn, 1)
            with h5py.File(fn, 'r') as f:
                version = int(f['Version'][()])
        if version != 3:
            raise RuntimeError('quadriga_lib at %s writes channel files of version %d, not 3'
                               % (quadriga_lib.__file__, version))
        with np.load(os.path.join(data_dir, 'hdf5_v2_expected.npz'), allow_pickle=False) as z:
            cls.oracle = {k: z[k] for k in z.files}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = self.tmp.name

    def tearDown(self):
        self.tmp.cleanup()

    def run_script(self, *args):
        r = subprocess.run([sys.executable, script, *args], capture_output=True, text=True, env=script_env)
        return r.returncode, r.stdout, r.stderr

    def run_patched(self, setup, *args):
        """Run the script with runpy after the Python code 'setup' (which may replace modules)."""
        code = 'import sys, runpy\n%s\nsys.argv = %r\nrunpy.run_path(%r, run_name="__main__")' % (
            setup, [script, *args], script)
        r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=script_env)
        return r.returncode, r.stdout, r.stderr

    def convert(self, name, rc=0):
        out = os.path.join(self.dir, 'v3_' + name)
        code, stdout, stderr = self.run_script(os.path.join(data_dir, name), out)
        self.assertEqual(code, rc, stderr)
        return out, stdout, stderr

    def slots(self, name):
        """Occupied slots of the input in the oracle: {linear index: oracle key prefix}."""
        return {int(k.split('/')[1]): '%s/%s/' % (name, k.split('/')[1])
                for k in self.oracle if k.startswith(name + '/') and k.split('/')[1].isdigit()}

    def assert_value(self, actual, expected, where):
        if isinstance(actual, str):
            self.assertEqual(actual, expected.item(), where)
            return
        a = np.asarray(actual)
        self.assertEqual((a.dtype, a.shape), (expected.dtype, expected.shape), where)
        self.assertEqual(a.tobytes(), expected.tobytes(), where)  # also -0.0 and NaN payloads

    def assert_slot(self, out, name, s, skip=()):
        """hdf5_read_channel of slot s of the output equals the oracle; 'skip' names datasets left out."""
        key = self.slots(name)[s]
        o = self.oracle
        chan, par = channel.hdf5_read_channel(out, *slot_index(s, o[name + '/layout']))
        if key + 'chan_keys' not in o:
            self.assertEqual(chan, [], key)
        else:
            self.assertEqual(len(chan), 1, key)
            self.assertEqual(list(chan[0]), list(o[key + 'chan_keys']), key)
            for k, v in chan[0].items():
                if key + 'chan/' + k in o:
                    self.assert_value(v, o[key + 'chan/' + k], key + k)
                    continue
                n = sum(1 for x in o if x.startswith(key + 'chan/' + k + '/'))
                self.assertEqual(len(v), n, key + k)
                for i, x in enumerate(v):
                    self.assert_value(x, o['%schan/%s/%d' % (key, k, i)], '%s%s[%d]' % (key, k, i))
        self.assertEqual(list(par), [k for k in o[key + 'par_keys'] if k not in skip], key)
        for k, v in par.items():
            self.assert_value(v, o[key + 'par/' + k], key + 'par/' + k)

    def assert_h5py(self, name, out, slots, skip=()):
        """The unstructured datasets and 'Initial_position' of the given slots, compared with h5py."""
        with h5py.File(os.path.join(data_dir, name), 'r') as fi, h5py.File(out, 'r') as fo:
            dims = fi['ChannelDims'][()]
            for s in slots:
                gi = fi['channel_%d' % fi['Order'][s]]
                go = fo['channel_%d' % fo['Order'][s]]
                names = [n for n in (link_names(gi['Data']) if 'Data' in gi else []) if n not in skip]
                self.assertEqual(channel.hdf5_read_dset_names(out, *slot_index(s, dims), prefix=''), names, s)
                for n in names:
                    a, b = stored_value(gi['Data'][n]), stored_value(go['Data'][n])
                    self.assertEqual(type(a), type(b), n)
                    if isinstance(a, bytes):
                        self.assertEqual(a, b, n)
                    else:
                        self.assertEqual((a.dtype, a.shape, a.tobytes()), (b.dtype, b.shape, b.tobytes()), n)
                self.assertEqual(gi.attrs.get('Initial_position'), go.attrs.get('Initial_position'), s)

    def assert_layout(self, name, out, has_data=None):
        dims, occupied = channel.hdf5_read_layout(out)
        np.testing.assert_array_equal(dims, self.oracle[name + '/layout'])
        expected = self.oracle[name + '/has_data'] if has_data is None else has_data
        np.testing.assert_array_equal(occupied, expected)

    def test_written(self):
        name = 'hdf5_v2_written.hdf5'
        out, stdout, stderr = self.convert(name)
        self.assertEqual(stderr, '')
        self.assertEqual(stdout, "Converted 8 channels from version 2 to version 3: '%s'\n" % out)
        self.assert_layout(name, out)
        for s in self.slots(name):
            self.assert_slot(out, name, s)
        self.assert_h5py(name, out, self.slots(name))

    def test_edited(self):
        name = 'hdf5_v2_edited.hdf5'
        out, stdout, stderr = self.convert(name)
        self.assertEqual(stderr, '')
        self.assert_layout(name, out)
        for s, key in self.slots(name).items():
            if key + 'error' not in self.oracle:
                self.assert_slot(out, name, s)
        self.assert_h5py(name, out, self.slots(name))
        self.assertIn('Size mismatch', self.oracle[name + '/1/error'].item())  # rejected by 0.12.4 itself
        self.assertIn('Size mismatch', self.oracle[name + '/2/error'].item())

        with h5py.File(os.path.join(data_dir, name), 'r') as f:
            g1, g2 = (f['channel_%d' % f['Order'][s]] for s in (1, 2))
            # No coefficients, last snapshot group deleted: the missing trailing delay is (1, 1, 0)
            chan = channel.hdf5_read_channel(out, 1)[0][0]
            self.assertEqual([d.shape for d in chan['delay']], [(1, 1, 2), (1, 1, 1), (1, 1, 0)])
            for i in range(2):
                np.testing.assert_array_equal(chan['delay'][i], g1['Snap_%d/delay' % i][()].T)
                np.testing.assert_array_equal(chan['path_gain'][i], g1['Snap_%d/path_gain' % i][()])
            self.assertEqual(chan['path_gain'][2].shape, (0,))
            # Delays of shape (1, n_tx, n) and (n_rx, 1, n): every antenna pair gets the value of its row / column
            chan = channel.hdf5_read_channel(out, 2)[0][0]
            for i in range(2):
                stored = g2['Snap_%d/delay' % i][()].T
                self.assertEqual(chan['delay'][i].shape, (2, 3, stored.shape[2]))
                np.testing.assert_array_equal(chan['delay'][i], np.broadcast_to(stored, chan['delay'][i].shape))
                np.testing.assert_array_equal(chan['coeff'][i].real, g2['Snap_%d/coeff_re' % i][()].T)

        # Custom prefixes keep their place in the creation order
        self.assertEqual(channel.hdf5_read_dset_names(out, 7, prefix=''), ['par_a', 'x_b', 'par_c', 'y_d'])
        self.assertEqual(channel.hdf5_read_dset_names(out, 8, prefix=''), ['x_e'])

    def test_lossy(self):
        name = 'hdf5_v2_lossy.hdf5'
        out, stdout, stderr = self.convert(name, rc=3)
        self.assertTrue(stdout.endswith('; 9 items were not converted (see the warnings)\n'), stdout)
        warnings = stderr.splitlines()
        self.assertEqual(len(warnings), 9, stderr)
        for w in ("slot (0, 0, 0, 0): dataset 'par_bad-name' is not converted",
                  "slot (0, 0, 0, 0): dataset 'par_i16' is not converted: unsupported type",
                  "slot (0, 0, 0, 0): dataset 'par_rank4' is not converted: unsupported rank",
                  "slot (0, 0, 0, 0): dataset 'par_enum' is not converted: unsupported type",
                  'slot (1, 0, 0, 0): empty channel; the slot is left empty',
                  "slot (2, 0, 0, 0): channel name 'name only' is dropped",
                  "slot (3, 0, 0, 0): channel name 'name with data' is dropped",
                  "slot (4, 0, 0, 0): dataset 'par_latin1' is not converted: string is not valid UTF-8",
                  'slot (5, 0, 0, 0): channel name truncated to 254 bytes'):
            self.assertEqual(sum(w in x for x in warnings), 1, w)
        self.assertTrue(all(x.startswith('hdf5_convert_v2.py: warning: ') for x in warnings), stderr)

        has_data = self.oracle[name + '/has_data'].copy()
        has_data[1:3] = 0  # the empty and the name-only slot
        self.assert_layout(name, out, has_data)
        skipped = ('par_bad-name', 'par_i16', 'par_rank4', 'par_enum', 'par_latin1')
        self.assert_slot(out, name, 0, skip=('bad-name',))
        self.assertEqual(self.oracle[name + '/3/chan/name'].item(), 'name with data')
        chan, par = channel.hdf5_read_channel(out, 3)
        self.assertEqual(chan, [{'name': ''}])  # the name is dropped, the dataset is kept
        self.assert_value(par['good'], self.oracle[name + '/3/par/good'], 'good')
        self.assert_h5py(name, out, (0, 3, 4), skipped)
        self.assertIn('UnicodeDecodeError', self.oracle[name + '/4/error'].item())
        par = channel.hdf5_read_channel(out, 4)[1]  # every other dataset of the slot
        self.assertEqual(list(par), ['good', 'after'])
        self.assert_value(par['good'], np.arange(4), 'good')
        self.assertEqual(par['after'], 2.0)
        self.assertIn('Name can have at most 255 characters', self.oracle[name + '/5/error'].item())
        chan = channel.hdf5_read_channel(out, 5)[0][0]
        self.assertEqual(chan['name'], 'a' * 254)

    def test_output_exists(self):
        src = os.path.join(data_dir, 'hdf5_v2_written.hdf5')
        out = os.path.join(self.dir, 'out.hdf5')
        with open(out, 'wb') as f:
            f.write(b'keep')
        code, _, stderr = self.run_script(src, out)
        self.assertEqual(code, 1)
        self.assertIn("output file '%s' exists; use --force to replace it" % out, stderr)
        with open(out, 'rb') as f:
            self.assertEqual(f.read(), b'keep')
        code, _, stderr = self.run_script('--force', src, out)
        self.assertEqual(code, 0, stderr)
        np.testing.assert_array_equal(channel.hdf5_read_layout(out)[0], [2, 4, 2, 1])
        self.assertEqual(os.listdir(self.dir), ['out.hdf5'])

    def test_rejected_inputs(self):
        d = self.dir
        text, no_version, v7, v3, same, short, bad = (os.path.join(d, n) for n in (
            'a.txt', 'b.hdf5', 'c.hdf5', 'd.hdf5', 'e.hdf5', 'f.hdf5', 'g.hdf5'))
        with open(text, 'w') as f:
            f.write('not HDF5')
        with h5py.File(no_version, 'w') as f:
            f['ChannelDims'] = np.ones(4, np.uint32)
        shutil.copy(os.path.join(data_dir, 'hdf5_v2_written.hdf5'), v7)
        with h5py.File(v7, 'r+') as f:
            f['Version'][()] = 7
        channel.hdf5_create_file(v3, 2)
        for fn in (same, short, bad):
            shutil.copy(os.path.join(data_dir, 'hdf5_v2_written.hdf5'), fn)
        with h5py.File(short, 'r+') as f:
            order = f['Order'][()]
            del f['Order']
            f['Order'] = order[:-1]
        with h5py.File(bad, 'r+') as f:
            del f['Version']
            f['Version'] = 'two'
        missing, folder, out = (os.path.join(d, n) for n in ('missing.hdf5', 'folder', 'out.hdf5'))
        os.mkdir(folder)
        no_dir = os.path.join(d, 'no_such_folder', 'out.hdf5')
        for args, message in [  # message = the start of the only line on stderr
                ((missing, out), "input file '%s' does not exist" % missing),
                ((text, out), "'%s' is not an HDF5 file" % text),
                ((d, out), "'%s' is not an HDF5 file" % d),
                ((no_version, out), "'%s' is not a quadriga-lib channel file (dataset 'Version' is missing)"
                 % no_version),
                ((v7, out), "'%s' is a channel file of version 7; only version 2 can be converted" % v7),
                ((v3, out), "'%s' is already a channel file of version 3; nothing to convert" % v3),
                ((short, out), "'%s': the storage index is corrupted (ChannelDims and Order do not match)" % short),
                ((same, same), 'input and output are the same file; choose another output name'),
                ((same, folder), "output '%s' is a directory" % folder),
                ((same, no_dir), "cannot create a temporary file next to '%s': " % no_dir),
                ((bad, out), "ValueError: invalid literal for int() with base 10: b'two'; nothing was written")]:
            files = sorted(os.listdir(d))
            code, stdout, stderr = self.run_script(*args)
            self.assertEqual(code, 1, message)
            self.assertEqual(stdout, '', message)
            self.assertTrue(stderr.startswith('hdf5_convert_v2.py: error: ' + message), stderr)
            self.assertEqual(stderr.count('\n'), 1, stderr)
            self.assertEqual(sorted(os.listdir(d)), files, message)
            self.assertEqual(os.listdir(os.path.join(d, 'folder')), [], message)
        code, _, stderr = self.run_script(text)
        self.assertEqual(code, 2)  # argparse usage error
        self.assertIn('usage:', stderr)

    def test_failure_leaves_nothing(self):
        src = os.path.join(self.dir, 'in.hdf5')
        shutil.copy(os.path.join(data_dir, 'hdf5_v2_written.hdf5'), src)
        with h5py.File(src, 'r+') as f:
            group = 'channel_%d' % f['Order'][1]
            del f[group]['coeff_im']
        out = os.path.join(self.dir, 'out.hdf5')
        code, stdout, stderr = self.run_script(src, out)
        self.assertEqual(code, 1)
        self.assertEqual(stdout, '')
        self.assertTrue(stderr.startswith("hdf5_convert_v2.py: error: slot (1, 0, 0, 0), group '%s': " % group),
                        stderr)
        self.assertTrue(stderr.endswith('; nothing was written\n'), stderr)
        self.assertEqual(os.listdir(self.dir), ['in.hdf5'])  # no output, no temporary folder

    def test_missing_modules(self):
        src = os.path.join(data_dir, 'hdf5_v2_written.hdf5')
        out = os.path.join(self.dir, 'out.hdf5')
        for module, hint in (('h5py', 'pip install h5py'), ('quadriga_lib', 'pip install quadriga-lib')):
            code, _, stderr = self.run_patched('sys.modules[%r] = None' % module, src, out)
            self.assertEqual(code, 1, stderr)
            self.assertIn(hint, stderr)
            self.assertEqual(os.listdir(self.dir), [])

        # A module that writes version 2 (quadriga-lib 0.12.4), stubbed: checked on the new temporary file
        stub = ('import types, h5py\n'
                'def create(fn, *dims):\n'
                '    with h5py.File(fn, "w") as f:\n'
                '        f["Version"] = 2\n'
                'sys.modules["quadriga_lib"] = types.ModuleType("quadriga_lib")\n'
                'sys.modules["quadriga_lib"].__file__ = "old/quadriga_lib.so"\n'
                'sys.modules["quadriga_lib"].channel = types.SimpleNamespace(hdf5_create_file=create)')
        code, _, stderr = self.run_patched(stub, src, out)
        self.assertEqual(code, 1, stderr)
        self.assertIn('quadriga_lib at old/quadriga_lib.so writes channel files of version 2; converting needs a '
                      'quadriga-lib version newer than 0.12.4', stderr)
        self.assertEqual(os.listdir(self.dir), [])

    def test_interrupted(self):
        # Ctrl-C, SIGTERM and SIGHUP while the first slot is written: exit 130, no output, no temporary folder
        src = os.path.join(data_dir, 'hdf5_v2_written.hdf5')
        out = os.path.join(self.dir, 'out.hdf5')
        stops = ['raise KeyboardInterrupt']
        stops += ['os.kill(os.getpid(), signal.%s)' % n for n in ('SIGTERM', 'SIGHUP') if os.name == 'posix']
        for stop in stops:
            setup = ('import os, signal\nfrom quadriga_lib import channel\n'
                     'def stop(*args):\n    %s\nchannel.hdf5_write_channel = stop' % stop)
            code, stdout, stderr = self.run_patched(setup, src, out)
            self.assertEqual(code, 130, stop + stderr)
            self.assertEqual((stdout, stderr), ('', 'hdf5_convert_v2.py: error: interrupted; nothing was written\n'))
            self.assertEqual(os.listdir(self.dir), [], stop)

    def test_input_unchanged(self):
        digest = lambda fn: hashlib.sha256(open(os.path.join(data_dir, fn), 'rb').read()).hexdigest()
        before = [digest(fn) for fn in fixtures]
        for fn in fixtures:
            self.convert(fn, rc=3 if 'lossy' in fn else 0)
        self.assertEqual([digest(fn) for fn in fixtures], before)


if __name__ == '__main__':
    unittest.main()
