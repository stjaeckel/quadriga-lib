#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
# Part of quadriga-lib — see LICENSE for terms.

"""Convert a quadriga-lib channel file of version 2 into a channel file of version 3.

Usage:  python tools/hdf5_convert_v2.py [--force] input.hdf5 output.hdf5

Channel files written by quadriga-lib 0.12.4 and earlier use file format version 2, which newer versions of
quadriga-lib reject. This script reads such a file with h5py and writes a version-3 file with the same storage
layout through the quadriga_lib Python module. Every slot of the output reads back with hdf5_read_channel as the
input did with quadriga-lib 0.12.4 (same fields, shapes, values and unstructured datasets in the same order), with
these exceptions: the channel IDs are renumbered; the items under "Limits" are skipped; channel groups that other
tools left without orientations read back with zero orientations; and channels that 0.12.4 read but rejected as
invalid are repaired where no value changes (a missing empty last delay of a channel without coefficients is added
as (1, 1, 0); delays of shape (1, n_tx, n) or (n_rx, 1, n) of writers from 2023/24 are expanded to all antenna
pairs). The input is never changed. The output appears only when the conversion is complete;
an existing output is replaced only with --force. MATLAB and Octave users run the script with Python as well.

Requirements: Python 3.10 or newer, numpy, h5py 3.0 or newer, and a quadriga_lib Python module that writes version 3
(any version newer than 0.12.4). The installed module is used; if there is none, the build in the folder ../lib next
to this script (a source checkout after "make python"). PYTHONPATH=<folder with the module> selects another one.

Limits: some items that only the C++ API with assume_valid = true or edits with other HDF5 tools can create have no
place in version 3 and are skipped with a warning: an occupied slot without any channel data (the slot is left
empty), the name of a channel that has no structured data, the bytes of a name beyond 255, and unstructured datasets
whose name has characters other than letters, digits and '_', whose type or rank (> 3) the library does not support,
or which hold a string that is not valid UTF-8.

Exit codes: 0 converted completely; 3 converted except the items named in the warnings; 1 error (nothing written);
2 wrong arguments; 130 interrupted by Ctrl-C, SIGTERM or SIGHUP (nothing written). Slot indices in messages are
0-based.
"""

import argparse
import os
import re
import shutil
import signal
import sys
import tempfile

import numpy as np

PROG = 'hdf5_convert_v2.py'
SNAP_NAME = re.compile(r'Snap_(0|[1-9][0-9]{0,9})')
DSET_NAME = re.compile(r'[A-Za-z0-9_]+')


class Failure(Exception):
    """An error that ends the run with exit code 1; the text is the complete message."""


def link_names(group):
    """Names of the links in 'group' in creation order, or in name order if the group does not track it."""
    import h5py
    names = []
    try:
        group.id.links.iterate(names.append, idx_type=h5py.h5.INDEX_CRT_ORDER)
    except RuntimeError:
        names = []
        group.id.links.iterate(names.append, idx_type=h5py.h5.INDEX_NAME)
    return [n.decode('utf-8', 'surrogateescape') for n in names]


def read_par(dset):
    """Value of an unstructured dataset as the 0.12.4 reader returned it: (value, None) or (None, reason)."""
    import h5py
    t, shape = dset.id.get_type(), dset.shape
    if shape is None or len(shape) > 3:
        return None, 'unsupported rank'
    if isinstance(t, h5py.h5t.TypeStringID):
        if int(np.prod(shape)) != 1:
            return None, 'string array'
        if t.is_variable_str():
            raw = np.asarray(dset[()], dtype=object).reshape(-1)[0]
            raw = (raw.encode('utf-8') if isinstance(raw, str) else bytes(raw)).split(b'\0', 1)[0]
        else:  # read with the file type as memory type: h5py's default read stops at the first NUL
            buf = np.empty(shape, dtype='S%d' % t.get_size())
            dset.id.read(h5py.h5s.ALL, h5py.h5s.ALL, buf, mtype=t)
            raw = buf.tobytes().rstrip(b'\0')
        try:
            return raw.decode('utf-8'), None
        except UnicodeDecodeError:
            return None, 'string is not valid UTF-8'
    if isinstance(t, h5py.h5t.TypeEnumID) or not isinstance(t, (h5py.h5t.TypeFloatID, h5py.h5t.TypeIntegerID)) \
            or t.get_size() not in (4, 8):
        return None, 'unsupported type'
    a = np.asarray(dset[()])
    a = a.astype(a.dtype.newbyteorder('='), copy=False)
    return (a[()] if a.ndim == 0 else a if a.ndim == 1 else a.T), None


def read_slot(group):
    """Read one channel group: (chan, par, extra, warnings) for hdf5_write_channel and hdf5_write_dset."""
    import h5py
    warnings = []

    def attr(obj, name):
        if name not in obj.attrs:
            raise ValueError("attribute '%s' is missing" % name)
        return obj.attrs[name]

    f32 = lambda x: np.asarray(x, dtype=np.float32)
    is_dset = lambda name: isinstance(group.get(name), h5py.Dataset)

    name = group.attrs.get('Name', b'')  # h5py stops at the first NUL (older writers left bytes after it)
    name = name.encode('utf-8') if isinstance(name, str) else bytes(name)
    if len(name) > 255:
        try:  # keep a UTF-8 name readable: cut at the last character boundary
            name.decode('utf-8')
            name = name[:255].decode('utf-8', 'ignore').encode('utf-8')
        except UnicodeDecodeError:
            name = name[:255]
        warnings.append('channel name truncated to %d bytes' % len(name))
    chan = {'name': name, 'initial_position': int(group.attrs.get('Initial_position', 0))}

    # Unstructured data: the leading 'par_' datasets go into 'par', the rest is written one by one
    par, extra = {}, []
    data = group.get('Data')
    if isinstance(data, h5py.Group):
        for link in link_names(data):
            dset = data.get(link)
            if not isinstance(dset, h5py.Dataset):
                continue
            if not DSET_NAME.fullmatch(link):
                warnings.append("dataset '%s' is not converted: the name has characters other than letters, digits "
                                "and '_'" % link)
                continue
            value, reason = read_par(dset)
            if reason:
                warnings.append("dataset '%s' is not converted: %s" % (link, reason))
            elif link.startswith('par_') and not extra:
                par[link[4:]] = value
            else:
                extra.append((link, value))

    n_snap = int(attr(group, 'NumSnap'))
    if n_snap == 0:
        if chan['initial_position'] == 0 and name:
            warnings.append("channel name '%s' is dropped: version 3 stores a name only together with channel data"
                            % name.decode('utf-8', 'replace'))
        return chan, par, extra, warnings

    n_tx, n_rx = int(attr(group, 'NumTx')), int(attr(group, 'NumRx'))
    if is_dset('CenterFrequency'):
        chan['center_frequency'] = f32(group['CenterFrequency'][()]).reshape(-1)
    elif 'CenterFrequency' in group.attrs:
        cf = f32(group.attrs['CenterFrequency']).reshape(-1)[:1]
        if not np.isnan(cf[0]):
            chan['center_frequency'] = cf
    for key in ('tx_position', 'rx_position', 'tx_orientation', 'rx_orientation'):
        if is_dset(key):
            chan[key] = f32(group[key][()]).T
        elif key in group.attrs or key.endswith('position'):
            chan[key] = f32(attr(group, key)).reshape(3, 1)

    # Snapshot groups: the channel group itself for one snapshot, else 'Snap_<i>' sorted by i
    if n_snap == 1:
        snaps = [(0, group)]
    else:
        index = sorted(int(n[5:]) for n in group if SNAP_NAME.fullmatch(n) and int(n[5:]) < n_snap)
        if index and n_snap - len(index) > 2**20:
            raise ValueError("'NumSnap' is much larger than the number of snapshots in the file")
        snaps = [(i, group['Snap_%d' % i]) for i in index]

    lists = {}

    def put(key, i, value, pad):  # missing snapshots before i get the empty entry 'pad'
        entries = lists.setdefault(key, [])
        entries += [pad] * (i - len(entries))
        entries.append(value)

    empty = {'path_gain': np.zeros(0, np.float32), 'path_length': np.zeros(0, np.float32),
             'path_angles': np.zeros((0, 4), np.float32), 'fbs_pos': np.zeros((3, 0), np.float32),
             'lbs_pos': np.zeros((3, 0), np.float32), 'interact_coord': np.zeros((3, 0), np.float32),
             'path_polarization': np.zeros((4, 0), np.complex64), 'no_interact': np.zeros(0, np.uint32),
             'coeff_re': np.zeros((n_rx, n_tx, 0), np.float32), 'coeff_im': np.zeros((n_rx, n_tx, 0), np.float32)}
    for i, s in snaps:
        if 'coeff_re' in s:
            put('coeff_re', i, f32(s['coeff_re'][()]).T, empty['coeff_re'])
            put('coeff_im', i, f32(s['coeff_im'][()]).T, empty['coeff_im'])
        if 'delay' in s:
            d = f32(s['delay'][()]).T
            if d.ndim != 3:
                raise ValueError("dataset 'delay' has %d dimensions, expected 3" % d.ndim)
            if 'coeff_re' in s and d.shape[:2] not in ((n_rx, n_tx), (1, 1)):  # old writers: rows or columns of 1
                d = np.broadcast_to(d, (n_rx, n_tx, d.shape[2])).copy()
            put('delay', i, d, np.zeros((d.shape[0], d.shape[1], 0), np.float32))
        for v2, key in (('path_gain', 'path_gain'), ('path_length', 'path_length'), ('path_angles', 'path_angles'),
                        ('path_fbs_pos', 'fbs_pos'), ('path_lbs_pos', 'lbs_pos')):
            if v2 in s:
                put(key, i, f32(s[v2][()]).T, empty[key])
        if 'path_polarization' in s:  # interleaved re/im pairs; a view keeps every bit (-0, NaN, Inf)
            pol = np.ascontiguousarray(s['path_polarization'][()], dtype=np.float32).view(np.complex64).T
            put('path_polarization', i, pol, empty['path_polarization'])
        if 'no_interact' in s:
            no_interact = np.asarray(s['no_interact'][()], dtype=np.uint32)
            put('no_interact', i, no_interact, empty['no_interact'])
            coord = f32(s['interact_coord'][()]).T if no_interact.sum() > 0 else empty['interact_coord']
            put('interact_coord', i, coord, empty['interact_coord'])

    # Trailing delays: (n_rx, n_tx, 0) as 0.12.4 with coefficients; without them (1, 1, 0), which is valid
    empty['delay'] = np.zeros((n_rx, n_tx, 0) if 'coeff_re' in lists else (1, 1, 0), np.float32)
    for key, entries in lists.items():
        chan[key] = entries + [empty[key]] * (n_snap - len(entries))
    return chan, par, extra, warnings


def main(argv):
    parser = argparse.ArgumentParser(prog=PROG, description=__doc__.split('\n')[0])
    parser.add_argument('--force', action='store_true', help='replace an existing output file')
    parser.add_argument('input', help='channel file of version 2 (read only)')
    parser.add_argument('output', help='channel file of version 3 to create')
    args = parser.parse_args(argv)
    fn_in, fn_out = args.input, args.output
    say = lambda kind, text: print('%s: %s: %s' % (PROG, kind, text), file=sys.stderr)

    def interrupt(signum, frame):  # a kill or a closed terminal cleans up like Ctrl-C
        raise KeyboardInterrupt
    for name in ('SIGTERM', 'SIGHUP'):  # Windows has no SIGHUP
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), interrupt)
    tmpdir = None
    try:
        if not os.path.exists(fn_in):
            raise Failure("input file '%s' does not exist" % fn_in)
        try:
            import h5py
        except ImportError:
            raise Failure('h5py is needed to read version-2 files: pip install h5py (or conda install h5py)')
        if not h5py.is_hdf5(fn_in):
            raise Failure("'%s' is not an HDF5 file" % fn_in)
        try:
            src = h5py.File(fn_in, 'r')
        except Exception as e:
            raise Failure("cannot open '%s': %s" % (fn_in, e))
        with src:
            for name in ('Version', 'ChannelDims', 'Order'):
                if not isinstance(src.get(name), h5py.Dataset):
                    raise Failure("'%s' is not a quadriga-lib channel file (dataset '%s' is missing)" % (fn_in, name))
            version = int(src['Version'][()])
            if version == 3:
                raise Failure("'%s' is already a channel file of version 3; nothing to convert" % fn_in)
            if version != 2:
                raise Failure("'%s' is a channel file of version %d; only version 2 can be converted"
                              % (fn_in, version))
            nx, ny, nz, nw = (int(n) for n in src['ChannelDims'][()])
            order = src['Order'][()]
            if nx * ny * nz * nw != order.size:
                raise Failure("'%s': the storage index is corrupted (ChannelDims and Order do not match)" % fn_in)
            if os.path.isdir(fn_out):
                raise Failure("output '%s' is a directory" % fn_out)
            if os.path.exists(fn_out) and os.path.samefile(fn_in, fn_out):
                raise Failure('input and output are the same file; choose another output name')
            if os.path.exists(fn_out) and not args.force:
                raise Failure("output file '%s' exists; use --force to replace it" % fn_out)

            try:
                try:
                    import quadriga_lib
                except ImportError:  # not installed: the build of a source checkout ("make python") next to the script
                    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'lib'))
                    import quadriga_lib
                from quadriga_lib import channel as qc
            except ImportError:
                raise Failure('the quadriga_lib Python module (a version newer than 0.12.4) is needed: pip install '
                              'quadriga-lib, or add the folder with the built module (quadriga-lib/lib) to PYTHONPATH')

            try:
                tmpdir = tempfile.mkdtemp(dir=os.path.dirname(os.path.abspath(fn_out)), prefix='.hdf5_convert_v2-')
                tmp = os.path.join(tmpdir, os.path.basename(fn_out))
                qc.hdf5_create_file(tmp, nx, ny, nz, nw)
            except Exception as e:
                raise Failure("cannot create a temporary file next to '%s': %s" % (fn_out, e))
            with h5py.File(tmp, 'r') as f:
                version = int(f['Version'][()])
            if version != 3:
                raise Failure('quadriga_lib at %s writes channel files of version %d; converting needs a quadriga-lib '
                              'version newer than 0.12.4 (one that writes version 3); a newer module can be selected '
                              'with PYTHONPATH=<folder with the module>' % (quadriga_lib.__file__, version))

            n_chan, n_lost = 0, 0
            for s in np.flatnonzero(order):
                ix, iy, iz, iw = int(s % nx), int(s // nx % ny), int(s // (nx * ny) % nz), int(s // (nx * ny * nz))
                group = 'channel_%d' % order[s]
                try:
                    chan, par, extra, warnings = read_slot(src[group])
                    filled = len(chan) > 2 or chan['initial_position'] != 0  # n_snap > 0 adds the positions
                    if filled or par:
                        qc.hdf5_write_channel(tmp, chan, par or None, ix, iy, iz, iw)
                    for name, value in extra:
                        qc.hdf5_write_dset(tmp, ix, iy, iz, iw, name, value, prefix='')
                except Exception as e:
                    text = e.args[0] if isinstance(e, KeyError) and e.args else e
                    raise Failure("slot (%d, %d, %d, %d), group '%s': %s; nothing was written"
                                  % (ix, iy, iz, iw, group, text))
                written = filled or par or extra
                if not written and not chan['name']:
                    warnings.append('empty channel; the slot is left empty')
                for w in warnings:
                    say('warning', 'slot (%d, %d, %d, %d): %s' % (ix, iy, iz, iw, w))
                n_chan += bool(written)
                n_lost += len(warnings)
        os.replace(tmp, fn_out)
    except Failure as e:
        say('error', e)
        return 1
    except KeyboardInterrupt:
        say('error', 'interrupted; nothing was written')
        return 130
    except Exception as e:
        say('error', '%s: %s; nothing was written' % (type(e).__name__, e))
        return 1
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)

    summary = 'Converted %d channels from version 2 to version 3: \'%s\'' % (n_chan, fn_out)
    if n_lost:
        print(summary + '; %d items were not converted (see the warnings)' % n_lost)
        return 3
    print(summary)
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
