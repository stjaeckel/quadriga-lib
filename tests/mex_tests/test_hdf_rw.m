% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_hdf_rw
% MEX interface test for the HDF5 functions: quadriga_lib.hdf5_create_file, hdf5_read_layout,
% hdf5_reshape_layout, hdf5_write_channel, hdf5_read_channel, hdf5_write_dset, hdf5_read_dset and
% hdf5_read_dset_names.
% Scope: argument and output handling, defaults, index conversion, type handling, error paths.
% The file format and the core behavior are tested in the Catch2 suite (test_hdf_functions.cpp).

% MOxUnit Setup
tmp = which('MOxUnitTestSuite');
if isempty(tmp)
    current_dir = pwd;
    cd('../external/MOxUnit-master/MOxUnit');
    moxunit_set_path();
    cd(current_dir);
end

fn = 'hdf_mex.hdf5';
fn2 = 'hdf_mex_multi.hdf5';
fn3 = 'hdf_mex_dims.hdf5';
non_hdf5 = 'not_hdf5.txt';
delete_files( fn, fn2, fn3, non_hdf5 );

%% hdf5_create_file

% Default layout; no output
quadriga_lib.hdf5_create_file(fn);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([65536, 1, 1, 1]));

% Recreating an existing file errors, also a file that is not an HDF5 file (thrown by C++)
assert_cpp_error( @() quadriga_lib.hdf5_create_file(fn), 'already exists.' );
fid = fopen(non_hdf5, 'w');
fprintf(fid, 'text');
fclose(fid);
assert_cpp_error( @() quadriga_lib.hdf5_create_file(non_hdf5), 'already exists.' );
delete(non_hdf5);

% Empty nx is the default
delete(fn);
quadriga_lib.hdf5_create_file(fn, []);
assertEqual(quadriga_lib.hdf5_read_layout(fn), uint32([65536, 1, 1, 1]));

% Custom storage layout; omitted dimensions are 1
delete(fn);
quadriga_lib.hdf5_create_file(fn, 12, 12);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([12, 12, 1, 1]));

% Non-double input, column vector
delete(fn);
quadriga_lib.hdf5_create_file(fn, single(3), single(2), single(1), single(2));
assertEqual(quadriga_lib.hdf5_read_layout(fn), uint32([3, 2, 1, 2]));

% Errors
delete(fn);
assert_cpp_error( @() quadriga_lib.hdf5_create_file(), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_create_file(fn, 1, 1, 1, 1, 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out1(@quadriga_lib.hdf5_create_file, fn), 'Wrong number of output arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_create_file(fn, 0, 1), 'must be at least 1' );   % thrown by C++
assert_cpp_error( @() quadriga_lib.hdf5_create_file(fn, 2.5), 'Input ''nx'' must be an integer from 0 to 4294967295.' );
assert_cpp_error( @() quadriga_lib.hdf5_create_file(fn, 65536, 65536), 'too many slots' );   % thrown by C++
assert_cpp_error( @() quadriga_lib.hdf5_create_file(5), 'Input ''fn'' must be a string' );
assertTrue( ~exist(fn, 'file') );

%% hdf5_reshape_layout

% Reshape layout (preserve total slot count)
quadriga_lib.hdf5_create_file(fn, 12, 12);
quadriga_lib.hdf5_reshape_layout(fn, 1, 1, 18, 8);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([1, 1, 18, 8]));

% Integer input
quadriga_lib.hdf5_reshape_layout(fn, uint32(144), uint32(1));
assertEqual(quadriga_lib.hdf5_read_layout(fn), uint32([144, 1, 1, 1]));

% Errors
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(fn, 145), 'must give the same number of slots as the file' );
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(fn, 0, 144), 'must give the same number of slots as the file' );   % thrown by C++
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(fn), 'Wrong number of input arguments.' );   % nx is mandatory
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(fn, 144, 1, 1, 1, 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out1(@quadriga_lib.hdf5_reshape_layout, fn, 144), 'Wrong number of output arguments.' );

% Omitted ny, nz, nw are 1; an empty nx is an error
delete(fn);
quadriga_lib.hdf5_create_file(fn, 256, 256);
quadriga_lib.hdf5_reshape_layout(fn, 65536);
assertEqual(quadriga_lib.hdf5_read_layout(fn), uint32([65536, 1, 1, 1]));
quadriga_lib.hdf5_reshape_layout(fn, 256, 256, [], []);
assertEqual(quadriga_lib.hdf5_read_layout(fn), uint32([256, 256, 1, 1]));
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(fn, []), 'must give the same number of slots as the file' );
delete(fn);

% Reshape on missing file
assert_cpp_error( @() quadriga_lib.hdf5_reshape_layout(fn, 144), 'does not exist.' );

% Stored data keeps its linear slot: 1-based (2, 3) in [4, 3] is slot 10
quadriga_lib.hdf5_create_file(fn, 4, 3);
quadriga_lib.hdf5_write_channel(fn, struct('name', 'keep', 'rx_pos', [1; 2; 3], 'tx_pos', [0; 0; 0], ...
    'coeff_re', 1, 'coeff_im', 0, 'delay', 1e-6), [], 2, 3);
quadriga_lib.hdf5_reshape_layout(fn, 2, 3, 1, 2);
[~, has_data] = quadriga_lib.hdf5_read_layout(fn);
assertTrue(has_data(2, 2, 1, 2));
assertEqual(nnz(has_data), 1);
chanR = quadriga_lib.hdf5_read_channel(fn, 2, 2, 1, 2);
assertEqual(chanR.name, 'keep');
delete(fn);

%% hdf5_read_layout

% Non-existent file: returns [0,0,0,0] and empty has_data
[storage_space, has_data] = quadriga_lib.hdf5_read_layout('does_not_exist.hdf5');
assertEqual(storage_space, uint32([0, 0, 0, 0]));
assertTrue(isempty(has_data));
assertTrue(islogical(has_data));

% Occupancy mask has the layout size and is 1-based
quadriga_lib.hdf5_create_file(fn, 3, 2, 1, 2);
quadriga_lib.hdf5_write_dset(fn, 'x', 1, 3, 1, 1, 2);
[storage_space, has_data] = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([3, 2, 1, 2]));
assertEqual(size(has_data), [3, 2, 1, 2]);
assertTrue(islogical(has_data));
assertTrue(has_data(3, 1, 1, 2));
assertEqual(nnz(has_data), 1);
quadriga_lib.hdf5_read_layout(fn);   % no output requested
delete(fn);

% Errors
assert_cpp_error( @() quadriga_lib.hdf5_read_layout(), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_layout(fn, 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out3(@quadriga_lib.hdf5_read_layout, fn), 'Wrong number of output arguments.' );

% Reading a non-HDF5 file errors
fid = fopen(non_hdf5, 'w');
fprintf(fid, 'This is not HDF5 data');
fclose(fid);
assert_cpp_error( @() quadriga_lib.hdf5_read_layout(non_hdf5), 'Not an HDF5 file' );
delete(non_hdf5);

%% Unstructured data (par) round-trip

% Build par struct covering all supported types and shapes
par = struct;
par.string = 'Buy Bitcoin!';
par.double = 21e6;
par.single = single(pi);
par.uint32 = uint32(21);
par.int32  = int32(-11001001);
par.uint64 = uint64(21e6 * 100e6);
par.int64  = -int64(21e6 * 100e6);
par.double_Col = (0:0.1:1)';
par.single_Col = -single((0:0.1:1)');
par.uint32_Col = uint32((14:18)');
par.int32_Col  = -int32((14:18)');
par.uint64_Col = uint64(21e6 * 100e6 + [0; 1]);
par.int64_Col  = -int64(21e6 * 100e6 + [0; 1]);
par.double_Row = 1:0.1:2;
par.single_Row = -single(1:0.1:2);
par.uint32_Row = uint32(17:19);
par.int32_Row  = -int32(12:19);
par.uint64_Row = uint64(21e6 * 100e6 + [2, 3]);
par.int64_Row  = -int64(21e6 * 100e6 + [3, 4]);
par.double_Mat = rand(4);
par.single_Mat = -single(rand(5));
par.uint32_Mat = randi(10, 3, 'uint32');
par.int32_Mat  = -randi(10, 4, 'int32');
par.uint64_Mat = uint64(randi(10, 5, 'uint32')) + uint64(21e6 * 100e6);
par.int64_Mat  = int64(randi(10, 6, 'int32'))  - int64(21e6 * 100e6);
par.double_Cube = rand(4, 3, 2);
par.single_Cube = -single(rand(5, 4, 3));
par.uint32_Cube = randi(10, [3, 3, 4], 'uint32');
par.int32_Cube  = -randi(10, [4, 5, 6], 'int32');
par.uint64_Cube = uint64(randi(10, [5, 6, 7], 'uint32')) + uint64(21e6 * 100e6);
par.int64_Cube  = int64(randi(10, [6, 7, 8], 'int32'))  - int64(21e6 * 100e6);

% Create file with layout large enough for all subsequent writes
quadriga_lib.hdf5_create_file(fn, 128, 8, 8, 8);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([128, 8, 8, 8]));

% Write par via hdf5_write_channel (requires a chan; use a minimal dummy)
dummy_chan = struct('name', 'par_holder');
code = quadriga_lib.hdf5_write_channel(fn, dummy_chan, par, 1, 1, 1, 1);
assertEqual(code, int32(0));   % new slot

% Too many outputs
assert_cpp_error( @() out2(@quadriga_lib.hdf5_write_channel, fn, dummy_chan, par, 2), 'Wrong number of output arguments' );

% Round-trip par via hdf5_read_channel
[chanR, parR] = quadriga_lib.hdf5_read_channel(fn);
assertTrue(~isfield(chanR, 'coeff_re'));    % no structured data at slot [1,1,1,1]
assertTrue(~isfield(chanR, 'rx_pos'));

fieldsPar  = fieldnames(par);
fieldsParR = fieldnames(parR);
assertEqual(length(fieldsPar), length(fieldsParR));
assertTrue(all(strcmp(fieldsPar, fieldsParR)));

for n = 1:length(fieldsPar)
    f = fieldsPar{n};
    assertEqual(class(par.(f)), class(parR.(f)));
    assertTrue(isequal(par.(f), parR.(f)));
end

% Dataset names list
par_names = quadriga_lib.hdf5_read_dset_names(fn);
assertEqual(length(fieldsPar), length(par_names));
assertTrue(all(strcmp(fieldsPar, par_names)));

% Per-field read/write_dset round-trip
for n = 1:length(fieldsPar)
    f = fieldsPar{n};
    data = quadriga_lib.hdf5_read_dset(fn, f, 1);
    assertTrue(isequal(par.(f), data));
    quadriga_lib.hdf5_write_dset(fn, f, data, 1, 2);
end

par_names = quadriga_lib.hdf5_read_dset_names(fn, 1, 2);
assertEqual(numel(par_names), length(fieldsPar));

% Overwriting a dset should error
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'string', 'Oh no, I bought Ethereum.', 1, 2), 'already exists' );

% Names of an empty location
par_names = quadriga_lib.hdf5_read_dset_names(fn, 12, 2, 2);
assertTrue(isempty(par_names));

% Reading a missing dataset returns []
missing = quadriga_lib.hdf5_read_dset(fn, 'does_not_exist', 1);
assertTrue(isempty(missing));

% Custom prefix on dset functions
quadriga_lib.hdf5_write_dset(fn, 'foo', pi, 1, 3, [], [], 'custom_');
% Default prefix lookup must not see it
names_default = quadriga_lib.hdf5_read_dset_names(fn, 1, 3);
assertTrue(~any(strcmp(names_default, 'foo')));
% Custom prefix lookup must see it
names_custom = quadriga_lib.hdf5_read_dset_names(fn, 1, 3, [], [], 'custom_');
assertTrue(any(strcmp(names_custom, 'foo')));
val = quadriga_lib.hdf5_read_dset(fn, 'foo', 1, 3, [], [], 'custom_');
assertEqual(val, pi);

% Complex data is rejected
parC = struct('complex', 1 + 21i);
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, dummy_chan, parC, 8), 'is complex; complex inputs are not supported.' );

%% hdf5_write_dset, hdf5_read_dset, hdf5_read_dset_names: arguments

% ix, iy, iz, iw: omitted or [] are 1; non-double classes
quadriga_lib.hdf5_write_dset(fn, 'loc_a', 1, [], []);
quadriga_lib.hdf5_write_dset(fn, 'loc_b', 2, uint32(2), uint32(3));
quadriga_lib.hdf5_write_dset(fn, 'loc_c', 3, single(4), single(5), single(6));
quadriga_lib.hdf5_write_dset(fn, 'loc_d', 4, 7, 8, 2, 3);
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_a', 1, 1, 1, 1), 1);
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_a'), 1);
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_b', 2, 3, 1, 1), 2);
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_c', int32(4), int32(5), int32(6)), 3);
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_d', 7, 8, 2, 3), 4);
assertEqual(quadriga_lib.hdf5_read_dset_names(fn, 7, 8, 2, 3), {'loc_d'});      % cell column
assertEqual(quadriga_lib.hdf5_read_dset_names(fn, uint64(7), uint64(8), uint64(2), uint64(3), []), {'loc_d'});
assertTrue(any(strcmp(quadriga_lib.hdf5_read_dset_names(fn, []), 'loc_a')));
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_b', 2, 3, [], [], []), 2);           % [] prefix is 'par_'
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'loc_b', 2, 3, [], [], 'par_'), 2);

% An empty string '' is an empty prefix: full names, also on write
assertEqual(quadriga_lib.hdf5_read_dset_names(fn, 2, 3, [], [], ''), {'par_loc_b'});
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'par_loc_b', 2, 3, [], [], ''), 2);
quadriga_lib.hdf5_write_dset(fn, 'raw', 6, 2, 3, [], [], '');
assertEqual(quadriga_lib.hdf5_read_dset_names(fn, 2, 3, [], [], ''), {'par_loc_b'; 'raw'});
assertEqual(quadriga_lib.hdf5_read_dset_names(fn, 2, 3), {'loc_b'});
assertEqual(quadriga_lib.hdf5_read_dset(fn, 'raw', 2, 3, [], [], ''), 6);

% Empty and missing results
empty_names = quadriga_lib.hdf5_read_dset_names(fn, 5, 5, 5, 5);
assertTrue(iscell(empty_names));
assertEqual(size(empty_names), [0, 1]);
missing = quadriga_lib.hdf5_read_dset(fn, 'loc_b', 1, 1);   % dataset at another slot
assertEqual(size(missing), [0, 0]);
assertTrue(isa(missing, 'double'));
assertTrue(isempty(quadriga_lib.hdf5_read_dset(fn, 'loc_a', 129, 1)));         % outside the layout
assertTrue(isempty(quadriga_lib.hdf5_read_dset_names(fn, 1, 9)));

% No output requested (hdf5_write_dset has none)
quadriga_lib.hdf5_write_dset(fn, 'no_out', 5);
quadriga_lib.hdf5_read_dset(fn, 'no_out');
quadriga_lib.hdf5_read_dset_names(fn);

% Errors
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x'), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', 1, 1, 1, 1, 1, 'par_', 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out1(@quadriga_lib.hdf5_write_dset, fn, 'x', 1), 'Wrong number of output arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset(fn), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset(fn, 'x', 1, 1, 1, 1, 'par_', 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out2(@quadriga_lib.hdf5_read_dset, fn, 'x'), 'Wrong number of output arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset_names(), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset_names(fn, 1, 1, 1, 1, 'par_', 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out2(@quadriga_lib.hdf5_read_dset_names, fn), 'Wrong number of output arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset(fn, 'x', 1, 0), 'cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset_names(fn, 1, 1, 0), 'cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', 1, 1, 1, 1, 0), 'cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', 1, 129, 1), 'must address a slot inside the storage layout' );   % thrown by C++
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', []), 'Input ''data'' cannot be empty.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', true), 'Input ''data'' has an unsupported data type.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', int16(1)), 'Input ''data'' has an unsupported data type.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', {1}), 'Input ''data'' has an unsupported data type.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', 1 + 2i), 'is complex; complex inputs are not supported.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'a-b', 1), 'Parameter name must only contain' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, '', 1), 'Input ''name'' cannot be empty.' );   % thrown by C++
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 'x', 1, [], [], [], [], 'p-'), 'Parameter name must only contain' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn, 1, 1), 'Input ''name'' must be a string' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset(1, 'x'), 'Input ''fn'' must be a string' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset(fn2, 'x'), 'does not exist.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_dset_names(fn2), 'does not exist.' );

%% Structured (chan) round-trip
% File stores structured data in single precision; wrapper reads back as double.
% So expected values for round-trip are double(single(input)).

rx_pos   = rand(3, 1);
tx_pos   = rand(3, 1);
coeff_re = rand(3, 2, 5, 3);
coeff_im = rand(3, 2, 5, 3);
delay_4d = rand(3, 2, 5, 3);

% Minimal chan: positions, coefficients, delay
chan = struct;
chan.rx_pos = rx_pos;
chan.tx_pos = tx_pos;
chan.coeff_re    = coeff_re;
chan.coeff_im    = coeff_im;
chan.delay       = delay_4d;
quadriga_lib.hdf5_write_channel(fn, chan, [], 3);

% Restore and compare
chanR = quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1);
assertEqual(chanR.rx_pos, double(single(rx_pos)));
assertEqual(chanR.tx_pos, double(single(tx_pos)));
assertEqual(chanR.coeff_re,    double(single(coeff_re)));
assertEqual(chanR.coeff_im,    double(single(coeff_im)));
assertEqual(chanR.delay,       double(single(delay_4d)));

% Single precision input gives the same file content
chan_single = chan;
chan_single.rx_pos = single(rx_pos);
chan_single.coeff_re    = single(coeff_re);
chan_single.delay       = single(delay_4d);
quadriga_lib.hdf5_write_channel(fn, chan_single, [], 9);
chanR2 = quadriga_lib.hdf5_read_channel(fn, 9, 1, 1, 1);
assertEqual(chanR2, chanR);

% Reverse snapshot order via snap argument
chanR = quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1, [3, 2, 1]);
assertEqual(chanR.coeff_re, double(single(coeff_re(:, :, :, [3, 2, 1]))));
assertEqual(chanR.coeff_im, double(single(coeff_im(:, :, :, [3, 2, 1]))));
assertEqual(chanR.delay,    double(single(delay_4d(:, :, :, [3, 2, 1]))));

% uint64 snap input is not modified by the call, a second call returns the same data
snap = uint64([2, 3]); keep = snap;
chanR = quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1, snap);
assertEqual(snap, uint64([2, 3]));
assertEqual(keep, uint64([2, 3]));
assertEqual(chanR.coeff_re, double(single(coeff_re(:, :, :, [2, 3]))));
chanR2 = quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1, snap);
assertEqual(chanR2.coeff_re, chanR.coeff_re);
assertEqual(chanR2.delay, chanR.delay);

% Empty snap reads all snapshots
chanR = quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1, []);
assertEqual(chanR.coeff_re, double(single(coeff_re)));

% Snapshot index out of bounds
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1, 0), 'Entries in ''i_snap'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn, 3, 1, 1, 1, 4), 'exceed the number of snapshots' );   % thrown by C++

% i_snap does not apply to a slot without snapshots: it keeps its unstructured data
[chanR, parR] = quadriga_lib.hdf5_read_channel(fn, 1, 1, 1, 1, 2);
assertTrue(~isfield(chanR, 'coeff_re'));
assertEqual(parR.string, par.string);

% A delay of size [1, 1, n_path, n_snap] (one value per path and snapshot) applies to all antenna pairs
delay_2d = rand(5, 3);
chan = struct;
chan.rx_pos = rx_pos;
chan.tx_pos = tx_pos;
chan.coeff_re    = coeff_re;
chan.coeff_im    = coeff_im;
chan.delay       = reshape(delay_2d, [1, 1, size(delay_2d, 1), size(delay_2d, 2)]);
quadriga_lib.hdf5_write_channel(fn, chan, [], 4);
[chanR, ~] = quadriga_lib.hdf5_read_channel(fn, 4, 1, 1, 1);
assertEqual(squeeze(chanR.delay), double(single(delay_2d)));

% Full chan: every structured field populated
center_freq  = 21e9;
name              = 'buy_more_bitcoin';
path_gain         = rand(5, 3);
path_length       = rand(5, 3);
path_polarization = rand(8, 5, 3);
path_angles       = rand(5, 4, 3);
fbs_pos           = rand(3, 5, 3);
lbs_pos           = rand(3, 5, 3);
n_interact       = [1 2 3 4 5 ; 5 4 3 2 1 ; 1 1 1 1 1]';
interact_coord    = rand(3, 15, 3);
interact_coord(:, 6:end, 3) = 0;   % only 5 interactions in snap 3
rx_orientation    = rand(3, 3);
tx_orientation    = rand(3, 3);

chan = struct;
chan.name              = name;
chan.rx_pos            = rx_pos;
chan.tx_pos            = tx_pos;
chan.coeff_re          = coeff_re;
chan.coeff_im          = coeff_im;
chan.delay             = delay_4d;
chan.center_freq  = center_freq;
chan.path_gain         = path_gain;
chan.path_length       = path_length;
chan.polmat = path_polarization;
chan.path_angles       = path_angles;
chan.fbs_pos           = fbs_pos;
chan.lbs_pos           = lbs_pos;
chan.n_interact       = n_interact;
chan.interact_coord    = interact_coord;
chan.rx_orientation    = rx_orientation;
chan.tx_orientation    = tx_orientation;
quadriga_lib.hdf5_write_channel(fn, chan, [], 5);

% Read back full chan and verify every field
[chanR, parR] = quadriga_lib.hdf5_read_channel(fn, 5, 1, 1, 1);
assertTrue(isempty(fieldnames(parR)));
assertEqual(chanR.name, name);
assertEqual(chanR.rx_pos,            double(single(rx_pos)));
assertEqual(chanR.tx_pos,            double(single(tx_pos)));
assertEqual(chanR.coeff_re,          double(single(coeff_re)));
assertEqual(chanR.coeff_im,          double(single(coeff_im)));
assertEqual(chanR.delay,             double(single(delay_4d)));
assertEqual(chanR.center_freq,  double(single(center_freq)));
assertEqual(chanR.path_gain,         double(single(path_gain)));
assertEqual(chanR.path_length,       double(single(path_length)));
assertEqual(chanR.polmat, double(single(path_polarization)));
assertEqual(chanR.path_angles,       double(single(path_angles)));
assertEqual(chanR.fbs_pos,           double(single(fbs_pos)));
assertEqual(chanR.lbs_pos,           double(single(lbs_pos)));
assertEqual(chanR.n_interact,       uint32(n_interact));
assertEqual(chanR.interact_coord,    double(single(interact_coord)));
assertEqual(chanR.rx_orientation,    double(single(rx_orientation)));
assertEqual(chanR.tx_orientation,    double(single(tx_orientation)));

% Single-snapshot subset
[chanR, ~] = quadriga_lib.hdf5_read_channel(fn, 5, 1, 1, 1, 3);
assertEqual(size(chanR.coeff_re, 4), 1);
assertEqual(chanR.coeff_re,  double(single(coeff_re(:, :, :, 3))));
assertEqual(chanR.path_gain, double(single(path_gain(:, 3))));

% initial_position round-trip
chan_ip = chan;
chan_ip.initial_position = int32(7);
quadriga_lib.hdf5_write_channel(fn, chan_ip, [], 6);
[chanR, ~] = quadriga_lib.hdf5_read_channel(fn, 6, 1, 1, 1);
assertEqual(chanR.initial_position, int32(7));

% initial_position is 1-based in MATLAB (0-based in C++ and in the file): 1 is the first snapshot, the default,
% which the struct does not show; 0 is rejected
chan_ip.initial_position = int32(1);
quadriga_lib.hdf5_write_channel(fn, chan_ip, [], 6);
[chanR, ~] = quadriga_lib.hdf5_read_channel(fn, 6, 1, 1, 1);
assertFalse(isfield(chanR, 'initial_position'));
chan_ip.initial_position = int32(0);
assert_cpp_error(@() quadriga_lib.hdf5_write_channel(fn, chan_ip, [], 6), 'Input ''chan.initial_position'' cannot be 0 (1-based index).');

% Combined par + chan write with sparse field set
rx_pos = rand(3, 3);
tx_pos = rand(3, 3);
center_freq = rand(3, 1);
chan = struct;
chan.name              = name;
chan.initial_position  = int32(2);
chan.rx_pos            = rx_pos;
chan.tx_pos            = tx_pos;
chan.coeff_re          = coeff_re;
chan.coeff_im          = coeff_im;
chan.delay             = delay_4d;
chan.center_freq  = center_freq;
chan.rx_orientation    = rx_orientation(:, 1);
chan.tx_orientation    = tx_orientation(:, 2);
quadriga_lib.hdf5_write_channel(fn, chan, par, 6, 1, 1, 2);

[chanR, parR] = quadriga_lib.hdf5_read_channel(fn, 6, 1, 1, 2);
assertTrue(~isempty(fieldnames(parR)));
assertEqual(chanR.rx_pos,            double(single(rx_pos)));
assertEqual(chanR.tx_pos,            double(single(tx_pos)));
assertEqual(chanR.coeff_re,          double(single(coeff_re)));
assertEqual(chanR.coeff_im,          double(single(coeff_im)));
assertEqual(chanR.delay,             double(single(delay_4d)));
assertEqual(chanR.center_freq,  double(single(center_freq)));
assertEqual(chanR.name,              name);
assertEqual(chanR.initial_position,  int32(2));
assertEqual(chanR.rx_orientation,    double(single(rx_orientation(:, 1))));
assertEqual(chanR.tx_orientation,    double(single(tx_orientation(:, 2))));
% Fields not written are absent from the result struct
assertTrue(~isfield(chanR, 'path_gain'));
assertTrue(~isfield(chanR, 'path_length'));
assertTrue(~isfield(chanR, 'polmat'));
assertTrue(~isfield(chanR, 'path_angles'));
assertTrue(~isfield(chanR, 'fbs_pos'));
assertTrue(~isfield(chanR, 'lbs_pos'));
assertTrue(~isfield(chanR, 'n_interact'));
assertTrue(~isfield(chanR, 'interact_coord'));

% Writing to an occupied slot replaces it; code is 1 there, 0 for a new slot; no warning
lastwarn('');
assertEqual(quadriga_lib.hdf5_write_channel(fn, chan, [], 7, 1, 1, 2), int32(0));
assertEqual(quadriga_lib.hdf5_write_channel(fn, [chan, chan], [], [7, 6], 1, 1, 2), int32([1; 1]));
assertTrue(isempty(lastwarn));
[~, parR] = quadriga_lib.hdf5_read_channel(fn, 6, 1, 1, 2);
assertTrue(isempty(fieldnames(parR)));   % the datasets of the old slot are removed

%% Layout / has_data inspection

% has_data is logical
[storage_space, has_data] = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([128, 8, 8, 8]));
assertTrue(islogical(has_data));
% Slots we wrote to
assertTrue(has_data(1, 1, 1, 1));
assertTrue(has_data(1, 2, 1, 1));
assertTrue(has_data(1, 3, 1, 1));
assertTrue(has_data(3, 1, 1, 1));
assertTrue(has_data(4, 1, 1, 1));
assertTrue(has_data(5, 1, 1, 1));
assertTrue(has_data(6, 1, 1, 1));
assertTrue(has_data(6, 1, 1, 2));
% A slot we did not touch
assertTrue(~has_data(20, 1, 1, 1));

%% hdf5_write_channel: wrapper-level input validation

% par size must match chan size
chan2 = repmat(chan, 2, 1);
par2  = repmat(struct('x', 1), 3, 1);
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan2, par2, [7, 1]), 'must have the same number of entries' );

% Non-struct par is rejected
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, [1, 2, 3], 7), 'must be a struct' );

% Non-struct or empty chan is rejected
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, [1, 2, 3], [], 7), 'Input ''chan'' must be a non-empty struct' );
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, struct([]), [], 7), 'Input ''chan'' must be a non-empty struct' );

% Zero or non-integer location is rejected
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, [], 0), 'Entries in ''ix'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, [], 7, 1, 1, 0), 'Entries in ''iw'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, [], 1.5), 'Entries in ''ix'' must be integers from 1 to 4294967295.' );

% Errors from C++: index outside the layout, invalid channel, unsupported par type
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, [], 129), 'Entry 1: Slot is outside the storage layout.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, [chan, chan], [], 7), 'Entry 2: Slot is listed twice.' );assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, struct('rx_pos', [1; 2], 'coeff_re', 1, ...
    'coeff_im', 1, 'delay', 1), [], 7), '''rx_pos'' is missing or ill-formatted' );
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, struct('l', true), 7), 'Input ''par.l'' has an unsupported data type.' );

% Argument counts
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn, chan, [], 1, 1, 1, 1, 1), 'Wrong number of input arguments.' );

% hdf5_write_dset creates a missing file that just holds the slot
delete_files(fn2);
quadriga_lib.hdf5_write_dset(fn2, 'foo', pi, 2, 3);
assertEqual(quadriga_lib.hdf5_read_layout(fn2), uint32([2, 3, 1, 1]));
assertEqual(quadriga_lib.hdf5_read_dset(fn2, 'foo', 2, 3), pi);
delete(fn2);

%% Multi-channel writes (struct array)

% Build a 3-element chan struct array
N = 3;
chans(N) = struct();   % preallocate
for k = 1:N
    chans(k).name             = sprintf('multi_%d', k);
    chans(k).rx_pos           = rand(3, 1);
    chans(k).tx_pos           = rand(3, 1);
    chans(k).coeff_re         = rand(2, 2, 4, 2);
    chans(k).coeff_im         = rand(2, 2, 4, 2);
    chans(k).delay            = rand(2, 2, 4, 2);
end

% A new file just holds the slots; one code per channel
code = quadriga_lib.hdf5_write_channel(fn2, chans);
assertEqual(code, int32(zeros(N, 1)));
assertEqual(quadriga_lib.hdf5_read_layout(fn2), uint32([N, 1, 1, 1]));

% Read all back with no indices — vector return
chansR = quadriga_lib.hdf5_read_channel(fn2);
assertEqual(numel(chansR), N);
for k = 1:N
    assertEqual(chansR(k).name, chans(k).name);
    assertEqual(chansR(k).coeff_re, double(single(chans(k).coeff_re)));
end

% A member filled only by a later channel is still a field; the other channels have [] there
delete(fn2);
mixed = chans(1:2);
mixed(2).fbs_pos = rand(3, 4, 2);
quadriga_lib.hdf5_write_channel(fn2, mixed);
chansR = quadriga_lib.hdf5_read_channel(fn2);
assertTrue(isempty(chansR(1).fbs_pos));
assertEqual(chansR(2).fbs_pos, double(single(mixed(2).fbs_pos)));
delete(fn2);
quadriga_lib.hdf5_write_channel(fn2, chansR);   % the [] fields write back as absent members
chansR = quadriga_lib.hdf5_read_channel(fn2);
assertTrue(isempty(chansR(1).fbs_pos));
assertEqual(chansR(2).fbs_pos, double(single(mixed(2).fbs_pos)));

% A new file is sized from the indices; [] is the default; non-double classes; no output requested
delete(fn2);
quadriga_lib.hdf5_write_channel(fn2, chans(1:2), [], [], uint32(2), [1, 3], single(2));
[storage_space, has_data] = quadriga_lib.hdf5_read_layout(fn2);
assertEqual(storage_space, uint32([2, 2, 3, 2]));
assertTrue(has_data(1, 2, 1, 2) && has_data(2, 2, 3, 2));
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn2, chans(1), [], 3), 'Entry 1: Slot is outside the storage layout.' );
assertEqual(nnz(has_data), 2);

% Scalar broadcasting on iy
delete(fn2);
quadriga_lib.hdf5_create_file(fn2, 5, 5);
quadriga_lib.hdf5_write_channel(fn2, chans, [], [1, 2, 3], 4);  % (1,4),(2,4),(3,4)
chansR = quadriga_lib.hdf5_read_channel(fn2, [1, 2, 3], 4);
assertEqual(numel(chansR), 3);
for k = 1:N
    assertEqual(chansR(k).name, chans(k).name);
end

% Index selection on read: given order is kept; [] is the full range; non-double classes
chansR = quadriga_lib.hdf5_read_channel(fn2, uint32([3, 1]), single(4));
assertEqual({chansR.name}, {chans(3).name, chans(1).name});
chansR = quadriga_lib.hdf5_read_channel(fn2, [], [], [], []);
assertEqual(numel(chansR), 3);

% Empty selection: both outputs are 0x0 structs; no output requested
[chansR, parR] = quadriga_lib.hdf5_read_channel(fn2, 4, 4);
assertTrue(isstruct(chansR) && isstruct(parR));
assertEqual(size(chansR), [0, 0]);
assertEqual(size(parR), [0, 0]);
quadriga_lib.hdf5_read_channel(fn2, 1, 4);

% par struct array with sparse field set per element
delete(fn2);
par_arr      = struct();
par_arr(1).a = 11;
par_arr(2).b = 'hello';
par_arr(3).a = 33;
par_arr(3).c = uint32(7);
quadriga_lib.hdf5_write_channel(fn2, chans, par_arr);
[chansR, parR] = quadriga_lib.hdf5_read_channel(fn2);
assertEqual(numel(chansR), N);
assertEqual(numel(parR), N);
% Field union: a, b, c
fn_par = sort(fieldnames(parR));
assertEqual(fn_par, sort({'a';'b';'c'}));
% Per-element population
assertEqual(parR(1).a, 11);
assertTrue(isempty(parR(1).b));
assertTrue(isempty(parR(1).c));
assertTrue(isempty(parR(2).a));
assertEqual(parR(2).b, 'hello');
assertEqual(parR(3).a, 33);
assertEqual(parR(3).c, uint32(7));

% par size must match chan size
par_short = struct('a', {1, 2});  % length 2 vs chans length 3
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn2, chans, par_short), 'must have the same number of entries' );

% Index vector length must match chan length (when not scalar)
assert_cpp_error( @() quadriga_lib.hdf5_write_channel(fn2, chans, [], [1, 2]), 'scalar or a vector of length' );  % 2 indices, 3 chans

% Empty slots are skipped on read
delete(fn2);
quadriga_lib.hdf5_create_file(fn2, 10, 1);
quadriga_lib.hdf5_write_channel(fn2, chans(1:2), [], [3, 7]);  % only slots 3 and 7 populated
chansR = quadriga_lib.hdf5_read_channel(fn2);   % default scans 1:10
assertEqual(numel(chansR), 2);
assertEqual(chansR(1).name, chans(1).name);     % column-major order: slot 3 first
assertEqual(chansR(2).name, chans(2).name);

% i_snap applies to every selected slot
chansR = quadriga_lib.hdf5_read_channel(fn2, [], [], [], [], 2);
assertEqual(numel(chansR), 2);
for k = 1:2
    assertEqual(chansR(k).coeff_re, double(single(chans(k).coeff_re(:, :, :, 2))));
end

%% hdf5_read_channel: errors

% An i_snap index beyond the snapshots of a selected slot (thrown by C++)
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn2, [], [], [], [], 3), 'exceed the number of snapshots' );

% Reading a non-existent file errors
assert_cpp_error( @() quadriga_lib.hdf5_read_channel('does_not_exist.hdf5'), 'does not exist' );

% Index out of bounds or not an integer on read
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn2, 999), 'outside the storage layout (10 slots)' );   % thrown by C++
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn2, 0), 'Entries in ''ix'' cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn2, 1, 1, 1, 2), 'Input ''iw'' contains indices outside' );
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn2, 1.5), 'Entries in ''ix'' must be integers from 1 to 4294967295.' );

% Argument counts
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.hdf5_read_channel(fn2, 1, 1, 1, 1, 1, 1), 'Wrong number of input arguments.' );
assert_cpp_error( @() out3(@quadriga_lib.hdf5_read_channel, fn2), 'Wrong number of output arguments.' );

%% Datasets with more dimensions than supported, sparse data
quadriga_lib.hdf5_create_file(fn3, 2, 2);
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn3, 'd4', rand(2,3,4,2), 1, 1), ...
    'Input ''data'' has more than 3 dimensions.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn3, 'd5', rand(2,3,4,2,2), 1, 1), ...
    'Input ''data'' has more than 3 dimensions.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn3, 'sp', sparse([0 0 1.5]), 1, 1), ...
    'Input ''data'' is sparse; sparse inputs are not supported.' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn3, 'x', 1, 0, 1), 'cannot be 0 (1-based index).' );
assert_cpp_error( @() quadriga_lib.hdf5_write_dset(fn3, 'x', 1, 1.5, 1), 'Input ''ix'' must be an integer from 1 to 4294967295.' );
quadriga_lib.hdf5_write_dset(fn3, 'd3', reshape(1:24, 2, 3, 4), 1, 1);     % 3-D is supported
assertEqual( quadriga_lib.hdf5_read_dset(fn3, 'd3', 1, 1), reshape(1:24, 2, 3, 4) );
assertEqual( numel(quadriga_lib.hdf5_read_dset_names(fn3, 1, 1)), 1 );   % nothing else was written

%% Cleanup
delete_files( fn, fn2, fn3, non_hdf5 );

end

% Request 1, 2 or 3 outputs so that the nargout guard of a wrapper fires
function out1(f, varargin)
x = f( varargin{:} ); %#ok<NASGU>
end

function out2(f, varargin)
[~, ~] = f( varargin{:} );
end

function out3(f, varargin)
[~, ~, ~] = f( varargin{:} );
end

function delete_files(varargin)
for n = 1:numel(varargin)
    if exist(varargin{n}, 'file')
        delete(varargin{n});
    end
end
end

% Assert that fn() raises quadriga_lib:CPPerror with a message that contains expected (substring).
function assert_cpp_error(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, 'quadriga_lib:CPPerror') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: quadriga_lib:CPPerror "', expected, '", GOT: ', ...
            ME.identifier, ' "', ME.message, '"']);
    end
end
end
