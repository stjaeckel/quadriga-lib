function test_hdf_rw

verbose = 0;
run_tests = inf;

% MOxUnit Setup
tmp = which('MOxUnitTestSuite');
if isempty(tmp)
    current_dir = pwd;
    cd('../external/MOxUnit-master/MOxUnit');
    moxunit_set_path();
    cd(current_dir);
end

fn = 'hdf_mex.hdf5';

% Clean slate
if exist(fn, 'file')
    delete(fn);
end

%% ===== File creation and layout =====

% Create file with default layout - returns 1x4 uint32 storage_space
tst = 0; if tst > run_tests; return; end
if verbose; disp("Test: Create file (default layout)"); end
storage_space = quadriga_lib.hdf5_create_file(fn);
assertEqual(storage_space, uint32([65536, 1, 1, 1]));

% Read layout back
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Read layout"); end
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([65536, 1, 1, 1]));

% Recreating existing file should error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Recreate file errors"); end
try
    quadriga_lib.hdf5_create_file(fn);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'File already exists.')));
end

% Custom storage layout
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Create file (custom layout)"); end
delete(fn);
quadriga_lib.hdf5_create_file(fn, [12, 12]);
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([12, 12, 1, 1]));

% Zero-dimension rejection
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Zero-dim rejected"); end
delete(fn);
try
    quadriga_lib.hdf5_create_file(fn, [0, 1]);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'cannot contain zeros')));
end

% Reshape layout (preserve total slot count)
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reshape layout"); end
quadriga_lib.hdf5_create_file(fn, [12, 12]);
s = quadriga_lib.hdf5_reshape_layout(fn, [1, 1, 18, 8]);
assertEqual(s, uint32([1, 1, 18, 8]));
storage_space = quadriga_lib.hdf5_read_layout(fn);
assertEqual(storage_space, uint32([1, 1, 18, 8]));

% Reshape mismatched element count
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reshape mismatch errors"); end
try
    quadriga_lib.hdf5_reshape_layout(fn, 145);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'Mismatch in number of elements')));
end

delete(fn);

% Reshape on missing file
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reshape non-existent file errors"); end
try
    quadriga_lib.hdf5_reshape_layout(fn, 144);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'File does not exist.')));
end

%% ===== HDF5 version =====

tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: HDF5 version"); end
v = quadriga_lib.hdf5_version;
assertTrue(ischar(v) && ~isempty(regexp(v, '^\d+\.\d+\.\d+', 'once')));

%% ===== Unstructured data (par) round-trip =====

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

% Write par-only via hdf5_write_channel: creates file with default-derived layout
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Write par-only"); end
storage_space = quadriga_lib.hdf5_write_channel(fn, [1, 1, 1, 1], par, []);
assertEqual(storage_space, uint32([128, 8, 8, 8]));

% Too many outputs
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Too many outputs errors"); end
try
    [~, ~] = quadriga_lib.hdf5_write_channel(fn, [2, 1, 1, 1], par, []);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'Wrong number of output arguments')));
end

% Writing entirely empty (no par, no chan) should be accepted
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Write empty par and chan"); end
quadriga_lib.hdf5_write_channel(fn, [2, 1, 1, 1], [], []);

% Round-trip par via hdf5_read_channel
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Read par round-trip"); end
[parR, chanR] = quadriga_lib.hdf5_read_channel(fn);
assertTrue(~isfield(chanR, 'coeff_re'));    % no structured data at slot [1,1,1,1]
assertTrue(~isfield(chanR, 'rx_position'));

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
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Dataset names"); end
par_names = quadriga_lib.hdf5_read_dset_names(fn);
assertEqual(length(fieldsPar), length(par_names));
assertTrue(all(strcmp(fieldsPar, par_names)));

% Per-field read/write_dset round-trip
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Per-field read/write_dset"); end
for n = 1:length(fieldsPar)
    f = fieldsPar{n};
    data = quadriga_lib.hdf5_read_dset(fn, 1, f);
    assertTrue(isequal(par.(f), data));
    quadriga_lib.hdf5_write_dset(fn, [1, 2], f, data);
end

par_names = quadriga_lib.hdf5_read_dset_names(fn, [1, 2]);
assertEqual(numel(par_names), length(fieldsPar));

% Overwriting a dset should error
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Overwriting dset errors"); end
try
    quadriga_lib.hdf5_write_dset(fn, [1, 2], 'string', 'Oh no, I bought Ethereum.');
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'already exists')));
end

% Names of an empty location
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Empty location names"); end
par_names = quadriga_lib.hdf5_read_dset_names(fn, [12, 2, 2]);
assertTrue(isempty(par_names));

% Reading a missing dataset returns []
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Missing dataset returns []"); end
missing = quadriga_lib.hdf5_read_dset(fn, 1, 'does_not_exist');
assertTrue(isempty(missing));

% Custom prefix on dset functions
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Custom dset prefix"); end
quadriga_lib.hdf5_write_dset(fn, [1, 3], 'foo', pi, 'custom_');
% Default prefix lookup must not see it
names_default = quadriga_lib.hdf5_read_dset_names(fn, [1, 3]);
assertTrue(~any(strcmp(names_default, 'foo')));
% Custom prefix lookup must see it
names_custom = quadriga_lib.hdf5_read_dset_names(fn, [1, 3], 'custom_');
assertTrue(any(strcmp(names_custom, 'foo')));
val = quadriga_lib.hdf5_read_dset(fn, [1, 3], 'foo', 'custom_');
assertEqual(val, pi);

% Complex data is rejected
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Complex data rejected"); end
parC.complex = 1 + 21i;
try
    quadriga_lib.hdf5_write_channel(fn, [2, 1, 1, 1], parC, []);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~strcmp(ME.identifier, 'moxunit:exceptionNotRaised'));
end

%% ===== Structured (chan) round-trip =====
% File stores structured data in single precision; wrapper reads back as double.
% So expected values for round-trip are double(single(input)).

rx_pos   = rand(3, 1);
tx_pos   = rand(3, 1);
coeff_re = rand(3, 2, 5, 3);
coeff_im = rand(3, 2, 5, 3);
delay_4d = rand(3, 2, 5, 3);

% Minimal chan: positions, coefficients, delay
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Minimal chan write"); end
chan = struct;
chan.rx_position = rx_pos;
chan.tx_position = tx_pos;
chan.coeff_re    = coeff_re;
chan.coeff_im    = coeff_im;
chan.delay       = delay_4d;
quadriga_lib.hdf5_write_channel(fn, [3, 1, 1, 1], [], chan);

% Restore and compare
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Restore minimal chan"); end
[~, chanR] = quadriga_lib.hdf5_read_channel(fn, 3);
assertEqual(chanR.rx_position, double(single(rx_pos)));
assertEqual(chanR.tx_position, double(single(tx_pos)));
assertEqual(chanR.coeff_re,    double(single(coeff_re)));
assertEqual(chanR.coeff_im,    double(single(coeff_im)));
assertEqual(chanR.delay,       double(single(delay_4d)));

% Reverse snapshot order via snap argument
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Reverse snapshot order"); end
[~, chanR] = quadriga_lib.hdf5_read_channel(fn, 3, [3, 2, 1]);
assertEqual(chanR.coeff_re, double(single(coeff_re(:, :, :, [3, 2, 1]))));
assertEqual(chanR.coeff_im, double(single(coeff_im(:, :, :, [3, 2, 1]))));
assertEqual(chanR.delay,    double(single(delay_4d(:, :, :, [3, 2, 1]))));

% Snapshot index out of bounds
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Snapshot out of bounds"); end
try
    quadriga_lib.hdf5_read_channel(fn, 3, 0);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'out of bounds')));
end
try
    quadriga_lib.hdf5_read_channel(fn, 3, 4);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'out of bounds')));
end

% Snap on slot with no structured data now errors (was silent in old API)
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Snap on par-only slot errors"); end
try
    quadriga_lib.hdf5_read_channel(fn, 1, 2);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'out of bounds')));
end

% 2D-delay shorthand was removed; explicit reshape works
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: 2D-shape delay via explicit reshape"); end
delay_2d = rand(5, 3);
chan = struct;
chan.rx_position = rx_pos;
chan.tx_position = tx_pos;
chan.coeff_re    = coeff_re;
chan.coeff_im    = coeff_im;
chan.delay       = reshape(delay_2d, [1, 1, size(delay_2d, 1), size(delay_2d, 2)]);
quadriga_lib.hdf5_write_channel(fn, [4, 1, 1, 1], [], chan);
[~, chanR] = quadriga_lib.hdf5_read_channel(fn, 4);
assertEqual(squeeze(chanR.delay), double(single(delay_2d)));

% Full chan: every structured field populated
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Full chan write"); end
center_frequency  = 21e9;
name              = 'buy_more_bitcoin';
path_gain         = rand(5, 3);
path_length       = rand(5, 3);
path_polarization = rand(8, 5, 3);
path_angles       = rand(5, 4, 3);
fbs_pos           = rand(3, 5, 3);
lbs_pos           = rand(3, 5, 3);
no_interact       = [1 2 3 4 5 ; 5 4 3 2 1 ; 1 1 1 1 1]';
interact_coord    = rand(3, 15, 3);
interact_coord(:, 6:end, 3) = 0;   % only 5 interactions in snap 3
rx_orientation    = rand(3, 3);
tx_orientation    = rand(3, 3);

chan = struct;
chan.name              = name;
chan.rx_position       = rx_pos;
chan.tx_position       = tx_pos;
chan.coeff_re          = coeff_re;
chan.coeff_im          = coeff_im;
chan.delay             = delay_4d;
chan.center_frequency  = center_frequency;
chan.path_gain         = path_gain;
chan.path_length       = path_length;
chan.path_polarization = path_polarization;
chan.path_angles       = path_angles;
chan.fbs_pos           = fbs_pos;
chan.lbs_pos           = lbs_pos;
chan.no_interact       = no_interact;
chan.interact_coord    = interact_coord;
chan.rx_orientation    = rx_orientation;
chan.tx_orientation    = tx_orientation;
quadriga_lib.hdf5_write_channel(fn, [5, 1, 1, 1], [], chan);

% Read back full chan and verify every field
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Full chan round-trip"); end
[parR, chanR] = quadriga_lib.hdf5_read_channel(fn, 5);
assertTrue(isempty(fieldnames(parR)));
assertEqual(chanR.name, name);
assertEqual(chanR.rx_position,       double(single(rx_pos)));
assertEqual(chanR.tx_position,       double(single(tx_pos)));
assertEqual(chanR.coeff_re,          double(single(coeff_re)));
assertEqual(chanR.coeff_im,          double(single(coeff_im)));
assertEqual(chanR.delay,             double(single(delay_4d)));
assertEqual(chanR.center_frequency,  double(single(center_frequency)));
assertEqual(chanR.path_gain,         double(single(path_gain)));
assertEqual(chanR.path_length,       double(single(path_length)));
assertEqual(chanR.path_polarization, double(single(path_polarization)));
assertEqual(chanR.path_angles,       double(single(path_angles)));
assertEqual(chanR.fbs_pos,           double(single(fbs_pos)));
assertEqual(chanR.lbs_pos,           double(single(lbs_pos)));
assertEqual(chanR.no_interact,       uint32(no_interact));
assertEqual(chanR.interact_coord,    double(single(interact_coord)));
assertEqual(chanR.rx_orientation,    double(single(rx_orientation)));
assertEqual(chanR.tx_orientation,    double(single(tx_orientation)));

% Single-snapshot subset
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Single-snapshot subset"); end
[~, chanR] = quadriga_lib.hdf5_read_channel(fn, 5, 3);
assertEqual(size(chanR.coeff_re, 4), 1);
assertEqual(chanR.coeff_re,  double(single(coeff_re(:, :, :, 3))));
assertEqual(chanR.path_gain, double(single(path_gain(:, 3))));

% initial_position round-trip
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: initial_position round-trip"); end
chan_ip = chan;
chan_ip.initial_position = int32(7);
quadriga_lib.hdf5_write_channel(fn, [6, 1, 1, 1], [], chan_ip);
[~, chanR] = quadriga_lib.hdf5_read_channel(fn, [6, 1, 1, 1]);
assertEqual(chanR.initial_position, int32(7));

% Combined par + chan write with sparse field set
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Combined par + chan, sparse fields"); end
rx_pos = rand(3, 3);
tx_pos = rand(3, 3);
center_frequency = rand(3, 1);
chan = struct;
chan.name              = name;
chan.initial_position  = int32(2);
chan.rx_position       = rx_pos;
chan.tx_position       = tx_pos;
chan.coeff_re          = coeff_re;
chan.coeff_im          = coeff_im;
chan.delay             = delay_4d;
chan.center_frequency  = center_frequency;
chan.rx_orientation    = rx_orientation(:, 1);
chan.tx_orientation    = tx_orientation(:, 2);
quadriga_lib.hdf5_write_channel(fn, [6, 1, 1, 2], par, chan);

[parR, chanR] = quadriga_lib.hdf5_read_channel(fn, [6, 1, 1, 2]);
assertTrue(~isempty(fieldnames(parR)));
assertEqual(chanR.rx_position,       double(single(rx_pos)));
assertEqual(chanR.tx_position,       double(single(tx_pos)));
assertEqual(chanR.coeff_re,          double(single(coeff_re)));
assertEqual(chanR.coeff_im,          double(single(coeff_im)));
assertEqual(chanR.delay,             double(single(delay_4d)));
assertEqual(chanR.center_frequency,  double(single(center_frequency)));
assertEqual(chanR.name,              name);
assertEqual(chanR.initial_position,  int32(2));
assertEqual(chanR.rx_orientation,    double(single(rx_orientation(:, 1))));
assertEqual(chanR.tx_orientation,    double(single(tx_orientation(:, 2))));
% Fields not written are absent from the result struct
assertTrue(~isfield(chanR, 'path_gain'));
assertTrue(~isfield(chanR, 'path_length'));
assertTrue(~isfield(chanR, 'path_polarization'));
assertTrue(~isfield(chanR, 'path_angles'));
assertTrue(~isfield(chanR, 'fbs_pos'));
assertTrue(~isfield(chanR, 'lbs_pos'));
assertTrue(~isfield(chanR, 'no_interact'));
assertTrue(~isfield(chanR, 'interact_coord'));

%% ===== Layout / has_data inspection =====

% has_data is logical (was uint32 in old API)
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: has_data is logical"); end
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

% Non-existent file: returns [0,0,0,0] and empty has_data
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Read layout of non-existent file"); end
[storage_space, has_data] = quadriga_lib.hdf5_read_layout('does_not_exist.hdf5');
assertEqual(storage_space, uint32([0, 0, 0, 0]));
assertTrue(isempty(has_data));

% Reading a non-HDF5 file errors
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Non-HDF5 file errors"); end
non_hdf5 = 'not_hdf5.txt';
fid = fopen(non_hdf5, 'w');
fprintf(fid, 'This is not HDF5 data');
fclose(fid);
try
    quadriga_lib.hdf5_read_layout(non_hdf5);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'Not an HDF5 file')));
end
delete(non_hdf5);

%% ===== Wrapper-level input validation =====

% chan as 2x1 struct array is rejected
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: chan struct array rejected"); end
chan2 = repmat(chan, 2, 1);
try
    quadriga_lib.hdf5_write_channel(fn, [7, 1, 1, 1], [], chan2);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, '1x1 struct')));
end

% Non-struct par is rejected
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Non-struct par rejected"); end
try
    quadriga_lib.hdf5_write_channel(fn, [7, 1, 1, 1], [1, 2, 3], chan);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'must be a struct')));
end

% Non-struct chan is rejected
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Non-struct chan rejected"); end
try
    quadriga_lib.hdf5_write_channel(fn, [7, 1, 1, 1], [], [1, 2, 3]);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'must be a struct')));
end

% Zero in location is rejected
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: Zero location rejected"); end
try
    quadriga_lib.hdf5_write_channel(fn, [0, 1, 1, 1], [], chan);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'cannot contain zeros')));
end

% hdf5_write_dset on non-existent file errors
tst = tst + 1; if tst > run_tests; return; end
if verbose; disp("Test: write_dset on missing file errors"); end
try
    quadriga_lib.hdf5_write_dset('does_not_exist.hdf5', 1, 'foo', pi);
    error('moxunit:exceptionNotRaised','Expected an error!');
catch ME
    assertTrue(~isempty(strfind(ME.message, 'File does not exist')));
end

%% ===== Final cleanup =====

if exist(fn, 'file')
    delete(fn);
end

end