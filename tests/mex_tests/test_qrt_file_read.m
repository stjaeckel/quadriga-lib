% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_qrt_file_read()
% MEX interface test for quadriga_lib.qrt_file_parse and quadriga_lib.qrt_file_read.

fn = 'data/test.qrt';

%% qrt_file_parse: v4 file
[n_cir, n_orig, n_dest, n_freq, cir_offset, orig_names, dest_names, version, ...
    center_freq, cir_pos, cir_orientation, orig_pos, orig_orientation] = ...
    quadriga_lib.qrt_file_parse(fn);

assertEqual(n_orig, uint64(3));
assertEqual(n_dest, uint64(2));
assertEqual(n_cir, uint64(7));
assertEqual(n_freq, uint64(1));
assertEqual(version, int32(4));
assertEqual(numel(cir_offset), double(n_dest));

assertEqual(cir_offset(1), uint64(0));
assertEqual(cir_offset(2), uint64(1));

assertEqual(numel(orig_names), 3);
assertEqual(numel(dest_names), 2);

assertEqual(orig_names{1}, 'TX1');
assertEqual(orig_names{2}, 'TX2');
assertEqual(orig_names{3}, 'TX3');

assertEqual(dest_names{1}, 'RX1');
assertEqual(dest_names{2}, 'RX2');

% Frequency
assertEqual(numel(center_freq), double(n_freq));
assertElementsAlmostEqual(center_freq(1), single(3.75e9), 'relative', 1e-6);

% CIR positions and orientations
assertEqual(size(cir_pos), [double(n_cir), 3]);
assertEqual(size(cir_orientation), [double(n_cir), 3]);

T = single([-8.83498, 57.1893, 1.0]);
assertElementsAlmostEqual(cir_pos(1, :), T, 'absolute', 1.5e-4);

T = single([-5.86144, 53.8124, 1.0]);
assertElementsAlmostEqual(cir_pos(2, :), T, 'absolute', 1.5e-4);

T = single([0, 0, 1.2753]);
assertElementsAlmostEqual(cir_orientation(2, :), T, 'absolute', 1.5e-4);

% Origin (TX) positions and orientations
assertEqual(size(orig_pos), [double(n_orig), 3]);
assertEqual(size(orig_orientation), [double(n_orig), 3]);

T = single([-12.9607, 59.6906, 2.0]);
assertElementsAlmostEqual(orig_pos(1, :), T, 'absolute', 1.5e-4);

T = single([-2.67888, 60.257, 2.0]);
assertElementsAlmostEqual(orig_pos(2, :), T, 'absolute', 1.5e-4);

T = single([0, 0, 0]);
assertElementsAlmostEqual(orig_orientation(1, :), T, 'absolute', 1.5e-4);

T = single([0, 0, pi]);
assertElementsAlmostEqual(orig_orientation(2, :), T, 'absolute', 1.5e-4);

%% qrt_file_read: first link (i_cir=1, i_orig=1, downlink)
[center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
    path_gain, path_length, M, aod, eod, aoa, eoa, path_coord] = ...
    quadriga_lib.qrt_file_read(fn, 1, 1, false, 1);

assertElementsAlmostEqual(center_freq(1), 3.75e9, 'relative', 1e-7); % v4 stores GHz, converted to Hz in float

T = [-12.9607; 59.6906; 2.0];
assertElementsAlmostEqual(tx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; 0];
assertElementsAlmostEqual(tx_orientation, T, 'absolute', 1.5e-4);

T = [-8.83498; 57.1893; 1.0];
assertElementsAlmostEqual(rx_pos, T, 'absolute', 1.5e-4);

%% qrt_file_read: second link (i_cir=2, i_orig=2, downlink)
[center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
    path_gain, path_length, M, aod, eod, aoa, eoa, path_coord] = ...
    quadriga_lib.qrt_file_read(fn, 2, 2, false, 1);

T = [-2.67888; 60.257; 2.0];
assertElementsAlmostEqual(tx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; pi];
assertElementsAlmostEqual(tx_orientation, T, 'absolute', 1.5e-4);

T = [-5.86144; 53.8124; 1.0];
assertElementsAlmostEqual(rx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; 1.2753];
assertElementsAlmostEqual(rx_orientation, T, 'absolute', 1.5e-4);

%% qrt_file_read: uplink (i_cir=2, i_orig=2, uplink=true)
% aoa, eoa still hold values from the downlink call above
aoa_dl = aoa{1};
eoa_dl = eoa{1};
fbs_dl = fbs_pos{1};
lbs_dl = lbs_pos{1};
M_dl = M{1};
path_gain_dl = path_gain{1};

% Uplink call: request outputs up to eod (12 outputs)
[~, tx_pos_ul, tx_ori_ul, rx_pos_ul, rx_ori_ul, fbs_ul, lbs_ul, ...
    path_gain_ul, ~, M_ul, aod_ul, eod_ul] = ...
    quadriga_lib.qrt_file_read(fn, 2, 2, true, 1);

T = [-5.86144; 53.8124; 1.0];
assertElementsAlmostEqual(tx_pos_ul, T, 'absolute', 1.5e-4);

T = [0; 0; 1.2753];
assertElementsAlmostEqual(tx_ori_ul, T, 'absolute', 1.5e-4);

T = [-2.67888; 60.257; 2.0];
assertElementsAlmostEqual(rx_pos_ul, T, 'absolute', 1.5e-4);

T = [0; 0; pi];
assertElementsAlmostEqual(rx_ori_ul, T, 'absolute', 1.5e-4);

% fbs/lbs swapped between uplink and downlink
assertElementsAlmostEqual(fbs_ul{1}, lbs_dl, 'absolute', 1.5e-4);
assertElementsAlmostEqual(lbs_ul{1}, fbs_dl, 'absolute', 1.5e-4);

% uplink aod/eod match downlink aoa/eoa
assertElementsAlmostEqual(aod_ul{1}, aoa_dl, 'absolute', 1.5e-4);
assertElementsAlmostEqual(eod_ul{1}, eoa_dl, 'absolute', 1.5e-4);

% Reciprocity: uplink M is the plain transpose of the downlink M (rows HV and VH swap),
% without conjugation; the path gain is the same in both directions
assertElementsAlmostEqual(M_ul{1}, M_dl([1 2 5 6 3 4 7 8], :, :), 'absolute', 1e-12);
assertElementsAlmostEqual(path_gain_ul{1}, path_gain_dl, 'relative', 1e-12);

%% qrt_file_parse: v5 file
fn5 = 'data/test_v5.qrt';

[n_cir5, n_orig5, n_dest5, n_freq5, cir_offset5, ~, ~, version5, ...
    center_freq5, cir_pos5, cir_orientation5, orig_pos5, orig_orientation5] = ...
    quadriga_lib.qrt_file_parse(fn5);

assertEqual(version5, int32(5));
assertEqual(n_freq5, uint64(2));

% Frequencies
assertEqual(numel(center_freq5), double(n_freq5));
assertElementsAlmostEqual(center_freq5(1), single(1.0e9), 'relative', 1e-6);
assertElementsAlmostEqual(center_freq5(2), single(1.5e9), 'relative', 1e-6);

% Shape checks
assertEqual(size(cir_pos5), [double(n_cir5), 3]);
assertEqual(size(cir_orientation5), [double(n_cir5), 3]);
assertEqual(size(orig_pos5), [double(n_orig5), 3]);
assertEqual(size(orig_orientation5), [double(n_orig5), 3]);

%% qrt_file_read: v5, normalize_polmat=0
[center_freq5, ~, ~, ~, ~, ~, ~, path_gain5, ~, M5] = ...
    quadriga_lib.qrt_file_read(fn5, 2, 1, false, 0);

M5 = M5{1};
path_gain5 = path_gain5{1};

assertEqual(numel(center_freq5), 2);
assertElementsAlmostEqual(center_freq5(1), 1.0e9, 'absolute', 1e-4);
assertElementsAlmostEqual(center_freq5(2), 1.5e9, 'absolute', 1e-4);

assertEqual(size(M5, 1), 8);
assertEqual(size(M5, 3), 2);
assertEqual(size(path_gain5, 2), 2);

T = [0.1131; 0; 0; 0; 0; 0; -0.1131; 0];
assertElementsAlmostEqual(M5(:, 1, 1), T, 'absolute', 1.5e-4);

T = [0.0866; 0; 0; 0; 0; 0; -0.0866; 0];
assertElementsAlmostEqual(M5(:, 1, 2), T, 'absolute', 1.5e-4);

%% qrt_file_read: v5, normalize_polmat=1 (default)
[~, ~, ~, ~, ~, ~, ~, ~, ~, M5n] = ...
    quadriga_lib.qrt_file_read(fn5, 2, 1, false, 1);

M5n = M5n{1};

T = [1; 0; 0; 0; 0; 0; -1; 0];
assertElementsAlmostEqual(M5n(:, 1, 2), T, 'absolute', 1.5e-4);


%% qrt_file_read: multiple CIRs in one call
[~, tx_pos_m, tx_ori_m, rx_pos_m, rx_ori_m, fbs_m, lbs_m, ...
    pg_m, plen_m, M_m, aod_m, eod_m, aoa_m, eoa_m, pc_m, n_interact_m, coord_m] = ...
    quadriga_lib.qrt_file_read(fn, [1, 2, 3], 1, false, 1);

% Scalar-per-CIR outputs: [3, n_out] matrices
assertEqual(size(tx_pos_m), [3, 3]);
assertEqual(size(tx_ori_m), [3, 3]);
assertEqual(size(rx_pos_m), [3, 3]);
assertEqual(size(rx_ori_m), [3, 3]);

% Variable-per-CIR outputs: cells of length n_out
assertEqual(numel(fbs_m),    3);
assertEqual(numel(lbs_m),    3);
assertEqual(numel(pg_m),     3);
assertEqual(numel(plen_m),   3);
assertEqual(numel(M_m),      3);
assertEqual(numel(aod_m),    3);
assertEqual(numel(eod_m),    3);
assertEqual(numel(aoa_m),    3);
assertEqual(numel(eoa_m),    3);
assertEqual(numel(pc_m),     3);
assertEqual(numel(n_interact_m), 3);
assertEqual(numel(coord_m),  3);

% Cross-check vs individual single-CIR reads for i_cir = 1 and 2
[~, tx1, ~, rx1, ~, fbs1, ~, pg1, plen1, M1, aod1] = ...
    quadriga_lib.qrt_file_read(fn, 1, 1, false, 1);
[~, tx2, ~, rx2, ~, fbs2, ~, pg2, plen2, M2, aod2] = ...
    quadriga_lib.qrt_file_read(fn, 2, 1, false, 1);

assertElementsAlmostEqual(tx_pos_m(:, 1), tx1, 'absolute', 1e-6);
assertElementsAlmostEqual(tx_pos_m(:, 2), tx2, 'absolute', 1e-6);
assertElementsAlmostEqual(rx_pos_m(:, 1), rx1, 'absolute', 1e-6);
assertElementsAlmostEqual(rx_pos_m(:, 2), rx2, 'absolute', 1e-6);

assertElementsAlmostEqual(fbs_m{1},  fbs1{1},  'absolute', 1e-6);
assertElementsAlmostEqual(fbs_m{2},  fbs2{1},  'absolute', 1e-6);
assertElementsAlmostEqual(pg_m{1},   pg1{1},   'absolute', 1e-6);
assertElementsAlmostEqual(pg_m{2},   pg2{1},   'absolute', 1e-6);
assertElementsAlmostEqual(plen_m{1}, plen1{1}, 'absolute', 1e-6);
assertElementsAlmostEqual(plen_m{2}, plen2{1}, 'absolute', 1e-6);
assertElementsAlmostEqual(M_m{1},    M1{1},    'absolute', 1e-6);
assertElementsAlmostEqual(M_m{2},    M2{1},    'absolute', 1e-6);
assertElementsAlmostEqual(aod_m{1},  aod1{1},  'absolute', 1e-6);
assertElementsAlmostEqual(aod_m{2},  aod2{1},  'absolute', 1e-6);

% path_coord is cell-of-cells
assertTrue(iscell(pc_m{1}));
assertEqual(numel(pc_m{1}), numel(aod_m{1}));   % one entry per path

%% qrt_file_read: default arguments
% Default i_cir reads ALL CIRs in the file (n_cir entries)
[~, tx_pos_d] = quadriga_lib.qrt_file_read(fn);
assertEqual(size(tx_pos_d), [3, double(n_cir)]);
assertElementsAlmostEqual(tx_pos_d(:, 1), tx1, 'absolute', 1e-6);

[~, tx_pos_e] = quadriga_lib.qrt_file_read(fn, []);
assertEqual(size(tx_pos_e), [3, double(n_cir)]);
assertElementsAlmostEqual(tx_pos_e(:, 1), tx1, 'absolute', 1e-6);

%% qrt_file_read: out-of-range i_cir errors
try
    quadriga_lib.qrt_file_read(fn, 999, 1, false, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''i_cir'' is out of range';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

%% qrt_file_read: zero (1-based) index errors
try
    quadriga_lib.qrt_file_read(fn, 0, 1, false, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Entries in ''i_cir'' cannot be 0 (1-based index).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

try
    quadriga_lib.qrt_file_read(fn, 1, 0, false, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''i_orig'' cannot be 0 (1-based index).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

%% qrt_file_read: uint64 i_cir is not modified by the call
i_cir = uint64([1 2]); keep = i_cir;
[~, ~, ~, ~, ~, ~, ~, pg_a] = quadriga_lib.qrt_file_read(fn, i_cir, 1, false, 1);
assertEqual(i_cir, uint64([1 2]));
assertEqual(keep, uint64([1 2]));
[~, ~, ~, ~, ~, ~, ~, pg_b] = quadriga_lib.qrt_file_read(fn, i_cir, 1, false, 1);
assertEqual(pg_b, pg_a);
[~, ~, ~, ~, ~, ~, ~, pg_c] = quadriga_lib.qrt_file_read(fn, [1 2], 1, false, 1);
assertEqual(pg_c, pg_a);

%% qrt_file_read: interact_type is uint8
out = cell(1, 18);
[out{:}] = quadriga_lib.qrt_file_read(fn5, 1);
assertTrue(isa(out{18}{1}, 'uint8'));

%% truncated file: parse and read throw
fid = fopen(fn5, 'r'); bytes = fread(fid, inf, 'uint8=>uint8'); fclose(fid);
fn_trunc = [tempname, '.qrt'];
fid = fopen(fn_trunc, 'w'); fwrite(fid, bytes(1:numel(bytes)-10), 'uint8'); fclose(fid);
expectedErrorMessage = 'truncated or corrupt';
try
    [~, ~, ~, ~, ~, c] = quadriga_lib.qrt_file_read(fn_trunc, 2, 1); % path data is read only for path outputs
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        delete(fn_trunc);
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end
try % a read error in the third chunk of the reader thread; the prefix names entry 41 (1-based)
    [~, ~, ~, ~, ~, c] = quadriga_lib.qrt_file_read(fn_trunc, [ones(1, 40), 2], 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage)) || ...
            isempty(strfind(ME.message, 'Snapshot 41: '))
        delete(fn_trunc);
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end
fid = fopen(fn_trunc, 'w'); fwrite(fid, bytes(1:40), 'uint8'); fclose(fid);
try
    c = quadriga_lib.qrt_file_parse(fn_trunc);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        delete(fn_trunc);
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end
delete(fn_trunc);

%% qrt_file_read: 40 CIRs are read in three chunks; each equals a single-CIR read
n_cir = quadriga_lib.qrt_file_parse(fn);
idx = mod(0:39, double(n_cir)) + 1;
out = cell(1, 18);
[out{:}] = quadriga_lib.qrt_file_read(fn, idx, 2, true);
assertEqual(size(out{2}), [3, 40]);
[~, tx_only, ~, rx_only] = quadriga_lib.qrt_file_read(fn, idx, 2, true); % positions only, no path data
assertEqual(tx_only, out{2});
assertEqual(rx_only, out{4});
for k = 1:40
    one = cell(1, 18);
    [one{:}] = quadriga_lib.qrt_file_read(fn, idx(k), 2, true);
    for i = 2:5
        assertEqual(out{i}(:, k), one{i});
    end
    for i = 6:18
        assertEqual(out{i}{k}, one{i}{1});
    end
end

%% qrt_file_read: a valid file with 0 CIRs gives empty outputs
% (written by qrt_file_init with cir_pos of size [0, 3] and one appended origin without paths)
fn0 = 'data/test_zero_cir.qrt';
out = cell(1, 18);
[out{:}] = quadriga_lib.qrt_file_read(fn0);
assertEqual(out{1}, 3e9);
for k = 2:5
    assertEqual(size(out{k}), [3, 0]);
end
for k = 6:18
    assertTrue(iscell(out{k}));
    assertEqual(size(out{k}), [0, 1]);
end
[cf0, tx0] = quadriga_lib.qrt_file_read(fn0, []);
assertEqual(size(tx0), [3, 0]);
failed = false;
try
    [~] = quadriga_lib.qrt_file_read(fn0, 1);
catch ME
    failed = ~isempty(strfind(ME.message, '''i_cir'' is out of range'));
end
assertTrue(failed);

%% qrt_file_read: a v4 file without origins stores no frequency
% (header "#QRT-BINv04" with n_orig = n_cir = n_dest = 0): one frequency of 0 Hz, empty outputs
fn_v4 = [tempname, '.qrt'];
fid = fopen(fn_v4, 'w'); fwrite(fid, [uint8('#QRT-BINv04'), zeros(1, 12, 'uint8')], 'uint8'); fclose(fid);
out = cell(1, 18);
[out{:}] = quadriga_lib.qrt_file_read(fn_v4);
delete(fn_v4);
assertEqual(out{1}, 0);
for k = 2:5
    assertEqual(size(out{k}), [3, 0]);
end
for k = 6:18
    assertTrue(iscell(out{k}));
    assertEqual(size(out{k}), [0, 1]);
end

%% v5 file storing Hz: same frequencies and path gains as the GHz file
% The frequency list starts at byte 27 (ID, n_orig, n_cir, n_dest, n_freq)
fn_hz = [tempname, '.qrt'];
copyfile(fn5, fn_hz);
fid = fopen(fn_hz, 'r+'); fseek(fid, 27, 'bof'); fwrite(fid, single([1.0e9, 1.5e9]), 'single'); fclose(fid);
[~, ~, ~, ~, ~, ~, ~, ~, center_freq_p] = quadriga_lib.qrt_file_parse(fn_hz);
[center_freq_hz, ~, ~, ~, ~, ~, ~, path_gain_hz] = quadriga_lib.qrt_file_read(fn_hz, 2, 1, false, 0);
delete(fn_hz);
assertElementsAlmostEqual(center_freq_p, single([1.0e9; 1.5e9]), 'relative', 1e-6);
assertElementsAlmostEqual(center_freq_hz, [1.0e9; 1.5e9], 'relative', 1e-6);
assertElementsAlmostEqual(path_gain_hz{1}, path_gain5, 'relative', 1e-12);

end
