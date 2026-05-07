function test_qrt_file_read()
% MOxUnit tests for quadriga_lib.qrt_file_parse and quadriga_lib.qrt_file_read

fn = 'data/test.qrt';

% ---------- qrt_file_parse: v4 file ----------
[no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version, ...
    fGHz, cir_pos, cir_orientation, orig_pos, orig_orientation] = ...
    quadriga_lib.qrt_file_parse(fn);

assertEqual(no_orig, uint64(3));
assertEqual(no_dest, uint64(2));
assertEqual(no_cir, uint64(7));
assertEqual(no_freq, uint64(1));
assertEqual(version, int32(4));
assertEqual(numel(cir_offset), double(no_dest));

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
assertEqual(numel(fGHz), double(no_freq));
assertElementsAlmostEqual(fGHz(1), single(3.75), 'absolute', 1e-6);

% CIR positions and orientations
assertEqual(size(cir_pos), [double(no_cir), 3]);
assertEqual(size(cir_orientation), [double(no_cir), 3]);

T = single([-8.83498, 57.1893, 1.0]);
assertElementsAlmostEqual(cir_pos(1, :), T, 'absolute', 1.5e-4);

T = single([-5.86144, 53.8124, 1.0]);
assertElementsAlmostEqual(cir_pos(2, :), T, 'absolute', 1.5e-4);

T = single([0, 0, 1.2753]);
assertElementsAlmostEqual(cir_orientation(2, :), T, 'absolute', 1.5e-4);

% Origin (TX) positions and orientations
assertEqual(size(orig_pos), [double(no_orig), 3]);
assertEqual(size(orig_orientation), [double(no_orig), 3]);

T = single([-12.9607, 59.6906, 2.0]);
assertElementsAlmostEqual(orig_pos(1, :), T, 'absolute', 1.5e-4);

T = single([-2.67888, 60.257, 2.0]);
assertElementsAlmostEqual(orig_pos(2, :), T, 'absolute', 1.5e-4);

T = single([0, 0, 0]);
assertElementsAlmostEqual(orig_orientation(1, :), T, 'absolute', 1.5e-4);

T = single([0, 0, pi]);
assertElementsAlmostEqual(orig_orientation(2, :), T, 'absolute', 1.5e-4);

% ---------- qrt_file_read: first link (i_cir=1, i_orig=1, downlink) ----------
[center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
    path_gain, path_length, M, aod, eod, aoa, eoa, path_coord] = ...
    quadriga_lib.qrt_file_read(fn, 1, 1, true, 1);

assertElementsAlmostEqual(center_freq(1), 3.75e9, 'absolute', 1e-4);

T = [-12.9607; 59.6906; 2.0];
assertElementsAlmostEqual(tx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; 0];
assertElementsAlmostEqual(tx_orientation, T, 'absolute', 1.5e-4);

T = [-8.83498; 57.1893; 1.0];
assertElementsAlmostEqual(rx_pos, T, 'absolute', 1.5e-4);

% ---------- qrt_file_read: second link (i_cir=2, i_orig=2, downlink) ----------
[center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
    path_gain, path_length, M, aod, eod, aoa, eoa, path_coord] = ...
    quadriga_lib.qrt_file_read(fn, 2, 2, true, 1);

T = [-2.67888; 60.257; 2.0];
assertElementsAlmostEqual(tx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; pi];
assertElementsAlmostEqual(tx_orientation, T, 'absolute', 1.5e-4);

T = [-5.86144; 53.8124; 1.0];
assertElementsAlmostEqual(rx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; 1.2753];
assertElementsAlmostEqual(rx_orientation, T, 'absolute', 1.5e-4);

% ---------- qrt_file_read: uplink (i_cir=2, i_orig=2, downlink=false) ----------
% aoa, eoa still hold values from the downlink call above
aoa_dl = aoa{1};
eoa_dl = eoa{1};
fbs_dl = fbs_pos{1};
lbs_dl = lbs_pos{1};

% Uplink call: only request up to eod (12 outputs), matching C++ test
[~, tx_pos_ul, tx_ori_ul, rx_pos_ul, rx_ori_ul, fbs_ul, lbs_ul, ...
    ~, ~, ~, aod_ul, eod_ul] = ...
    quadriga_lib.qrt_file_read(fn, 2, 2, false, 1);

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

% ---------- qrt_file_parse: v5 file ----------
fn5 = 'data/test_v5.qrt';

[no_cir5, no_orig5, no_dest5, no_freq5, cir_offset5, ~, ~, version5, ...
    fGHz5, cir_pos5, cir_orientation5, orig_pos5, orig_orientation5] = ...
    quadriga_lib.qrt_file_parse(fn5);

assertEqual(version5, int32(5));
assertEqual(no_freq5, uint64(2));

% Frequencies
assertEqual(numel(fGHz5), double(no_freq5));
assertElementsAlmostEqual(fGHz5(1), single(1.0), 'absolute', 1e-6);
assertElementsAlmostEqual(fGHz5(2), single(1.5), 'absolute', 1e-6);

% Shape checks
assertEqual(size(cir_pos5), [double(no_cir5), 3]);
assertEqual(size(cir_orientation5), [double(no_cir5), 3]);
assertEqual(size(orig_pos5), [double(no_orig5), 3]);
assertEqual(size(orig_orientation5), [double(no_orig5), 3]);

% ---------- qrt_file_read: v5, normalize_M=0 ----------
[center_freq5, ~, ~, ~, ~, ~, ~, path_gain5, ~, M5] = ...
    quadriga_lib.qrt_file_read(fn5, 2, 1, true, 0);

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

% ---------- qrt_file_read: v5, normalize_M=1 (default) ----------
[~, ~, ~, ~, ~, ~, ~, ~, ~, M5n] = ...
    quadriga_lib.qrt_file_read(fn5, 2, 1, true, 1);

M5n = M5n{1};

T = [1; 0; 0; 0; 0; 0; -1; 0];
assertElementsAlmostEqual(M5n(:, 1, 2), T, 'absolute', 1.5e-4);


% ---------- qrt_file_read: multiple CIRs in one call ----------
[~, tx_pos_m, tx_ori_m, rx_pos_m, rx_ori_m, fbs_m, lbs_m, ...
    pg_m, plen_m, M_m, aod_m, eod_m, aoa_m, eoa_m, pc_m, no_int_m, coord_m] = ...
    quadriga_lib.qrt_file_read(fn, [1, 2, 3], 1, true, 1);

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
assertEqual(numel(no_int_m), 3);
assertEqual(numel(coord_m),  3);

% Cross-check vs individual single-CIR reads for i_cir = 1 and 2
[~, tx1, ~, rx1, ~, fbs1, ~, pg1, plen1, M1, aod1] = ...
    quadriga_lib.qrt_file_read(fn, 1, 1, true, 1);
[~, tx2, ~, rx2, ~, fbs2, ~, pg2, plen2, M2, aod2] = ...
    quadriga_lib.qrt_file_read(fn, 2, 1, true, 1);

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

% ---------- qrt_file_read: default arguments ----------
[~, tx_pos_d] = quadriga_lib.qrt_file_read(fn);
assertEqual(size(tx_pos_d), [3, 1]);
assertElementsAlmostEqual(tx_pos_d, tx1, 'absolute', 1e-6);

[~, tx_pos_e] = quadriga_lib.qrt_file_read(fn, []);
assertEqual(size(tx_pos_e), [3, 1]);

% ---------- qrt_file_read: out-of-range i_cir errors ----------
try
    quadriga_lib.qrt_file_read(fn, 999, 1, true, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'CIR index exceeds number of CIRs in file.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

% ---------- qrt_file_read: zero (1-based) index errors ----------
try
    quadriga_lib.qrt_file_read(fn, 0, 1, true, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Entries in ''i_cir'' or ''i_orig'' must be >= 1';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

try
    quadriga_lib.qrt_file_read(fn, 1, 0, true, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Entries in ''i_cir'' or ''i_orig'' must be >= 1';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

end