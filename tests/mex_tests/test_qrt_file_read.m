function test_qrt_file_read()
% MOxUnit tests for quadriga_lib.qrt_file_parse and quadriga_lib.qrt_file_read

fn = 'data/test.qrt';

% ---------- qrt_file_parse: v4 file ----------
[no_cir, no_orig, no_dest, no_freq, cir_offset, orig_names, dest_names, version] = ...
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

% ---------- qrt_file_read: first link (i_cir=0, i_orig=0, downlink) ----------
[center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
    path_gain, path_length, M, aod, eod, aoa, eoa, path_coord] = ...
    quadriga_lib.qrt_file_read(fn, 0, 0, true, 1);

assertElementsAlmostEqual(center_freq(1), 3.75e9, 'absolute', 1e-4);

T = [-12.9607; 59.6906; 2.0];
assertElementsAlmostEqual(tx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; 0];
assertElementsAlmostEqual(tx_orientation, T, 'absolute', 1.5e-4);

T = [-8.83498; 57.1893; 1.0];
assertElementsAlmostEqual(rx_pos, T, 'absolute', 1.5e-4);

% ---------- qrt_file_read: second link (i_cir=1, i_orig=1, downlink) ----------
[center_freq, tx_pos, tx_orientation, rx_pos, rx_orientation, fbs_pos, lbs_pos, ...
    path_gain, path_length, M, aod, eod, aoa, eoa, path_coord] = ...
    quadriga_lib.qrt_file_read(fn, 1, 1, true, 1);

T = [-2.67888; 60.257; 2.0];
assertElementsAlmostEqual(tx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; pi];
assertElementsAlmostEqual(tx_orientation, T, 'absolute', 1.5e-4);

T = [-5.86144; 53.8124; 1.0];
assertElementsAlmostEqual(rx_pos, T, 'absolute', 1.5e-4);

T = [0; 0; 1.2753];
assertElementsAlmostEqual(rx_orientation, T, 'absolute', 1.5e-4);

% ---------- qrt_file_read: uplink (i_cir=1, i_orig=1, downlink=false) ----------
% aoa, eoa still hold values from the downlink call above
aoa_dl = aoa;
eoa_dl = eoa;
fbs_dl = fbs_pos;
lbs_dl = lbs_pos;

% Uplink call: only request up to eod (12 outputs), matching C++ test
[~, tx_pos_ul, tx_ori_ul, rx_pos_ul, rx_ori_ul, fbs_ul, lbs_ul, ...
    ~, ~, ~, aod_ul, eod_ul] = ...
    quadriga_lib.qrt_file_read(fn, 1, 1, false, 1);

T = [-5.86144; 53.8124; 1.0];
assertElementsAlmostEqual(tx_pos_ul, T, 'absolute', 1.5e-4);

T = [0; 0; 1.2753];
assertElementsAlmostEqual(tx_ori_ul, T, 'absolute', 1.5e-4);

T = [-2.67888; 60.257; 2.0];
assertElementsAlmostEqual(rx_pos_ul, T, 'absolute', 1.5e-4);

T = [0; 0; pi];
assertElementsAlmostEqual(rx_ori_ul, T, 'absolute', 1.5e-4);

% fbs/lbs swapped between uplink and downlink
assertElementsAlmostEqual(fbs_ul, lbs_dl, 'absolute', 1.5e-4);
assertElementsAlmostEqual(lbs_ul, fbs_dl, 'absolute', 1.5e-4);

% uplink aod/eod match downlink aoa/eoa
assertElementsAlmostEqual(aod_ul, aoa_dl, 'absolute', 1.5e-4);
assertElementsAlmostEqual(eod_ul, eoa_dl, 'absolute', 1.5e-4);

% ---------- qrt_file_parse: v5 file ----------
fn5 = 'data/test_v5.qrt';

[~, ~, ~, no_freq5, ~, ~, ~, version5] = ...
    quadriga_lib.qrt_file_parse(fn5);

assertEqual(version5, int32(5));
assertEqual(no_freq5, uint64(2));

% ---------- qrt_file_read: v5, normalize_M=0 ----------
[center_freq5, ~, ~, ~, ~, ~, ~, path_gain5, ~, M5] = ...
    quadriga_lib.qrt_file_read(fn5, 1, 0, true, 0);

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
    quadriga_lib.qrt_file_read(fn5, 1, 0, true, 1);

T = [1; 0; 0; 0; 0; 0; -1; 0];
assertElementsAlmostEqual(M5n(:, 1, 2), T, 'absolute', 1.5e-4);

end
