function test_generate_speaker

% ================================================================================
% Output structure and types
% ================================================================================

% Default call returns a struct array with all expected fields
data = quadriga_lib.generate_speaker();
required_fields = {'e_theta_re', 'e_theta_im', 'e_phi_re', 'e_phi_im', ...
                   'azimuth_grid', 'elevation_grid', 'element_pos', ...
                   'coupling_re', 'coupling_im', 'center_freq', 'name'};
for i = 1:numel(required_fields)
    assertTrue(isfield(data, required_fields{i}));
end

% Default produces multiple frequency entries
assertTrue(numel(data) > 1);

% All four pattern fields have identical shape per entry
sz = size(data(1).e_theta_re);
assertTrue(isequal(size(data(1).e_theta_im), sz));
assertTrue(isequal(size(data(1).e_phi_re), sz));
assertTrue(isequal(size(data(1).e_phi_im), sz));

% center_freq is scalar per entry; concatenation yields 1D array
cf = [data.center_freq];
assertTrue(isvector(cf));
assertTrue(numel(cf) == numel(data));

% Frequency samples should be positive and ascending
assertTrue(all(cf > 0));
assertTrue(all(diff(cf) > 0));

% Grids are 1D vectors
assertTrue(isvector(data(1).azimuth_grid));
assertTrue(isvector(data(1).elevation_grid));

% element_pos is (3, n_elements); single driver -> 1 element
assertTrue(isequal(size(data(1).element_pos), [3, 1]));

% ================================================================================
% Angular resolution
% ================================================================================

% 5 deg: azimuth 73, elevation 37
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], [], 5.0);
assertTrue(numel(data(1).azimuth_grid) == 73);
assertTrue(numel(data(1).elevation_grid) == 37);

% 10 deg: azimuth 37, elevation 19
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], [], 10.0);
assertTrue(numel(data(1).azimuth_grid) == 37);
assertTrue(numel(data(1).elevation_grid) == 19);

% Changing resolution changes first two pattern dims
d5  = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], [], 5.0);
d10 = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], [], 10.0);
assertTrue(size(d5(1).e_theta_re, 1) > size(d10(1).e_theta_re, 1));
assertTrue(size(d5(1).e_theta_re, 2) > size(d10(1).e_theta_re, 2));

% ================================================================================
% Custom frequency vector
% ================================================================================

% Explicit frequency vector is reflected in center_freq
freqs = [100.0, 500.0, 1000.0, 5000.0, 10000.0];
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], freqs);
assertTrue(numel(data) == 5);
assertElementsAlmostEqual([data.center_freq], freqs, 'absolute', 1e-6);

% Single frequency -> length-1 struct array, scalar center_freq
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], 1000.0);
assertTrue(numel(data) == 1);
assertTrue(isscalar(data.center_freq));

% Two frequencies
freqs = [500.0, 5000.0];
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], [], [], [], [], [], freqs);
assertTrue(numel(data) == 2);
assertElementsAlmostEqual([data.center_freq], freqs, 'absolute', 1e-6);

% ================================================================================
% Driver types
% ================================================================================

% Piston driver
data = quadriga_lib.generate_speaker('piston');
assertTrue(all(isfinite(data(1).e_theta_re(:))));
assertTrue(max(abs(data(1).e_theta_re(:))) > 0);

% Horn driver
data = quadriga_lib.generate_speaker('horn', 0.025, 1500.0, 20000.0);
assertTrue(all(isfinite(data(1).e_theta_re(:))));
assertTrue(max(abs(data(1).e_theta_re(:))) > 0);

% Omni driver
data = quadriga_lib.generate_speaker('omni', 0.165, 30.0, 300.0);
assertTrue(all(isfinite(data(1).e_theta_re(:))));
assertTrue(max(abs(data(1).e_theta_re(:))) > 0);

% Omni + monopole at passband center should be nearly isotropic
data = quadriga_lib.generate_speaker('omni', 0.165, 30.0, 300.0, [], [], [], 'monopole', ...
                                      [], [], [], [], [], 100.0, 10.0);
pat = data(1).e_theta_re;
rel_spread = (max(pat(:)) - min(pat(:))) / mean(abs(pat(:)));
assertTrue(rel_spread < 0.01);

% ================================================================================
% Radiation types
% ================================================================================

% All four radiation types produce finite output
rad_types = {'monopole', 'hemisphere', 'dipole', 'cardioid'};
for k = 1:numel(rad_types)
    data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], rad_types{k}, ...
                                          [], [], [], [], [], 1000.0);
    assertTrue(all(isfinite(data(1).e_theta_re(:))));
end

% Dipole: near-zero at 90 deg off-axis
data = quadriga_lib.generate_speaker('omni', [], 30.0, 20000.0, [], [], [], 'dipole', ...
                                      [], [], [], [], [], 1000.0, 5.0);
pat = data(1).e_theta_re;
el_grid = data(1).elevation_grid;
az_grid = data(1).azimuth_grid;
[~, el_mid] = min(abs(el_grid));
[~, az_0]   = min(abs(az_grid));
[~, az_90]  = min(abs(az_grid - pi/2));
on_axis  = abs(pat(el_mid, az_0));
off_axis = abs(pat(el_mid, az_90));
if on_axis > 0
    assertTrue(off_axis / on_axis < 0.15);
end

% Cardioid: near-zero at 180 deg (rear)
data = quadriga_lib.generate_speaker('omni', [], 30.0, 20000.0, [], [], [], 'cardioid', ...
                                      [], [], [], [], [], 2000.0, 5.0);
pat = data(1).e_theta_re;
el_grid = data(1).elevation_grid;
az_grid = data(1).azimuth_grid;
[~, el_mid] = min(abs(el_grid));
[~, az_0]   = min(abs(az_grid));
[~, az_180] = min(abs(az_grid - pi));
on_axis = abs(pat(el_mid, az_0));
at_rear = abs(pat(el_mid, az_180));
if on_axis > 0
    assertTrue(at_rear / on_axis < 0.15);
end

% ================================================================================
% Frequency response (bandpass behavior)
% ================================================================================

% Peak on-axis response should occur within passband
freqs = [20.0, 50.0, 100.0, 500.0, 1000.0, 5000.0, 10000.0, 18000.0];
data = quadriga_lib.generate_speaker([], [], 80.0, 12000.0, [], [], [], [], ...
                                      [], [], [], [], [], freqs, 10.0);
el_grid = data(1).elevation_grid;
az_grid = data(1).azimuth_grid;
[~, el_mid] = min(abs(el_grid));
[~, az_0]   = min(abs(az_grid));
on_axis = zeros(1, numel(freqs));
for f = 1:numel(freqs)
    on_axis(f) = abs(data(f).e_theta_re(el_mid, az_0));
end
[~, peak_idx] = max(on_axis);
peak_freq = freqs(peak_idx);
assertTrue(peak_freq >= 80.0);
assertTrue(peak_freq <= 12000.0);

% Below-passband rolloff: 500 Hz >> 20 Hz
freqs = [20.0, 500.0];
data = quadriga_lib.generate_speaker([], [], 80.0, 12000.0, [], [], [], [], ...
                                      [], [], [], [], [], freqs, 10.0);
el_mid = floor(size(data(1).e_theta_re, 1) / 2) + 1;
az_mid = floor(size(data(1).e_theta_re, 2) / 2) + 1;
amp_20  = abs(data(1).e_theta_re(el_mid, az_mid));
amp_500 = abs(data(2).e_theta_re(el_mid, az_mid));
assertTrue(amp_500 > amp_20 * 3);

% Above-passband rolloff: 5 kHz > 18 kHz
freqs = [5000.0, 18000.0];
data = quadriga_lib.generate_speaker([], [], 80.0, 12000.0, [], [], [], [], ...
                                      [], [], [], [], [], freqs, 10.0);
el_mid = floor(size(data(1).e_theta_re, 1) / 2) + 1;
az_mid = floor(size(data(1).e_theta_re, 2) / 2) + 1;
amp_5k  = abs(data(1).e_theta_re(el_mid, az_mid));
amp_18k = abs(data(2).e_theta_re(el_mid, az_mid));
assertTrue(amp_5k > amp_18k);

% ================================================================================
% Sensitivity scaling (+10 dB -> factor ~3.16 in amplitude)
% ================================================================================

d85 = quadriga_lib.generate_speaker([], [], [], [], [], [], 85.0, [], ...
                                     [], [], [], [], [], 1000.0, 10.0);
d95 = quadriga_lib.generate_speaker([], [], [], [], [], [], 95.0, [], ...
                                     [], [], [], [], [], 1000.0, 10.0);
max_85 = max(abs(d85(1).e_theta_re(:)));
max_95 = max(abs(d95(1).e_theta_re(:)));
ratio = max_95 / max_85;
assertElementsAlmostEqual(ratio, 10.0^(10.0/20.0), 'absolute', 1e-2);

% ================================================================================
% Rolloff slope: steeper attenuates out-of-band more, in-band unchanged
% ================================================================================

freqs = [20.0, 500.0];
d12 = quadriga_lib.generate_speaker([], [], 80.0, [], 12.0, [], [], [], ...
                                     [], [], [], [], [], freqs, 10.0);
d48 = quadriga_lib.generate_speaker([], [], 80.0, [], 48.0, [], [], [], ...
                                     [], [], [], [], [], freqs, 10.0);
el = floor(size(d12(1).e_theta_re, 1) / 2) + 1;
az = floor(size(d12(1).e_theta_re, 2) / 2) + 1;
amp_20_12  = abs(d12(1).e_theta_re(el, az));
amp_20_48  = abs(d48(1).e_theta_re(el, az));
amp_500_12 = abs(d12(2).e_theta_re(el, az));
amp_500_48 = abs(d48(2).e_theta_re(el, az));
assertElementsAlmostEqual(amp_500_12, amp_500_48, 'absolute', 0.1);
assertTrue(amp_20_12 > amp_20_48);

% ================================================================================
% Piston directivity vs frequency and radius
% ================================================================================

% Piston narrows at higher frequencies
freqs = [200.0, 8000.0];
data = quadriga_lib.generate_speaker('piston', 0.05, 50.0, 20000.0, [], [], [], 'monopole', ...
                                      [], [], [], [], [], freqs, 5.0);
el_grid = data(1).elevation_grid;
az_grid = data(1).azimuth_grid;
[~, el_mid] = min(abs(el_grid));
[~, az_0]   = min(abs(az_grid));
[~, az_90]  = min(abs(az_grid - pi/2));
ratio_lo = abs(data(1).e_theta_re(el_mid, az_90)) / ...
           max(abs(data(1).e_theta_re(el_mid, az_0)), 1e-30);
ratio_hi = abs(data(2).e_theta_re(el_mid, az_90)) / ...
           max(abs(data(2).e_theta_re(el_mid, az_0)), 1e-30);
assertTrue(ratio_lo > ratio_hi);

% Larger radius -> narrower beam at same frequency
d_small = quadriga_lib.generate_speaker('piston', 0.03, 50.0, 20000.0, [], [], [], 'monopole', ...
                                         [], [], [], [], [], 5000.0, 5.0);
d_large = quadriga_lib.generate_speaker('piston', 0.10, 50.0, 20000.0, [], [], [], 'monopole', ...
                                         [], [], [], [], [], 5000.0, 5.0);
el_grid = d_small(1).elevation_grid;
az_grid = d_small(1).azimuth_grid;
[~, el_mid] = min(abs(el_grid));
[~, az_0]   = min(abs(az_grid));
[~, az_90]  = min(abs(az_grid - pi/2));
ratio_small = abs(d_small(1).e_theta_re(el_mid, az_90)) / ...
              max(abs(d_small(1).e_theta_re(el_mid, az_0)), 1e-30);
ratio_large = abs(d_large(1).e_theta_re(el_mid, az_90)) / ...
              max(abs(d_large(1).e_theta_re(el_mid, az_0)), 1e-30);
assertTrue(ratio_small > ratio_large);

% ================================================================================
% Horn parameters
% ================================================================================

% Custom coverage
data = quadriga_lib.generate_speaker('horn', 0.025, 1500.0, 20000.0, [], [], [], [], ...
                                      90.0, 60.0, [], [], [], 4000.0, 5.0);
assertTrue(all(isfinite(data(1).e_theta_re(:))));
assertTrue(max(abs(data(1).e_theta_re(:))) > 0);

% Auto coverage (0 -> derived internally)
data = quadriga_lib.generate_speaker('horn', 0.025, 1500.0, 20000.0, [], [], [], [], ...
                                      0.0, 0.0, [], [], [], 4000.0, 5.0);
assertTrue(all(isfinite(data(1).e_theta_re(:))));

% ================================================================================
% Baffle parameters
% ================================================================================

% Different baffle sizes change hemisphere pattern (different baffle-step freqs)
d_small = quadriga_lib.generate_speaker([], [], [], [], [], [], [], 'hemisphere', ...
                                         [], [], [], 0.10, 0.10, 1000.0, 10.0);
d_large = quadriga_lib.generate_speaker([], [], [], [], [], [], [], 'hemisphere', ...
                                         [], [], [], 0.40, 0.40, 1000.0, 10.0);
assertTrue(~isequal(d_small(1).e_theta_re, d_large(1).e_theta_re));

% ================================================================================
% Auto frequency generation
% ================================================================================

% Auto freqs span the passband
data = quadriga_lib.generate_speaker([], [], 200.0, 8000.0);
cf = [data.center_freq];
assertTrue(cf(1) <= 200.0);
assertTrue(cf(end) >= 8000.0);

% Auto freqs within audible range
data = quadriga_lib.generate_speaker();
cf = [data.center_freq];
assertTrue(cf(1) >= 20.0);
assertTrue(cf(end) <= 20000.0);

% ================================================================================
% E-phi zero, piston+monopole e_theta_im zero
% ================================================================================

% e_phi fields are zero (acoustic: scalar pressure)
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], ...
                                      [], [], [], [], [], 1000.0);
assertElementsAlmostEqual(data(1).e_phi_re, zeros(size(data(1).e_phi_re)), 'absolute', 1e-14);
assertElementsAlmostEqual(data(1).e_phi_im, zeros(size(data(1).e_phi_im)), 'absolute', 1e-14);

% Piston + monopole: real-valued directivity, imag part zero
data = quadriga_lib.generate_speaker('piston', [], [], [], [], [], [], 'monopole', ...
                                      [], [], [], [], [], 1000.0);
assertElementsAlmostEqual(data(1).e_theta_im, zeros(size(data(1).e_theta_im)), ...
                          'absolute', 1e-14);

% ================================================================================
% Coupling matrix
% ================================================================================

% Single driver -> 1x1 identity coupling
data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], [], ...
                                      [], [], [], [], [], 1000.0);
assertElementsAlmostEqual(data(1).coupling_re, 1.0, 'absolute', 1e-14);
assertElementsAlmostEqual(data(1).coupling_im, 0.0, 'absolute', 1e-14);

% ================================================================================
% Name
% ================================================================================

data = quadriga_lib.generate_speaker('piston');
assertTrue(ischar(data(1).name));
assertTrue(~isempty(data(1).name));

% ================================================================================
% Cutoff shift moves passband
% ================================================================================

freqs = [200.0, 2000.0, 10000.0];
d_low  = quadriga_lib.generate_speaker([], [], 50.0,   1000.0, [], [], [], [], ...
                                        [], [], [], [], [], freqs, 10.0);
d_high = quadriga_lib.generate_speaker([], [], 3000.0, 18000.0, [], [], [], [], ...
                                        [], [], [], [], [], freqs, 10.0);
el = floor(size(d_low(1).e_theta_re, 1) / 2) + 1;
az = floor(size(d_low(1).e_theta_re, 2) / 2) + 1;

% Low-pass: 200 Hz > 10 kHz
amp_low_200 = abs(d_low(1).e_theta_re(el, az));
amp_low_10k = abs(d_low(3).e_theta_re(el, az));
assertTrue(amp_low_200 > amp_low_10k);

% High-pass: 10 kHz > 200 Hz
amp_high_200 = abs(d_high(1).e_theta_re(el, az));
amp_high_10k = abs(d_high(3).e_theta_re(el, az));
assertTrue(amp_high_10k > amp_high_200);

% ================================================================================
% Errors
% ================================================================================

% Invalid driver type
try
    data = quadriga_lib.generate_speaker('plasma');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        rethrow(ME);
    end
end

% Invalid radiation type
try
    data = quadriga_lib.generate_speaker([], [], [], [], [], [], [], 'laserbeam');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        rethrow(ME);
    end
end

% Too many output arguments
try
    [~, ~] = quadriga_lib.generate_speaker();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || ...
       isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ...
              ['EXPECTED: "', expectedErrorMessage, '", GOT: "', ME.message, '"']);
    end
end

end