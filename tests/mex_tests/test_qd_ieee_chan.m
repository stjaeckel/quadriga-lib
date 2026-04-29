function test_qd_ieee_chan

% =====================================================================
% Covered tests:
% - K-Factor model for Type A (xpol)
% - Correctness of TX/RX positions and orientations (TGac defaults + manual)
% - LOS Steering matrix (Xpol isolation)
% - Floors and distances: path counts, floor indicator, relative path gains
% - Seed consistency
% - Wall penetration loss (default 5 dB, custom 7 dB, per-user vector)
% - TGax floor penetration loss formula (n=2,3,4 vs n=1 reference)
% - Doppler shift on the LOS path (moving station)
% - Uplink-downlink reciprocity (time-domain)
% =====================================================================


% ---------------------------------------------------------------------
% 1) Type A, single user, cross-polarised antenna
% ---------------------------------------------------------------------
ant = quadriga_lib.generate_arrayant('xpol');
chan = quadriga_lib.get_channels_ieee_indoor(ant, ant, 'A');

assertEqual(numel(chan), 1);

sz = size(chan(1).coeff_re);
assertEqual(sz(1), 2);   % n_rx
assertEqual(sz(2), 2);   % n_tx
assertEqual(sz(3), 2);   % n_path
% n_snap = sz(4) if 4D, else 1; we only require it is 1 here.
n_snap = 1;
if numel(sz) >= 4
    n_snap = sz(4);
end
assertEqual(n_snap, 1);

assertElementsAlmostEqual(chan(1).center_frequency(1), 5.25e9, 'absolute', 0);

% Both paths have equal power at KF = 0 dB
pg = chan(1).path_gain(:);
assertEqual(numel(pg), 2);
assertElementsAlmostEqual(pg(1), pg(2), 'absolute', 1e-14);

% Path gain reconstructed from coefficients (sum |H|^2 over rx*tx, * 0.5)
H_pow = chan(1).coeff_re(:, :, :, 1).^2 + chan(1).coeff_im(:, :, :, 1).^2;
pg_from_coeff = squeeze(sum(sum(H_pow, 1), 2)) * 0.5;
assertElementsAlmostEqual(pg, pg_from_coeff(:), 'absolute', 1e-12);

% TX at origin facing east; RX 4.99 m east facing west (yaw = pi)
assertElementsAlmostEqual(chan(1).tx_position(:, 1),    [0; 0; 0],     'absolute', 0);
assertElementsAlmostEqual(chan(1).tx_orientation(:, 1), [0; 0; 0],     'absolute', 0);
assertElementsAlmostEqual(chan(1).rx_position(:, 1),    [4.99; 0; 0],  'absolute', 0);
assertElementsAlmostEqual(chan(1).rx_orientation(:, 1), [0; 0; pi],    'absolute', 1e-12);

% Off-diagonal LOS coefficients should be zero (perfect Xpol isolation)
H0_re = chan(1).coeff_re(:, :, 1, 1);
H0_im = chan(1).coeff_im(:, :, 1, 1);
assertElementsAlmostEqual(H0_re(1, 2), 0, 'absolute', 0);
assertElementsAlmostEqual(H0_re(2, 1), 0, 'absolute', 0);
assertElementsAlmostEqual(H0_im(1, 2), 0, 'absolute', 0);
assertElementsAlmostEqual(H0_im(2, 1), 0, 'absolute', 0);

% Diagonal elements have equal power
p00 = H0_re(1, 1)^2 + H0_im(1, 1)^2;
p11 = H0_re(2, 2)^2 + H0_im(2, 2)^2;
assertElementsAlmostEqual(p00, p11, 'absolute', 0);


% ---------------------------------------------------------------------
% 2) Type A, two users, default TGac offset angles
% ---------------------------------------------------------------------
ant = quadriga_lib.generate_arrayant('omni', 30.0);
chan = call_ieee(ant, ant, 'A', ...
    'CarrierFreq_Hz', 2.4e9, ...
    'tap_spacing_s', 1e-8, ...
    'n_users', 2);

assertEqual(numel(chan), 2);

% User 1: AoD = -78.0189°, AoA = -135.3011° (TGac IEEE 802.11-09/0308r12)
aod_deg = -78.0189;
aoa_deg = -135.3011;
cx = 4.99 * cos(aod_deg * pi/180);
cy = 4.99 * sin(aod_deg * pi/180);
ori_deg = aod_deg - aoa_deg - 180;
assertElementsAlmostEqual(chan(1).rx_position(:, 1), [cx; cy; 0], 'absolute', 1e-5);
assertElementsAlmostEqual(chan(1).rx_orientation(3, 1) * 180/pi, ori_deg, 'absolute', 1e-4);

% User 2: AoD = -142.9707°, AoA = 115.1550°
aod_deg = -142.9707;
aoa_deg = 115.1550;
cx = 4.99 * cos(aod_deg * pi/180);
cy = 4.99 * sin(aod_deg * pi/180);
ori_deg = aod_deg - aoa_deg + 180;
assertElementsAlmostEqual(chan(2).rx_position(:, 1), [cx; cy; 0], 'absolute', 1e-5);
assertElementsAlmostEqual(chan(2).rx_orientation(3, 1) * 180/pi, ori_deg, 'absolute', 1e-4);


% ---------------------------------------------------------------------
% 3) Type A, two users, manual offset angles (all 100°)
% ---------------------------------------------------------------------
offset_angles = 100 * ones(4, 2);
chan = call_ieee(ant, ant, 'A', ...
    'CarrierFreq_Hz', 2.4e9, ...
    'tap_spacing_s', 1e-8, ...
    'n_users', 2, ...
    'Dist_m', 1.99, ...
    'offset_angles', offset_angles, ...
    'n_subpath', 100, ...
    'seed', 11);

assertEqual(numel(chan), 2);

aod_deg = 100;
aoa_deg = 100;
cx = 1.99 * cos(aod_deg * pi/180);
cy = 1.99 * sin(aod_deg * pi/180);
ori_deg = aod_deg - aoa_deg + 180;

for u = 1:2
    assertElementsAlmostEqual(chan(u).rx_position(:, 1), [cx; cy; 0], 'absolute', 1e-5);
    assertElementsAlmostEqual(chan(u).rx_orientation(3, 1) * 180/pi, ori_deg, 'absolute', 1e-4);
end


% ---------------------------------------------------------------------
% 4) Type B, three users, mixed distances and floors
% ---------------------------------------------------------------------
chan = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 2.4e9, ...
    'tap_spacing_s', 10e-9, ...
    'n_users', 3, ...
    'Dist_m', [4.0, 8.0, 2.0], ...
    'n_floors', uint64([0, 0, 1]));

assertEqual(numel(chan), 3);

% User 1: dist 4 < dBP=5, no floor → LOS exists, 13 paths
assertEqual(size(chan(1).coeff_re, 3), 13);
% User 2: dist 8 > dBP=5 → NLOS only, 12 paths
assertEqual(size(chan(2).coeff_re, 3), 12);
% User 3: n_floors=1 → NLOS only, 12 paths
assertEqual(size(chan(3).coeff_re, 3), 12);

% Floor indicator on rx_pos(3)
assertElementsAlmostEqual(chan(1).rx_position(3, 1), 0, 'absolute', 0);
assertElementsAlmostEqual(chan(2).rx_position(3, 1), 0, 'absolute', 0);
assertElementsAlmostEqual(chan(3).rx_position(3, 1), 3, 'absolute', 0);

% Relative path gains for user 1 (KF=0 dB → first two paths equal)
p = chan(1).path_gain(:);
p = p / (2 * p(1));
p_dB = 10 * log10(p);
expected = -[3.01; 3.01; 5.40; 10.8; 3.20; 16.2; 6.30; ...
             21.7; 9.40; 12.5; 15.6; 18.7; 21.8];
assertElementsAlmostEqual(p_dB, expected, 'absolute', 0.01);

% Relative path gains for user 2 (NLOS only, first 4 entries)
p = chan(2).path_gain(:);
p = p / p(1);
p_dB = 10 * log10(p);
assertElementsAlmostEqual(p_dB(1:4), -[0; 5.40; 10.8; 3.20], 'absolute', 0.01);

% Relative path gains for user 3 (NLOS only via floor, same model B NLOS)
p = chan(3).path_gain(:);
p = p / p(1);
p_dB = 10 * log10(p);
assertElementsAlmostEqual(p_dB(1:4), -[0; 5.40; 10.8; 3.20], 'absolute', 0.01);

% Spot-check delays for user 1 (omni: all rx,tx pairs share the same delay)
d_ns = chan(1).delay(:, :, :, 1) * 1e9;
sz = size(d_ns);
d_per_path = squeeze(mean(reshape(d_ns, sz(1)*sz(2), sz(3)), 1));
expected_ns = [0, 0, 10, 20, 20, 30, 30, 40, 40, 50, 60, 70, 80].';
assertElementsAlmostEqual(d_per_path(:), expected_ns, 'absolute', 1.5);


% ---------------------------------------------------------------------
% 5) Seed consistency: identical seed → identical channel
% ---------------------------------------------------------------------
chan_a = call_ieee(ant, ant, 'B', 'CarrierFreq_Hz', 2.4e9, ...
                   'Dist_m', 3, 'seed', 1234);
chan_b = call_ieee(ant, ant, 'B', 'CarrierFreq_Hz', 2.4e9, ...
                   'Dist_m', 3, 'seed', 1234);

assertElementsAlmostEqual(chan_a(1).coeff_re,  chan_b(1).coeff_re,  'absolute', 0);
assertElementsAlmostEqual(chan_a(1).coeff_im,  chan_b(1).coeff_im,  'absolute', 0);
assertElementsAlmostEqual(chan_a(1).delay,     chan_b(1).delay,     'absolute', 0);
assertElementsAlmostEqual(chan_a(1).path_gain, chan_b(1).path_gain, 'absolute', 0);


% ---------------------------------------------------------------------
% 6) Wall penetration loss (NEW: n_walls / wall_loss)
%
% Wall loss is a deterministic additive term in the path loss. With a
% fixed seed and otherwise identical inputs, walls don't change path
% structure or RNG draws, so the SF realisation is identical and the
% total path power in dB must differ by exactly -n_walls * wall_loss.
% ---------------------------------------------------------------------
seed = 42;

% Reference: no walls
chan_ref = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 5.25e9, 'Dist_m', 3, 'seed', seed, ...
    'n_walls', uint64(0), 'wall_loss', 5.0);

% 3 walls @ default 5 dB → -15 dB
chan_w3 = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 5.25e9, 'Dist_m', 3, 'seed', seed, ...
    'n_walls', uint64(3), 'wall_loss', 5.0);

assertEqual(size(chan_ref(1).coeff_re, 3), size(chan_w3(1).coeff_re, 3));
P_ref = sum(chan_ref(1).path_gain(:));
P_w3  = sum(chan_w3(1).path_gain(:));
assertElementsAlmostEqual(10*log10(P_w3 / P_ref), -15.0, 'absolute', 1e-9);

% 2 walls @ custom 7 dB → -14 dB
chan_w2_7 = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 5.25e9, 'Dist_m', 3, 'seed', seed, ...
    'n_walls', uint64(2), 'wall_loss', 7.0);

assertEqual(size(chan_ref(1).coeff_re, 3), size(chan_w2_7(1).coeff_re, 3));
P_w2_7 = sum(chan_w2_7(1).path_gain(:));
assertElementsAlmostEqual(10*log10(P_w2_7 / P_ref), -14.0, 'absolute', 1e-9);

% Per-user n_walls vector — each user's ratio set by its own wall count
chan3_ref = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 5.25e9, 'n_users', 3, ...
    'Dist_m', [3, 3, 3], 'seed', seed, ...
    'n_walls', uint64([0, 0, 0]), 'wall_loss', 5.0);

chan3_w = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 5.25e9, 'n_users', 3, ...
    'Dist_m', [3, 3, 3], 'seed', seed, ...
    'n_walls', uint64([1, 2, 3]), 'wall_loss', 5.0);

assertEqual(numel(chan3_ref), 3);
assertEqual(numel(chan3_w),   3);
for u = 1:3
    assertEqual(size(chan3_ref(u).coeff_re, 3), size(chan3_w(u).coeff_re, 3));
    pr = sum(chan3_ref(u).path_gain(:, 1));
    pw = sum(chan3_w(u).path_gain(:, 1));
    expected = -double(u) * 5.0;   % users 1,2,3 with 1,2,3 walls @ 5 dB
    assertElementsAlmostEqual(10*log10(pw / pr), expected, 'absolute', 1e-9);
end


% ---------------------------------------------------------------------
% 7) TGax floor penetration loss formula (NEW: dual-band floor logic)
%
% For TGax (CarrierFreq >= 1 GHz) the per-floor penetration loss is
%     FL(n) = 18.3 * n^((n+2)/(n+1) - 0.46)   [dB], for n_floors >= 1
% Comparing two runs with n_floors > 0 ensures both lose the LOS path,
% so path structure and SF realisation are identical at a fixed seed.
% ---------------------------------------------------------------------
seed = 7;
FL = @(n) 18.3 .* n.^((n + 2) ./ (n + 1) - 0.46);

chan1 = call_ieee(ant, ant, 'B', ...
    'CarrierFreq_Hz', 5.25e9, 'Dist_m', 3, ...
    'n_floors', uint64(1), 'seed', seed);
P1 = sum(chan1(1).path_gain(:));

for nf = [2, 3, 4]
    chan_n = call_ieee(ant, ant, 'B', ...
        'CarrierFreq_Hz', 5.25e9, 'Dist_m', 3, ...
        'n_floors', uint64(nf), 'seed', seed);
    assertEqual(size(chan_n(1).coeff_re, 3), size(chan1(1).coeff_re, 3));
    Pn = sum(chan_n(1).path_gain(:));
    expected = -(FL(nf) - FL(1));
    assertElementsAlmostEqual(10*log10(Pn / P1), expected, 'absolute', 1e-9);
end


% ---------------------------------------------------------------------
% 8) Doppler shift on the LOS path (moving station)
% ---------------------------------------------------------------------
fGHz                = 2.4;
update_rate_s       = 0.0025;
observation_time_s  = 10.0;
speed_station_kmh   = 50.0;

chan = call_ieee(ant, ant, 'A', ...
    'CarrierFreq_Hz', fGHz * 1e9, 'tap_spacing_s', 10e-9, ...
    'observation_time', observation_time_s, ...
    'update_rate', update_rate_s, ...
    'speed_station_kmh', speed_station_kmh, ...
    'speed_env_kmh', 0.0, ...
    'seed', 1234);

sz = size(chan(1).coeff_re);
if numel(sz) >= 4
    n_snap = sz(4);
else
    n_snap = 1;
end
assert(n_snap > 100, 'Doppler test needs many snapshots');

% LOS path coefficient over snapshots (omni: n_rx = n_tx = 1)
h_los_re = squeeze(chan(1).coeff_re(1, 1, 1, :));
h_los_im = squeeze(chan(1).coeff_im(1, 1, 1, :));
h_los    = h_los_re(:) + 1i * h_los_im(:);

H  = fftshift(fft(h_los));
df = 1 / (n_snap * update_rate_s);
freqs = (-floor(n_snap/2) : floor((n_snap-1)/2)) * df;

[~, idx_peak] = max(abs(H));
f_peak = freqs(idx_peak);

v = speed_station_kmh / 3.6;
fD_expected = v * (fGHz * 1e9) / 3e8;

assertElementsAlmostEqual(abs(f_peak), fD_expected, 'absolute', 5.0);


% ---------------------------------------------------------------------
% 9) Uplink-downlink reciprocity (time domain)
%
% For zero observation time and a fixed seed, the uplink channel must
% equal the conjugate transpose of the downlink channel per path, with
% identical (transposed) delays.
% ---------------------------------------------------------------------
ant = quadriga_lib.generate_arrayant('xpol');   % 2 elements at each end

common = {'CarrierFreq_Hz', 2.4e9, 'tap_spacing_s', 5e-9, ...
          'n_users', 2, 'Doppler_effect', 40.0, 'seed', 1234};

chan_dl = call_ieee(ant, ant, 'F', 'uplink', false, common{:});
chan_ul = call_ieee(ant, ant, 'F', 'uplink', true,  common{:});

assertEqual(numel(chan_dl), 2);
assertEqual(numel(chan_ul), 2);

sz_dl = size(chan_dl(1).coeff_re);
sz_ul = size(chan_ul(1).coeff_re);
assertEqual(sz_dl(3), sz_ul(3));   % same n_path
assertEqual(sz_dl(1), sz_ul(2));   % n_rx_DL = n_tx_UL
assertEqual(sz_dl(2), sz_ul(1));   % n_tx_DL = n_rx_UL

n_path = sz_dl(3);
max_err = 0;
for k = 1:n_path
    Hdl_re = chan_dl(1).coeff_re(:, :, k, 1);
    Hdl_im = chan_dl(1).coeff_im(:, :, k, 1);
    Hul_re = chan_ul(1).coeff_re(:, :, k, 1);
    Hul_im = chan_ul(1).coeff_im(:, :, k, 1);

    % Reciprocity: H_ul = conj(H_dl).' (real transposes, imag negated transposed)
    e_re = max(max(abs(Hul_re - Hdl_re.')));
    e_im = max(max(abs(Hul_im + Hdl_im.')));

    Ddl = chan_dl(1).delay(:, :, k, 1);
    Dul = chan_ul(1).delay(:, :, k, 1);
    e_d  = max(max(abs(Dul - Ddl.')));

    max_err = max([max_err, e_re, e_im, e_d]);
end
assertElementsAlmostEqual(max_err, 0, 'absolute', 1e-10);

end


% =====================================================================
% Helper: call get_channels_ieee_indoor with name-value defaults
% =====================================================================
function chan = call_ieee(ant_ap, ant_sta, model, varargin)

% Documented defaults (match the C++ / Python signatures)
p.CarrierFreq_Hz     = 5.25e9;
p.tap_spacing_s      = 10e-9;
p.n_users            = 1;
p.observation_time   = 0.0;
p.update_rate        = 1e-3;
p.speed_station_kmh  = 0.0;
p.speed_env_kmh      = 1.2;
p.Dist_m             = 4.99;
p.n_floors           = uint64(0);
p.uplink             = false;
p.offset_angles      = [];
p.n_subpath          = 20;
p.Doppler_effect     = 50.0;
p.seed               = -1;
p.KF_linear          = NaN;
p.XPR_NLOS_linear    = NaN;
p.SF_std_dB_LOS      = NaN;
p.SF_std_dB_NLOS     = NaN;
p.dBP_m              = NaN;
p.n_walls            = uint64(0);
p.wall_loss          = 5.0;

% Apply name-value overrides
if mod(numel(varargin), 2) ~= 0
    error('call_ieee:badArgs', 'Expected name-value pairs.');
end
for k = 1:2:numel(varargin)
    name = varargin{k};
    if ~isfield(p, name)
        error('call_ieee:unknownArg', 'Unknown argument: %s', name);
    end
    p.(name) = varargin{k+1};
end

chan = quadriga_lib.get_channels_ieee_indoor( ...
    ant_ap, ant_sta, model, ...
    p.CarrierFreq_Hz, p.tap_spacing_s, p.n_users, ...
    p.observation_time, p.update_rate, ...
    p.speed_station_kmh, p.speed_env_kmh, ...
    p.Dist_m, p.n_floors, p.uplink, ...
    p.offset_angles, p.n_subpath, p.Doppler_effect, p.seed, ...
    p.KF_linear, p.XPR_NLOS_linear, ...
    p.SF_std_dB_LOS, p.SF_std_dB_NLOS, p.dBP_m, ...
    p.n_walls, p.wall_loss);

end