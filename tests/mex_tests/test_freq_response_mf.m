% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_freq_response_mf
% MEX interface test for quadriga_lib.freq_response_mf.
% Scope: argument and output handling, index conversion, cell-array batches, kernels, error paths.
% Physics validation lives in the Catch2 suite (test_freq_response_mf.cpp).

%% Common test data
% [n_rx=4, n_tx=3, n_path=2]
coeff = zeros(4,3,2);
coeff(:,1,1) = 0.25:0.25:1;
coeff(:,2,1) = 1:4;
coeff(:,3,1) = 1j*(1:4);
coeff(:,:,2) = -coeff(:,:,1);

fc = 299792458.0;   % wavelength = 1 m
delay = zeros(1,1,2);
delay(1) = 1.0/fc;
delay(2) = 1.5/fc;

% Replicate envelope (= coeff) across 3 input frequencies and bake in the
% delay phase exp(-j 2pi freq tau), simulating get_channels_multifreq output.
freq  = [1e9 ; 2e9 ; 3e9];
coeff_4d = zeros(4, 3, 2, 3);
delay_4d    = repmat(delay, [1 1 1 3]);  % [1, 1, 2, 3]
for f = 1:3
    for p = 1:2
        coeff_4d(:,:,p,f) = coeff(:,:,p) * exp(-1j*2*pi*freq(f)*delay(1,1,p));
    end
end
freq_new = linspace(1e9, 3e9, 11)';

%% Cross-check against the single-frequency function
% Constant envelope across freq -> SLERP is identity. H must match a single-frequency
% call with pilot_grid * bandwidth = freq_new.
hmat_m = quadriga_lib.freq_response_mf( coeff_4d, delay_4d, freq, freq_new );
hmat_sf = quadriga_lib.freq_response( coeff, delay, freq_new/fc, fc );
assertEqual( size(hmat_m), [4 3 11] );
assertTrue( isa(hmat_m, 'double') );
assertElementsAlmostEqual( hmat_m, hmat_sf, 'absolute', 1e-5 );

% Real coefficients are complex coefficients with imaginary part 0
h_r = quadriga_lib.freq_response_mf( real(coeff_4d), delay_4d, freq, freq_new );
h_c = quadriga_lib.freq_response_mf( complex(real(coeff_4d), zeros(size(coeff_4d))), delay_4d, freq, freq_new );
assertEqual( h_r, h_c );

% 3D delay gives the same result as the 4D delay of the generators
assertEqual( quadriga_lib.freq_response_mf( coeff_4d, delay, freq, freq_new ), hmat_m );

% Single output carrier
hmat_m1 = quadriga_lib.freq_response_mf( coeff_4d, delay_4d, freq, 1.4e9 );
assertEqual( size(hmat_m1), [4 3] );
assertElementsAlmostEqual( hmat_m1, hmat_m(:,:,3), 'absolute', 1e-6 );

%% SLERP magnitude interpolation across freq
% Envelope amplitude ramps 1, 2, 3 across freq; expect linear interpolation.
amps  = [1 2 3];
tau_r = 100e-9;
c = reshape(amps(:) .* exp(-1j*2*pi*freq*tau_r), 1, 1, 1, 3);
h_r = quadriga_lib.freq_response_mf( c, tau_r, freq, freq_new );
assertElementsAlmostEqual( squeeze(abs(h_r)), interp1(freq, amps, freq_new), 'absolute', 1e-5 );

%% Absolute carriers for single-frequency coefficients
% One input frequency f0 (a 3D array): H(f) = sum_p c_p * exp(-j 2 pi (f - f0) tau_p), the result of
% freq_response with pilot_grid * bandwidth = f - f0. Carriers in any order, also below f0.
cre = reshape([1 0.5], 1, 1, 2); cim = reshape([0 0.25], 1, 1, 2); dl = reshape([10e-9 35e-9], 1, 1, 2);
f0 = 3e9;
Hx = @(f) sum(squeeze(cre + 1j*cim) .* exp(-2j*pi*(f(:).' - f0) .* squeeze(dl)), 1).';
fa = f0 + [10e6; 0; 20e6; -10e6];
cb = (cre + 1j*cim) .* exp(-2j*pi*f0*dl);   % delay phase at f0, as the generators return it
h = quadriga_lib.freq_response_mf( cb, dl, f0, fa );
assertElementsAlmostEqual( squeeze(h), Hx(fa), 'absolute', 1e-5 );
h_b = quadriga_lib.freq_response( complex(cre, cim), dl, (fa - f0) / 10e6, 10e6 );
assertElementsAlmostEqual( h, h_b, 'absolute', 1e-5 );

%% Remove_delay_phase = false (envelope coefficients)
% A constant envelope c gets the delay phase of freq_new only: H(f) = sum_p c_p * exp(-j 2 pi f tau_p)
ce = cat(4, cre + 1j*cim, cre + 1j*cim);
fin = [1e9; 1.01e9]; fout = [1.002e9; 1.007e9];   % not on a multiple of 1/tau, so the two settings differ
h_e = quadriga_lib.freq_response_mf( ce, dl, fin, fout, false );
He = sum(squeeze(cre + 1j*cim) .* exp(-2j*pi*fout.' .* squeeze(dl)), 1).';
assertElementsAlmostEqual( squeeze(h_e), He, 'absolute', 1e-6 );
h_t = quadriga_lib.freq_response_mf( ce, dl, fin, fout, true );
assertTrue( max(abs(h_t(:) - h_e(:))) > 0.1 );

%% Batch as cell arrays, i_snap
cf_b = { coeff_4d, 2 * coeff_4d, coeff_4d(:,:,1,:) };
dl_b = { delay_4d, delay, delay(1) };
h_b = quadriga_lib.freq_response_mf( cf_b, dl_b, freq, freq_new );
assertEqual( size(h_b), [4 3 11 3] );
assertEqual( h_b(:,:,:,1), hmat_m );
assertElementsAlmostEqual( h_b(:,:,:,2), 2 * hmat_m, 'absolute', 1e-12 );
h_p = quadriga_lib.freq_response_mf( cf_b{3}, dl_b{3}, freq, freq_new );
assertEqual( h_b(:,:,:,3), h_p );

h_s = quadriga_lib.freq_response_mf( cf_b, dl_b, freq, freq_new, true, [3 1] );
assertEqual( h_s, h_b(:,:,:,[3 1]) );

% Without assignment the output is ans; [] equals an omitted optional input
quadriga_lib.freq_response_mf( cf_b, dl_b, freq, freq_new );
assertEqual( ans, h_b );
assertEqual( quadriga_lib.freq_response_mf( cf_b, dl_b, freq, freq_new, [], [], [] ), h_b );

% Single-precision inputs give double outputs
h_sp = quadriga_lib.freq_response_mf( single(coeff_4d), single(delay), single(freq), single(freq_new) );
assertTrue( isa(h_sp, 'double') );
assertElementsAlmostEqual( h_sp, hmat_m, 'absolute', 1e-5 );

%% Errors
assert_throws( @() quadriga_lib.freq_response_mf( coeff_4d, delay_4d(:,:,:,1:2), freq, freq_new ), ...
    'Input ''delay'' must have a 4th dimension of 1 or the length of ''freq''.' );
assert_throws( @() quadriga_lib.freq_response_mf( coeff_4d, delay, freq(1:2), freq_new ), ...
    'must have the same length as ''freq''' );
assert_throws( @() quadriga_lib.freq_response_mf( cf_b, delay, freq, freq_new ), 'must both be cell arrays' );
assert_throws( @() quadriga_lib.freq_response_mf( coeff_4d, dl_b, freq, freq_new ), 'must both be cell arrays' );
assert_throws( @() quadriga_lib.freq_response_mf( {coeff_4d, coeff_4d(1,:,:,:)}, {delay, delay}, freq, freq_new ), ...
    'n_rx and n_tx of the first selected snapshot' );
if ~exist('OCTAVE_VERSION', 'builtin') % Octave has no complex integers
    assert_throws( @() quadriga_lib.freq_response_mf( complex(int32(1), int32(1)), delay(1), freq(1), freq_new ), 'unsupported data type' );
end
assert_throws( @() quadriga_lib.freq_response_mf( cf_b, dl_b, freq, freq_new, true, 0 ), '1-based' );
assert_throws( @() quadriga_lib.freq_response_mf( cf_b, dl_b, freq, freq_new, true, 4 ), ...
    'outside the range of the' );
assert_throws( @() quadriga_lib.freq_response_mf( coeff_4d, delay, flipud(freq), freq_new ), ...
    'strictly ascending' );
assert_throws( @() quadriga_lib.freq_response_mf( coeff_4d, delay, freq ), 'Wrong number of input arguments' );
assert_throws( @() quadriga_lib.freq_response_mf( coeff_4d, delay, freq, freq_new, true, [], 0, 1 ), ...
    'Wrong number of input arguments' );
assert_throws( @() two_outputs( coeff_4d, delay, freq, freq_new ), 'Wrong number of output arguments' );

% use_kernel (input 7) reaches C++ for one snapshot and a batch: 1 is accepted, 3 is rejected by C++
modes = { {coeff_4d, delay, freq, freq_new, true, []}, {{coeff_4d}, {delay}, freq, freq_new, true, []} };
for m = 1:2
    [~] = quadriga_lib.freq_response_mf( modes{m}{:}, 1 );
    assert_throws( @() quadriga_lib.freq_response_mf( modes{m}{:}, 3 ), 'Input ''use_kernel'' must be 0, 1 or 2.' );
end
end

function two_outputs( varargin )
[~, ~] = quadriga_lib.freq_response_mf( varargin{:} );
end

function assert_throws( fn, expected )
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end
