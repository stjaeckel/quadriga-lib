% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_freq_response
% MEX interface test for quadriga_lib.freq_response.
% Scope: argument and output handling, index conversion, cell-array batches, kernels, error paths.
% Physics validation lives in the Catch2 suite (test_freq_response.cpp).

%% Common single-frequency test data
% [n_rx=4, n_tx=3, n_path=2]
coeff = zeros(4,3,2);
coeff(:,1,1) = 0.25:0.25:1;
coeff(:,2,1) = 1:4;
coeff(:,3,1) = 1j*(1:4);
coeff(:,:,2) = -coeff(:,:,1);

fc = 299792458.0;   % wavelength = 1 m

% Planar-wave delays, broadcast across all RX/TX: [1, 1, n_path]
delay = zeros(1,1,2);
delay(1) = 1.0/fc;
delay(2) = 1.5/fc;

pilots = 0:0.1:2;   % 21 pilot positions, range 0..2

%% One snapshot
hmat = quadriga_lib.freq_response( coeff, delay, pilots, fc );

assertEqual( size(hmat), [4 3 21] );
assertTrue( isa(hmat, 'double') );

% pilot=0 and pilot=2 -> rotation multiple of 2pi; slice0 + slice1 = 0
T = zeros(4, 3);
assertElementsAlmostEqual( hmat(:,:,1),  T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat(:,:,21), T, 'absolute', 3e-6 );

% pilot=1 -> slice0 phase = 2pi (=> 1), slice1 phase = 3pi (=> -1), H = 2*slice0
T = [ 0.5, 2, 0 ; 1, 4, 0 ; 1.5, 6, 0 ; 2, 8, 0 ] + 1j * [ 0, 0, 2 ; 0, 0, 4 ; 0, 0, 6 ; 0, 0, 8 ];
assertElementsAlmostEqual( hmat(:,:,11), T, 'absolute', 1.5e-6 );

% Linearity: col 2 of coeff is 4x col 1
assertElementsAlmostEqual( hmat(:,1,:)*4, hmat(:,2,:), 'absolute', 1.5e-6 );

% Real coefficients are complex coefficients with imaginary part 0
h_r = quadriga_lib.freq_response( real(coeff), delay, pilots, fc );
h_c = quadriga_lib.freq_response( complex(real(coeff), zeros(size(coeff))), delay, pilots, fc );
assertEqual( h_r, h_c );

%% Single carrier
hmat_s1 = quadriga_lib.freq_response( coeff, delay, 0.5, fc );
assertEqual( size(hmat_s1), [4 3] );
hmat_ref = quadriga_lib.freq_response( coeff, delay, [0.4, 0.5, 0.6], fc );
assertElementsAlmostEqual( hmat_s1, hmat_ref(:,:,2), 'absolute', 1e-6 );

%% Batch: cell arrays, snapshot s has its coefficients scaled by s
n_snap = 3;
cf = cell(n_snap, 1); dl = cell(n_snap, 1);
for s = 1:n_snap
    cf{s} = s * coeff;
    dl{s} = delay;
end

%% All snapshots (i_snap omitted)
hmat_ms = quadriga_lib.freq_response( cf, dl, pilots, fc );
assertEqual( size(hmat_ms), [4 3 21 3] );
for s = 1:n_snap
    assertElementsAlmostEqual( hmat_ms(:,:,:,s), s * hmat, 'absolute', 1e-5 );
end

%% i_snap
assertEqual( quadriga_lib.freq_response( cf, dl, pilots, fc, [1 2 3] ), hmat_ms );

hmat_sub = quadriga_lib.freq_response( cf, dl, pilots, fc, [3 1] );
assertEqual( size(hmat_sub), [4 3 21 2] );
assertEqual( hmat_sub, hmat_ms(:,:,:,[3 1]) );

% One array is one snapshot
hmat_1 = quadriga_lib.freq_response( coeff, delay, pilots, fc, [1 1] );
assertEqual( size(hmat_1), [4 3 21 2] );
assertEqual( hmat_1(:,:,:,2), hmat );

% Without assignment the output is ans
quadriga_lib.freq_response( cf, dl, pilots, fc );
assertEqual( ans, hmat_ms );

% [] equals an omitted optional input
assertEqual( quadriga_lib.freq_response( cf, dl, pilots, fc, [], [] ), hmat_ms );

% Single-precision inputs give double outputs
h_sp = quadriga_lib.freq_response( single(coeff), single(delay), single(pilots), single(fc) );
assertTrue( isa(h_sp, 'double') );
assertElementsAlmostEqual( h_sp, hmat, 'absolute', 1e-5 );

%% The number of paths may differ between snapshots
cf_v = { coeff, coeff(:,:,1) };
dl_v = { delay, delay(1) };
h_v = quadriga_lib.freq_response( cf_v, dl_v, pilots, fc );
h_1 = quadriga_lib.freq_response( cf_v{2}, dl_v{2}, pilots, fc );
assertEqual( size(h_v), [4 3 21 2] );
assertEqual( h_v(:,:,:,1), hmat );
assertEqual( h_v(:,:,:,2), h_1 );

%% Errors
assert_throws( @() quadriga_lib.freq_response( cat(4, coeff, coeff), delay, pilots, fc ), 'more than 3 dimensions' );
assert_throws( @() quadriga_lib.freq_response( cf, delay, pilots, fc ), 'must both be cell arrays' );
assert_throws( @() quadriga_lib.freq_response( coeff, dl, pilots, fc ), 'must both be cell arrays' );
if ~exist('OCTAVE_VERSION', 'builtin') % Octave has no complex integers
    assert_throws( @() quadriga_lib.freq_response( complex(int32(1), int32(1)), delay(1), pilots, fc ), 'unsupported data type' );
end
assert_throws( @() quadriga_lib.freq_response( coeff, complex(delay, delay), pilots, fc ), 'complex inputs are not supported' );
assert_throws( @() quadriga_lib.freq_response( cf, dl, pilots, fc, [0 1] ), '1-based' );
assert_throws( @() quadriga_lib.freq_response( cf, dl, pilots, fc, [1 5] ), 'outside the range of the' );
assert_throws( @() quadriga_lib.freq_response( {coeff, zeros(2,3,2)}, {delay, delay}, pilots, fc ), ...
    'same number of rows and columns' );
assert_throws( @() quadriga_lib.freq_response( cf, dl(1:2), pilots, fc ), 'same number of snapshots' );
assert_throws( @() quadriga_lib.freq_response( coeff, delay, pilots ), 'Wrong number of input arguments' );
assert_throws( @() quadriga_lib.freq_response( coeff, delay, pilots, fc, [], 0, 1 ), 'Wrong number of input arguments' );
assert_throws( @() two_outputs( coeff, delay, pilots, fc ), 'Wrong number of output arguments' );

% use_kernel (input 6) reaches C++ for one snapshot and a batch: 1 is accepted, 3 is rejected by C++
cc = reshape(complex(sin(1:24), cos(1:24)), 2, 3, 4); d4 = reshape((1:4) * 1e-8, 1, 1, 4);
pg = linspace(0, 1, 5)';
modes = { {cc, d4, pg, 1e8, []}, {{cc, cc}, {d4, d4}, pg, 1e8, []} };
for m = 1:2
    [~] = quadriga_lib.freq_response( modes{m}{:}, 1 );
    assert_throws( @() quadriga_lib.freq_response( modes{m}{:}, 3 ), 'Input ''use_kernel'' must be 0, 1 or 2.' );
end
end

function two_outputs( varargin )
[~, ~] = quadriga_lib.freq_response( varargin{:} );
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
