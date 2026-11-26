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
[ hmat_re, hmat_im ] = quadriga_lib.freq_response( real(coeff), imag(coeff), delay, pilots, fc );

assertEqual( size(hmat_re), [4 3 21] );
assertEqual( size(hmat_im), [4 3 21] );

% pilot=0 and pilot=2 -> rotation multiple of 2pi; slice0 + slice1 = 0
T = zeros(4, 3);
assertElementsAlmostEqual( hmat_re(:,:,1),  T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_im(:,:,1),  T, 'absolute', 1.5e-6 );
assertElementsAlmostEqual( hmat_re(:,:,21), T, 'absolute', 3e-6 );
assertElementsAlmostEqual( hmat_im(:,:,21), T, 'absolute', 3e-6 );

% pilot=1 -> slice0 phase = 2pi (=> 1), slice1 phase = 3pi (=> -1), H = 2*slice0
T = [ 0.5, 2, 0 ; 1, 4, 0 ; 1.5, 6, 0 ; 2, 8, 0 ];
assertElementsAlmostEqual( hmat_re(:,:,11), T, 'absolute', 1.5e-6 );

% Linearity: col 2 of coeff is 4x col 1
assertElementsAlmostEqual( hmat_re(:,1,:)*4, hmat_re(:,2,:), 'absolute', 1.5e-6 );

%% Single carrier
[ hmat_re_s1, hmat_im_s1 ] = quadriga_lib.freq_response( real(coeff), imag(coeff), delay, 0.5, fc );
assertEqual( size(hmat_re_s1), [4 3] );
[ hmat_re_ref, hmat_im_ref ] = quadriga_lib.freq_response( real(coeff), imag(coeff), delay, [0.4, 0.5, 0.6], fc );
assertElementsAlmostEqual( hmat_re_s1, hmat_re_ref(:,:,2), 'absolute', 1e-6 );
assertElementsAlmostEqual( hmat_im_s1, hmat_im_ref(:,:,2), 'absolute', 1e-6 );

%% Batch: cell arrays, snapshot s has its coefficients scaled by s
n_snap = 3;
cre = cell(n_snap, 1); cim = cell(n_snap, 1); dl = cell(n_snap, 1);
for s = 1:n_snap
    cre{s} = s * real(coeff);
    cim{s} = s * imag(coeff);
    dl{s} = delay;
end

%% All snapshots (i_snap omitted)
[ hmat_re_ms, hmat_im_ms ] = quadriga_lib.freq_response( cre, cim, dl, pilots, fc );
assertEqual( size(hmat_re_ms), [4 3 21 3] );
assertEqual( size(hmat_im_ms), [4 3 21 3] );
for s = 1:n_snap
    assertElementsAlmostEqual( hmat_re_ms(:,:,:,s), s * hmat_re, 'absolute', 1e-5 );
    assertElementsAlmostEqual( hmat_im_ms(:,:,:,s), s * hmat_im, 'absolute', 1e-5 );
end

%% i_snap
[ hmat_re_a, hmat_im_a ] = quadriga_lib.freq_response( cre, cim, dl, pilots, fc, [1 2 3] );
assertEqual( hmat_re_a, hmat_re_ms );
assertEqual( hmat_im_a, hmat_im_ms );

[ hmat_re_sub, hmat_im_sub ] = quadriga_lib.freq_response( cre, cim, dl, pilots, fc, [3 1] );
assertEqual( size(hmat_re_sub), [4 3 21 2] );
assertEqual( hmat_re_sub, hmat_re_ms(:,:,:,[3 1]) );
assertEqual( hmat_im_sub, hmat_im_ms(:,:,:,[3 1]) );

% One array is one snapshot
[ hmat_re_1, hmat_im_1 ] = quadriga_lib.freq_response( real(coeff), imag(coeff), delay, pilots, fc, [1 1] );
assertEqual( size(hmat_re_1), [4 3 21 2] );
assertEqual( hmat_re_1(:,:,:,2), hmat_re );
assertEqual( hmat_im_1(:,:,:,1), hmat_im );

% Only the first output; without assignment it is ans
hmat_re_o = quadriga_lib.freq_response( cre, cim, dl, pilots, fc );
assertEqual( hmat_re_o, hmat_re_ms );
quadriga_lib.freq_response( cre, cim, dl, pilots, fc );
assertEqual( ans, hmat_re_ms );

% [] equals an omitted optional input
assertEqual( quadriga_lib.freq_response( cre, cim, dl, pilots, fc, [], [] ), hmat_re_ms );

% Single-precision inputs give double outputs
[ hr_sp, hi_sp ] = quadriga_lib.freq_response( single(real(coeff)), single(imag(coeff)), single(delay), ...
    single(pilots), single(fc) );
assertTrue( isa(hr_sp, 'double') );
assertElementsAlmostEqual( hr_sp + 1j*hi_sp, hmat_re + 1j*hmat_im, 'absolute', 1e-5 );

%% The number of paths may differ between snapshots
cre_v = { real(coeff), real(coeff(:,:,1)) };
cim_v = { imag(coeff), imag(coeff(:,:,1)) };
dl_v = { delay, delay(1) };
[ hr_v, hi_v ] = quadriga_lib.freq_response( cre_v, cim_v, dl_v, pilots, fc );
[ hr_1, hi_1 ] = quadriga_lib.freq_response( cre_v{2}, cim_v{2}, dl_v{2}, pilots, fc );
assertEqual( size(hr_v), [4 3 21 2] );
assertEqual( hr_v(:,:,:,1), hmat_re );
assertEqual( hr_v(:,:,:,2), hr_1 );
assertEqual( hi_v(:,:,:,2), hi_1 );

%% Errors
assert_throws( @() quadriga_lib.freq_response( cat(4, real(coeff), real(coeff)), ...
    cat(4, imag(coeff), imag(coeff)), delay, pilots, fc ), 'more than 3 dimensions' );
assert_throws( @() quadriga_lib.freq_response( cre, imag(coeff), dl, pilots, fc ), 'must all be cell arrays' );
assert_throws( @() quadriga_lib.freq_response( real(coeff), cim, dl, pilots, fc ), 'must all be cell arrays' );
assert_throws( @() quadriga_lib.freq_response( cre, cim, dl, pilots, fc, [0 1] ), '1-based' );
assert_throws( @() quadriga_lib.freq_response( cre, cim, dl, pilots, fc, [1 5] ), 'outside the range of the' );
assert_throws( @() quadriga_lib.freq_response( {real(coeff), zeros(2,3,2)}, {imag(coeff), zeros(2,3,2)}, ...
    {delay, delay}, pilots, fc ), 'same number of rows and columns' );
assert_throws( @() quadriga_lib.freq_response( cre, cim, dl(1:2), pilots, fc ), 'same number of snapshots' );
assert_throws( @() quadriga_lib.freq_response( real(coeff), imag(coeff), delay, pilots ), 'Wrong number of input arguments' );
assert_throws( @() quadriga_lib.freq_response( real(coeff), imag(coeff), delay, pilots, fc, [], 0, 1 ), ...
    'Wrong number of input arguments' );
assert_throws( @() three_outputs( real(coeff), imag(coeff), delay, pilots, fc ), 'Wrong number of output arguments' );

% use_kernel (input 7) reaches C++ for one snapshot and a batch: 1 is accepted, 3 is rejected by C++
cr = reshape(sin(1:24), 2, 3, 4); ci = reshape(cos(1:24), 2, 3, 4); d4 = reshape((1:4) * 1e-8, 1, 1, 4);
pg = linspace(0, 1, 5)';
modes = { {cr, ci, d4, pg, 1e8, []}, {{cr, cr}, {ci, ci}, {d4, d4}, pg, 1e8, []} };
for m = 1:2
    [~] = quadriga_lib.freq_response( modes{m}{:}, 1 );
    assert_throws( @() quadriga_lib.freq_response( modes{m}{:}, 3 ), 'Input ''use_kernel'' must be 0, 1 or 2.' );
end
end

function three_outputs( varargin )
[~, ~, ~] = quadriga_lib.freq_response( varargin{:} );
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
