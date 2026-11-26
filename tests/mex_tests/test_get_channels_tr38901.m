% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_get_channels_tr38901
% MEX interface test for quadriga_lib.get_channels_tr38901.
% Scope: argument and output handling, 1-based indices, the channel struct array, optional inputs, error paths.
% The channels are tested in the Catch2 suite (test_get_channels_tr38901.cpp).

%% Setup: a 2x1 BS panel and a cross-polarized UT
bs = quadriga_lib.generate_antenna( '3gpp', 10, 3.5e9, [], [2, 1, 1, 1] );
ut = quadriga_lib.generate_antenna( 'xpol', 10 );

%% Basic call: one site with 3 sectors, 2 UTs per cell; UT fastest; 1-based index
[ chan, gain_dB, layout, index ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 2, 7, [], 0 );
assertEqual( size( chan ), [ 1, 6 ] );
assertEqual( size( gain_dB ), [ 6, 3 ] );
assertEqual( class( index ), 'uint64' );
assertEqual( index(2, :), uint64( 1 : 6 ) );
assertEqual( index(3, :), uint64( ones( 1, 6 ) ) );
[ ~, best ] = max( gain_dB, [], 2 );
assertEqual( index(1, :), uint64( best' ) );
assertEqual( size( chan(1).coeff_re, 1 ), 2 );                                  % 2 UT ports (xpol)
assertEqual( size( chan(1).coeff_re, 2 ), 2 );                                  % 2 BS ports
assertEqual( layout.links.method, '38.901' );
assertEqual( size( layout.pairing ), [ 2, 6 ] );

%% The returned layout gives the same channels through get_channels_layout
[ ch2, idx2 ] = quadriga_lib.get_channels_layout( layout );
keep = idx2(1, :) == index(1, idx2(2, :));
assertEqual( sum( keep ), 6 );
c2 = ch2(keep);
for n = 1 : 6
    assertEqual( chan(n).coeff_re, c2(n).coeff_re );
end

%% The layout tuples reach generate_layout: ISD 200 m, one sector, BS at 30 m, UTs outdoors at 1.8 m
[ ~, gt, Lt ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 3, 5, 200, [0, NaN, 1], [], 30, 1.8, 0 );
L = quadriga_lib.generate_layout( '38.901_UMa', 3, 3.5e9, 200, [0, NaN, 1], [], 30, 1.8, 0, [], true, 5 );
assertEqual( size( gt ), [ 3, 1 ] );
assertEqual( Lt.tx.pos, L.tx.pos );
assertEqual( Lt.rx.pos, L.rx.pos );

%% Optional inputs, also as []
a = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', [ 3.5e9, 28e9 ], 1, 5, [], 0 );
assertEqual( size( a ), [ 1, 6 ] );
assertEqual( a(4).center_freq, 28e9 );
b = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [NaN, NaN, 10], [], [ 0.01, 5e-3 ] );
assertEqual( size( b(1).coeff_re, 4 ), 3 );
assertElementsAlmostEqual( norm( b(1).rx_pos(:, 3) - b(1).rx_pos(:, 1) ), 0.1, 'absolute', 1e-9 );
[ c, ~, Lc ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], [], [], 'nlos', 'rays', [], [], false );
assertTrue( all( Lc.links.los == 0 ) );
assertTrue( isfield( c, 'path_gain' ) );
c_on = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], [], [], 'nlos', 'rays' );
assertTrue( sum( c(1).coeff_re(:).^2 + c(1).coeff_im(:).^2 ) > 1e4 * sum( c_on(1).coeff_re(:).^2 + c_on(1).coeff_im(:).^2 ) );   % no path loss
[ ~, gd, Ld ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], 0, [], [], [], 'rsrp' );
assertTrue( all( Ld.rx.indoor == 0 ) );
[ ~, gl ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], 0 );
assertTrue( any( abs( gd(:) - gl(:) ) > 1e-6 ) );                               % RSRP and coupling gain differ
[ ~, o0 ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 1, 5, [], 0, [], [], [], 1, [], [], [], [], [], [], [], 0 );
[ ~, o1 ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 1, 5, [], 0, [], [], [], 1, [], [], [], [], [], [], [], 1 );
assertTrue( all( o1(:) > o0(:) ) );                                             % the low-loss building model at all UTs
f = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], [], [], [], [], [], [], [], true, [], {} );
[ s, ~, ~, is ] = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], [], [], [], [], [], 'site' );
assertEqual( size( s ), [ 1, 9 ] );                                             % 3 sectors of the serving site per UT
assertEqual( is(2, :), uint64( [ 1, 1, 1, 2, 2, 2, 3, 3, 3 ] ) );
s = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0, [], [], [], [], [], [], [], [], 'all' );
assertEqual( size( s ), [ 1, 9 ] );                                             % one site: all = site
g = quadriga_lib.get_channels_tr38901( bs, ut, 'UMi', 3.5e9, 1, 5, [], 0 );
assertEqual( f(1).tx_pos, g(1).rx_pos );
assertElementsAlmostEqual( f(1).coeff_re(:, :, 1), g(1).coeff_re(:, :, 1).', 'absolute', 0 );

%% Defaults: 3.5 GHz, one UT per cell of the 12 InH cells
h = quadriga_lib.get_channels_tr38901( bs, ut, 'InH_Open', [], [], 1 );
assertEqual( size( h ), [ 1, 12 ] );
assertEqual( h(1).center_freq, 3.5e9 );

%% Outputs: none and fewer
quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 2, 7, [], 0 );
assertTrue( isequaln( ans, chan ) );

%% Type handling: single frequency, int32 counts
k = quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', single( 3.5e9 ), int32( 2 ), 7, [], int32( 0 ) );
assertEqual( size( k ), [ 1, 6 ] );

%% Errors
assert_throws( @() quadriga_lib.get_channels_tr38901( bs, ut ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 1, 1, [], 0, [], [], [], [], [], 'auto', 'taps', 'large_scale', 'serving', true, false, NaN, {}, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out5( bs, ut, 'UMa' ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 1, 1, [], 0, [], [], [], [], [ 1, 2, 3 ] ), 'Input ''time_grid'' must have at most 2 entries' );   % C++
assert_throws( @() quadriga_lib.get_channels_tr38901( bs, ut, 'UMa', 3.5e9, 1, 1, [], 0, [], [], [], [], [], 'LOS' ), 'Input ''los'' must be ''auto'', ''los'' or ''nlos''.' );    % C++

end

% Request 5 outputs so that the output-count guard fires
function out5(varargin)
[~, ~, ~, ~, ~] = quadriga_lib.get_channels_tr38901( varargin{:} ); %#ok<ASGLU>
end

% Assert that fn() raises an error whose message contains expected (substring).
function assert_throws(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end
