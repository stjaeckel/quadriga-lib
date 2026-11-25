% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_generate_layout
% MEX interface test for quadriga_lib.generate_layout.
% Scope: argument and output handling, units of the tuples, struct fields, defaults, error paths.
% The presets, the drop and the random streams are tested in the Catch2 suite (test_generate_layout.cpp).

%% Setup: UMa, one ring of sites (7 sites with 3 sectors = 21 Tx nodes), 2 UTs per cell
d2r = pi / 180;
L = quadriga_lib.generate_layout( '38.901_UMa', 2, [], [ 1, NaN, 3 ], [], [], [], [], [], true, 42 );

%% Basic call: sizes, classes and values of the preset
assertEqual( fieldnames( L )', { 'name', 'tx', 'rx', 'links', 'seed', 'center_freq', 'antenna', 'antenna_cols', 'antenna_args', 'wrap', 'pairing' } );
assertEqual( size( L.tx.pos ), [ 3, 21 ] );
assertEqual( size( L.rx.pos ), [ 3, 42 ] );                                     % 2 UTs in each of the 21 cells
assertEqual( L.tx.name(1:4)', { 'BS1', 'BS1', 'BS1', 'BS2' } );
assertEqual( L.rx.name(1:2)', { 'UT1', 'UT2' } );
assertEqual( L.tx.sector(1:4), uint32( [ 0; 1; 2; 0 ] ) );
assertEqual( L.tx.offset(1:3), uint64( [ 1; 2; 3 ] ) );                         % 1-based
assertEqual( L.tx.pos(:, 1), [ 0; 0; 25 ] );                                    % site 1 at the origin, hBS = 25 m
assertElementsAlmostEqual( norm( L.tx.pos(1:2, 4) - L.tx.pos(1:2, 1) ), 500, 'absolute', 1e-9 );   % ISD 500 m
assertElementsAlmostEqual( L.tx.orientation(3, 1:3), [ 30, 150, -90 ] * d2r, 'absolute', 1e-14 );   % boresights, in rad
assertEqual( L.tx.scenario(1:2)', { 'UMa', 'UMa' } );
assertEqual( size( L.wrap ), [ 2, 2 ] );
assertEqual( L.seed, int64( 42 ) );
assertTrue( isa( L.rx.indoor, 'int32' ) && numel( L.rx.indoor ) == 42 );
assertTrue( all( L.rx.pos(3, :) >= 1.5 ) );
assertElementsAlmostEqual( L.rx.speed, 3 / 3.6 * ones( 42, 1 ), 'absolute', 1e-14 );              % 3 km/h
assertTrue( isempty( L.links.i_tx ) && isempty( L.center_freq ) && isempty( L.antenna ) );

%% The layout is a function of the seed
L2 = quadriga_lib.generate_layout( '38.901_UMa', 2, [], [ 1, NaN, 3 ], [], [], [], [], [], true, 42 );
assertTrue( isequaln( L, L2 ) );
L3 = quadriga_lib.generate_layout( '38.901_UMa', 2, [], [ 1, NaN, 3 ], [], [], [], [], [], true, 43 );
assertEqual( L3.tx.pos, L.tx.pos );
assertFalse( isequal( L3.rx.pos, L.rx.pos ) );

%% Optional inputs: tuples in degrees and metres (NaN = preset value), options, per_cell, seed
L4 = quadriga_lib.generate_layout( '38.901_UMa', 5, [ 1000, NaN ], [ 0, NaN, 1 ], [], [ 10, NaN, 15 ], [ NaN, NaN, NaN, 0, 90 ], [], [], false, 7 );
assertEqual( size( L4.tx.pos ), [ 3, 1 ] );
assertEqual( size( L4.rx.pos ), [ 3, 5 ] );                                     % per_cell = false: 5 UTs in total
assertEqual( L4.tx.pos(3), 10 );
assertElementsAlmostEqual( L4.tx.orientation(:, 1)', [ 0, 0, 45 ] * d2r, 'absolute', 1e-14 );     % boresight 30 deg + offset 15 deg
assertElementsAlmostEqual( L4.rx.orientation(1:2, :), [ zeros( 1, 5 ); 90 * d2r * ones( 1, 5 ) ], 'absolute', 1e-14 );
Ls = quadriga_lib.generate_layout( '38.901_SMa', 1, [], [ 0, NaN, 1 ], [], [], [], [], struct( 'p_residential', 0.5 ), true, 3 );
assertEqual( size( Ls.tx.pos ), [ 3, 1 ] );
Ln = quadriga_lib.generate_layout( '38.901_UMa', 0, [], [ 0, NaN, 1 ] );        % without UTs; no seed: one is drawn
assertTrue( isempty( Ln.rx.pos ) );
assertTrue( Ln.seed >= 0 );
Lh = quadriga_lib.generate_layout( '38.901_InH_Open', 0, [], [], [], [], [], [], [], true, 5 );                       % indoor hall, 6 x 2 BSs at 20 m, no wrap
assertEqual( size( Lh.tx.pos ), [ 3, 12 ] );
assertEqual( Lh.tx.pos(:, [ 1, 12 ]), [ -50, 50; -10, 10; 3, 3 ] );
assertEqual( size( Lh.wrap ), [ 0, 0 ] );

%% Outputs: none
quadriga_lib.generate_layout( '38.901_InH_Open', 0, [], [], [], [], [], [], [], true, 5 );
assertTrue( isequaln( ans, Lh ) );

%% Type handling: single and int32 arguments
L5 = quadriga_lib.generate_layout( '38.901_UMa', int32( 5 ), single( [ 1000; NaN ] ), single( [ 0; NaN; 1 ] ), [], single( [ 10; NaN; 15 ] ), single( [ NaN, NaN, NaN, 0, 90 ] ), [], [], 0, uint8( 7 ) );
assertTrue( isequaln( L5, L4 ) );

%% Errors
assert_throws( @() quadriga_lib.generate_layout(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.generate_layout( '38.901_UMa', 1, [], [], [], [], [], [], [], true, 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( '38.901_UMa' ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.generate_layout( 1 ), 'Input ''type'' must be a string' );
assert_throws( @() quadriga_lib.generate_layout( '38.901_UMa', 1.5 ), 'Input ''n_ut'' must be' );
assert_throws( @() quadriga_lib.generate_layout( '38.901_UMa', 1, [], [], [], [], [], [], [], true, -2 ), 'Input ''seed'' must be -1 or a non-negative integer.' );
assert_throws( @() quadriga_lib.generate_layout( '38.901_InH_Open', 3, [], [], [], [], [], [], struct( 'foo', 1 ) ), 'Input ''options'' has an unknown key (''foo'').' );   % C++
assert_throws( @() quadriga_lib.generate_layout( '38.901_InH_Open', 3, [], [], [], [], [], [], 5 ), 'Input ''options'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.generate_layout( 'no_such_preset' ), 'no_such_preset' );                                % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.generate_layout( varargin{:} ); %#ok<ASGLU>
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
