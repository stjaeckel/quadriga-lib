% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_drop_terminals
% MEX interface test for quadriga_lib.drop_terminals.
% Scope: argument and output handling, input and output layout, defaults, error paths.
% The cells, the drop rules and the random streams are tested in the Catch2 suite (test_generate_layout.cpp).

%% Setup: UMa, one ring (7 sites, 3 sectors), without UTs
L = quadriga_lib.generate_layout( '38.901_UMa', 0, [], [ 1, NaN, 3 ], [], [], [], [], [], true, 9 );
assertTrue( isempty( L.rx.pos ) );

%% Basic call: the drop equals that of generate_layout with the same seed; the input stays unchanged
L1 = quadriga_lib.drop_terminals( L, 2, [], [], [], [], true, 9 );
assertEqual( size( L1.rx.pos ), [ 3, 42 ] );                                    % 2 UTs in each of the 21 cells
assertEqual( L1.rx.name(1:2)', { 'UT1', 'UT2' } );
assertEqual( L1.rx.offset(1:3), uint64( [ 1; 2; 3 ] ) );                        % 1-based
assertEqual( L1.tx, L.tx );
assertTrue( isempty( L.rx.pos ) );
Lg = quadriga_lib.generate_layout( '38.901_UMa', 2, [], [ 1, NaN, 3 ], [], [], [], [], [], true, 9 );
assertTrue( isequaln( L1, Lg ) );

%% The member tx.cell: the micro sites of M.2412 Dense Urban config 3 have no cells of their own (uint32, 0)
D = quadriga_lib.generate_layout( 'M.2412_DenseUrban', 0, [], [], [], [], [], [], struct( 'config', 3 ), true, 4 );
assertEqual( D.tx.cell, uint32( [ ones( 57, 1 ); zeros( 171, 1 ) ] ) );
D1 = quadriga_lib.drop_terminals( D, 1, [], [], [], [], true, 4 );
assertEqual( size( D1.rx.pos ), [ 3, 228 ] );                                   % 4 UTs in each of the 57 macro cells
assertEqual( D1.tx, D.tx );
D.tx.cell = [];                                                                 % empty: all sites have cells
D2 = quadriga_lib.drop_terminals( D, 1, [], [], [], [], true, 4 );
assertEqual( size( D2.rx.pos ), [ 3, 228 ] );
D.tx.cell = uint32( zeros( 228, 1 ) );
assert_throws( @() quadriga_lib.drop_terminals( D, 1 ), 'Input ''layout'' has no site with ''cell'' = 1.' );

%% Optional inputs: bounds, ut, per_cell and seed
H = quadriga_lib.generate_layout( '38.901_InH_Open', 0, [], [], [], [], [], [], [], true, 5 );
H1 = quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ], [], [], [], false, 2 );
assertEqual( size( H1.rx.pos ), [ 3, 4 ] );                                     % per_cell = false: 4 UTs in total
assertTrue( all( abs( H1.rx.pos(1, :) ) <= 60 ) && all( abs( H1.rx.pos(2, :) ) <= 25 ) );
assertEqual( H1.rx.pos(3, :), ones( 1, 4 ) );                                   % UT height in an indoor hall
H2 = quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ], [ 1.5, 1, 0.8, 0, 90, 0 ], [], [], false, 2 );   % ut in m, m/s and degrees
assertEqual( H2.rx.pos(3, :), 1.5 * ones( 1, 4 ) );
assertElementsAlmostEqual( H2.rx.speed, 0.8 * ones( 4, 1 ), 'absolute', 1e-14 );
assertElementsAlmostEqual( H2.rx.orientation, [ zeros( 1, 4 ); pi / 2 * ones( 1, 4 ); zeros( 1, 4 ) ], 'absolute', 1e-14 );
H3 = quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ], [], [], [], false, 2 );
assertTrue( isequaln( H3, H1 ) );
H4 = quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ], [], [], [], false, 3 );
assertFalse( isequal( H4.rx.pos, H1.rx.pos ) );
H5 = quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ] );                  % omitted: per_cell, seed = layout seed
assertEqual( size( H5.rx.pos ), [ 3, 48 ] );
assertEqual( H5.seed, H.seed );

%% Outputs: none
quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ], [], [], [], false, 2 );
assertTrue( isequaln( ans, H1 ) );

%% Type handling: single and int32 arguments
H6 = quadriga_lib.drop_terminals( H, int32( 4 ), single( [ -60; 60; -25; 25 ] ), [], [], [], 0, int8( 2 ) );
assertTrue( isequaln( H6, H1 ) );

%% Errors
assert_throws( @() quadriga_lib.drop_terminals( H ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.drop_terminals( H, 1, [], [], [], [], true, 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( H, 1 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.drop_terminals( 1, 4 ), 'Input ''layout_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.drop_terminals( H, 4, [], [], [], [], true, -2 ), 'Input ''seed'' must be -1 or a non-negative integer.' );
assert_throws( @() quadriga_lib.drop_terminals( H, 4 ), 'Input ''bounds'' is required' );                                  % C++
assert_throws( @() quadriga_lib.drop_terminals( H, 4, [ 60, -60, -25, 25 ] ), 'Input ''bounds'' must be empty or' );          % C++
assert_throws( @() quadriga_lib.drop_terminals( H, 4, [ -60, 60, -25, 25 ], [], [], struct( 'foo', 1 ) ), 'unknown key' );   % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.drop_terminals( varargin{:} ); %#ok<ASGLU>
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
