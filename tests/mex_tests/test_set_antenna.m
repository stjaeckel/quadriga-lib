% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_set_antenna
% MEX interface test for quadriga_lib.set_antenna.
% Scope: argument and output handling, 1-based node indices, the argument struct, the antenna lists of the layout, error paths.
% The renumbering of the antennas, the bands and the evaluation rules are tested in the Catch2 suite (test_set_antenna.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals without antennas, two bands
fn_tx = 'test_set_antenna_tx.csv';
fn_rx = 'test_set_antenna_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
write_text( fn_tx, 'name,x,y,z\nA,0,0,25\nB,500,0,25\n' );
write_text( fn_rx, 'name,x,y,z\nU1,10,20,1.5\nU2,30,40,1.5\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertEqual( L.tx.i_antenna, uint32( [ 0; 0 ] ) );

%% Basic call: a 3GPP panel of 2 x 2 x 1 x 2 elements for all Tx nodes
L2 = quadriga_lib.set_antenna( L, '3GPP', struct( 'panel', [ 2, 2, 1, 2 ] ) );
assertEqual( L2.tx.i_antenna, uint32( [ 1; 1 ] ) );                             % one shared entry; 1-based, 0 = generic antenna
assertEqual( L2.rx.i_antenna, uint32( [ 0; 0 ] ) );
assertEqual( L2.antenna_cols, { 'antenna'; 'panel' } );
assertEqual( L2.antenna_args, { { '3GPP,"2,2,1,2"' } } );
assertEqual( size( L2.antenna ), [ 1, 1 ] );
assertEqual( size( L2.antenna{1} ), [ 1, 2 ] );                                 % an array spaced in wavelengths: one per band
assertEqual( size( L2.antenna{1}(2).element_pos ), [ 3, 8 ] );
assertEqual( L.tx.i_antenna, uint32( [ 0; 0 ] ) );                              % the input stays unchanged
assertTrue( isempty( L.antenna ) );

%% Optional inputs: rx_side and 1-based nodes, a band, evaluate
L3 = quadriga_lib.set_antenna( L, '3GPP', struct( 'panel', [ 2, 2, 1, 2 ] ), true, 2 );
assertEqual( L3.rx.i_antenna, uint32( [ 0; 1 ] ) );
assertEqual( L3.tx.i_antenna, uint32( [ 0; 0 ] ) );
L4 = quadriga_lib.set_antenna( L, 'dipole', [], false, [], 2e9 );              % one band: a line per band, the layout is not valid yet
assertEqual( L4.antenna_args, { { '2000000000,dipole'; '3500000000,' } } );
L4 = quadriga_lib.set_antenna( L4, '3GPP', struct( 'panel', [ 2, 2, 1, 2 ] ), false, [], 3.5e9 );
assertEqual( L4.antenna_args, { { '2000000000,dipole,'; '3500000000,3GPP,"2,2,1,2"' } } );
assertEqual( size( L4.antenna{1}(2).element_pos ), [ 3, 8 ] );
L5 = quadriga_lib.set_antenna( L, '3GPP', struct( 'panel', [ 2, 2, 1, 2 ] ), false, [], NaN, false );   % no evaluation
assertEqual( size( L5.antenna ), [ 0, 1 ] );
assertEqual( L5.tx.i_antenna, uint32( [ 1; 1 ] ) );
assertEqual( L5.antenna_args, L2.antenna_args );
L6 = quadriga_lib.set_antenna( L2, 'none', [], false, 1 );                      % no antenna for the first node
assertEqual( L6.tx.i_antenna, uint32( [ 0; 1 ] ) );
L7 = quadriga_lib.set_antenna( L, '3GPP', struct( 'panel', single( [ 2; 2; 1; 2 ] ), 'center_freq', 3.5e9 ) );   % the band in args
assertEqual( L7.antenna_args, { { '2000000000,,'; '3500000000,3GPP,"2,2,1,2"' } } );   % the other band is not set

%% Outputs: none
quadriga_lib.set_antenna( L, '3GPP', struct( 'panel', [ 2, 2, 1, 2 ] ) );
assertTrue( isequaln( ans, L2 ) );

%% Type handling: single and uint8 arguments, a column of nodes
L8 = quadriga_lib.set_antenna( L, '3GPP', struct( 'panel', single( [ 2, 2, 1, 2 ] ) ), 0, uint8( [ 1; 2 ] ), single( NaN ), 1 );
assertTrue( isequaln( L8, L2 ) );

%% Errors
assert_throws( @() quadriga_lib.set_antenna( L ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.set_antenna( L, 'dipole', [], false, [], NaN, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( L, 'dipole' ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.set_antenna( 1, 'dipole' ), 'Input ''layout_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.set_antenna( L, 'dipole', 5 ), 'Input ''args'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.set_antenna( L, 'dipole', [], false, 0 ), 'Entries in ''i_node'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.set_antenna( L, 'dipole', [], false, 3 ), 'Input ''i_node'' has an index outside the nodes of the Tx side.' );   % C++
assert_throws( @() quadriga_lib.set_antenna( L, '3GPP', struct( 'foo', 1 ) ), 'Input ''args'' has the unknown argument ''foo''.' );               % C++
assert_throws( @() quadriga_lib.set_antenna( L, 'dipole', struct( 'panel', [ 2, 2, 1, 2 ] ) ), 'is not used by the antenna type ''dipole''' );   % C++
assert_throws( @() quadriga_lib.set_antenna( L, 'nope' ), 'Input ''type'' must be an antenna type of generate_antenna' );                       % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.set_antenna( varargin{:} ); %#ok<ASGLU>
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

% Write a text file; escapes such as \n in the text are converted by fprintf (the texts hold no percent signs)
function write_text(fn, text)
fid = fopen(fn, 'w');
fprintf(fid, text);
fclose(fid);
end

% Delete the files that exist
function delete_files(files)
for n = 1 : numel(files)
    if exist(files{n}, 'file')
        delete(files{n});
    end
end
end
