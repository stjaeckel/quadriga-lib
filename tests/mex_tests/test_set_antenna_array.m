% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_set_antenna_array
% MEX interface test for quadriga_lib.set_antenna_array.
% Scope: argument and output handling, 1-based node indices, the stored array, error paths.
% The renumbering of the antennas and the bands are tested in the Catch2 suite (test_set_antenna.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals without antennas, two bands; an omni-directional array
fn_tx = 'test_set_antenna_array_tx.csv';
fn_rx = 'test_set_antenna_array_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
write_text( fn_tx, 'name,x,y,z\nA,0,0,25\nB,500,0,25\n' );
write_text( fn_rx, 'name,x,y,z\nU1,10,20,1.5\nU2,30,40,1.5\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
ant = quadriga_lib.generate_antenna( 'omni', 10 );

%% Basic call: the array is stored for all bands of the Tx nodes
L2 = quadriga_lib.set_antenna_array( L, ant );
assertEqual( L2.tx.i_antenna, uint32( [ 1; 1 ] ) );                             % 1-based, 0 = generic antenna
assertEqual( L2.rx.i_antenna, uint32( [ 0; 0 ] ) );
assertEqual( size( L2.antenna ), [ 1, 1 ] );
assertEqual( size( L2.antenna{1} ), [ 1, 1 ] );                                 % one array for all bands
assertEqual( L2.antenna{1}.e_theta_re, ant.e_theta_re );
assertEqual( L2.antenna{1}.azimuth_grid, ant.azimuth_grid );
assertEqual( L2.antenna_args, { cell( 0, 1 ) } );                               % no line: an evaluated array without description
assertTrue( isempty( L.antenna ) );                                             % the input stays unchanged

%% Optional inputs: rx_side with 1-based nodes, a band
L3 = quadriga_lib.set_antenna_array( L, ant, true, 2 );
assertEqual( L3.rx.i_antenna, uint32( [ 0; 1 ] ) );
assertEqual( L3.tx.i_antenna, uint32( [ 0; 0 ] ) );
L4 = quadriga_lib.set_antenna_array( L, ant, false, [], 3.5e9 );               % one band: the other band is not set yet
assertEqual( L4.center_freq, [ 2e9; 3.5e9 ] );
assertEqual( L4.tx.i_antenna, uint32( [ 1; 1 ] ) );
assertEqual( size( L4.antenna{1} ), [ 1, 2 ] );
assertEqual( numel( L4.antenna{1}(2).e_theta_re ), numel( ant.e_theta_re ) );
assertTrue( isempty( L4.antenna{1}(1).e_theta_re ) );                          % band 1 is not evaluated
L4b = quadriga_lib.set_antenna( L4, 'dipole', [], true );                       % the Tx arrays pass through C++ unchanged, also the band that is not evaluated
assertTrue( isequaln( L4b.antenna{1}, L4.antenna{1} ) );

%% Outputs: none
quadriga_lib.set_antenna_array( L, ant );
assertTrue( isequaln( ans, L2 ) );

%% Type handling: uint8 indices as a column, single band
L5 = quadriga_lib.set_antenna_array( L, ant, 0, uint8( [ 1; 2 ] ), single( NaN ) );
assertTrue( isequaln( L5, L2 ) );

%% Errors
assert_throws( @() quadriga_lib.set_antenna_array( L ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.set_antenna_array( L, ant, false, [], NaN, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( L, ant ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.set_antenna_array( 1, ant ), 'Input ''layout_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.set_antenna_array( L, 1 ), 'Input ''array'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.set_antenna_array( L, [ ant, ant ] ), 'Input ''array'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.set_antenna_array( L, ant, false, 0 ), 'Entries in ''i_node'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.set_antenna_array( L, ant, false, 3 ), 'Input ''i_node'' has an index outside the nodes of the Tx side.' );   % C++
bad = ant;
bad.e_theta_re = bad.e_theta_re(:, :, [ 1, 1 ]);
assert_throws( @() quadriga_lib.set_antenna_array( L, bad ), 'Input ''array'' is not valid' );                                  % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.set_antenna_array( varargin{:} ); %#ok<ASGLU>
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
