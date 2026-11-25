% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_set_antenna_qdant
% MEX interface test for quadriga_lib.set_antenna_qdant.
% Scope: argument and output handling, 1-based node indices, the stored line and array, error paths.
% The renumbering of the antennas, the bands and the CSV round trip are tested in the Catch2 suite (test_set_antenna.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals without antennas, two bands; a QDANT file with two entries
fn_tx = 'test_set_antenna_qdant_tx.csv';
fn_rx = 'test_set_antenna_qdant_rx.csv';
fn_q = 'test_set_antenna_qdant.qdant';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx, fn_q } ) ); %#ok<NASGU>
write_text( fn_tx, 'name,x,y,z\nA,0,0,25\nB,500,0,25\n' );
write_text( fn_rx, 'name,x,y,z\nU1,10,20,1.5\nU2,30,40,1.5\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
a1 = quadriga_lib.generate_antenna( 'omni', 30 );
a2 = quadriga_lib.generate_antenna( 'dipole', 30 );
quadriga_lib.qdant_write( fn_q, a1, 1 );
quadriga_lib.qdant_write( fn_q, a2, 2 );
F = strrep( fullfile( pwd, fn_q ), '\', '/' );                                  % normalised absolute path
r2 = quadriga_lib.qdant_read( fn_q, 2 );

%% Basic call: entry 1 for all bands of the Tx nodes
L2 = quadriga_lib.set_antenna_qdant( L, fn_q );
assertEqual( L2.tx.i_antenna, uint32( [ 1; 1 ] ) );                             % 1-based, 0 = generic antenna
assertEqual( L2.rx.i_antenna, uint32( [ 0; 0 ] ) );
assertEqual( L2.antenna_cols, { 'antenna'; 'qdant_file'; 'qdant_id' } );
assertEqual( L2.antenna_args, { { [ 'qdant,', F, ',1' ] } } );
assertEqual( size( L2.antenna{1} ), [ 1, 1 ] );                                 % one array for all bands
assertTrue( isempty( L.antenna ) );                                             % the input stays unchanged

%% Optional inputs: entry, rx with 1-based nodes, a band, no evaluation
L3 = quadriga_lib.set_antenna_qdant( L, fn_q, 2, true, 2 );
assertEqual( L3.rx.i_antenna, uint32( [ 0; 1 ] ) );
assertEqual( L3.tx.i_antenna, uint32( [ 0; 0 ] ) );
assertEqual( L3.antenna_args, { { [ 'qdant,', F, ',2' ] } } );
assertEqual( L3.antenna{1}.e_theta_re, r2.e_theta_re );
L4 = quadriga_lib.set_antenna_qdant( L, fn_q, 2, false, [], 3.5e9 );           % one band: the other band is not set yet
assertEqual( L4.antenna_args, { { '2000000000,,,'; [ '3500000000,qdant,', F, ',2' ] } } );
L5 = quadriga_lib.set_antenna_qdant( L, fn_q, 2, false, [], NaN, false );
assertEqual( L5.antenna_args, { { [ 'qdant,', F, ',2' ] } } );
assertTrue( isempty( L5.antenna ) );                                            % not evaluated

%% Outputs: none
quadriga_lib.set_antenna_qdant( L, fn_q );
assertTrue( isequaln( ans, L2 ) );

%% Type handling: uint8 entry and indices as a column, single band
L6 = quadriga_lib.set_antenna_qdant( L, fn_q, uint8( 1 ), 0, uint8( [ 1; 2 ] ), single( NaN ) );
assertTrue( isequaln( L6, L2 ) );

%% Errors
assert_throws( @() quadriga_lib.set_antenna_qdant( L ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.set_antenna_qdant( L, fn_q, 1, false, [], NaN, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( L, fn_q ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.set_antenna_qdant( L, fn_q, -1 ), 'qdant_id' );
assert_throws( @() quadriga_lib.set_antenna_qdant( L, fn_q, false, false, 0 ), 'Entries in ''i_node'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.set_antenna_qdant( L, fn_q, 0 ), 'Input ''qdant_id'' must be between 1 and 2147483647.' );           % C++
assert_throws( @() quadriga_lib.set_antenna_qdant( L, 'no_such_file.qdant' ), 'QDANT file ''no_such_file.qdant'' does not exist or is not a file.' );   % C++
assert_throws( @() quadriga_lib.set_antenna_qdant( L, fn_q, 1, false, 3 ), 'Input ''i_node'' has an index outside the nodes of the Tx side.' );   % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.set_antenna_qdant( varargin{:} ); %#ok<ASGLU>
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
