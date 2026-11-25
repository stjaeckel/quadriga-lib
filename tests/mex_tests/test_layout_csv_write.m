% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_layout_csv_write
% MEX interface test for quadriga_lib.layout_csv_write.
% Scope: argument handling, the files that are written, the layout struct as input, error paths.
% The format, the compact form and the QDANT export are tested in the Catch2 suite (test_layout_csv.cpp, test_set_antenna.cpp).

%% Setup: a layout read from files (3 Tx nodes, 2 Rx terminals, 2 frequencies)
fn_tx = 'test_layout_csv_write_tx.csv';
fn_rx = 'test_layout_csv_write_rx.csv';
fn_q = 'test_layout_csv_write_rx_1.qdant';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx, fn_q } ) ); %#ok<NASGU>
write_text( fn_tx, [ 'name,sector,x,y,z,heading,tilt,antenna,panel,note\n', ...
    'BS1,0,0,0,25,30,10,3gpp,"2,2,1,2",a\n', ...
    'BS1,1,,,,150,,,,\n', ...
    'BS2,0,500,0,25,90,0,dipole,,b\n' ] );
write_text( fn_rx, [ 'name,x,y,z,heading,speed,direction_az,scenario,pairing\n', ...
    'UT1,10,20,1.5,45,0.833,120,UMa,BS1\n', ...
    'UT2,0,0,1.5,0,,,,\n', ...
    'UT2,1,0,1.5,,,,,\n', ...
    'UT2,2,0,1.5,90,,,,\n', ...
    'UT2,2,1,1.5,,,,UMa_NLOS,\n' ] );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
delete_files( { fn_tx, fn_rx } );

%% Basic call: reading the files gives the layout back
quadriga_lib.layout_csv_write( fn_tx, fn_rx, L );
assertTrue( isequaln( quadriga_lib.layout_csv_read( fn_tx, fn_rx ), L ) );
size_compact = numel( fileread( fn_rx ) );
assertTrue( ~isempty( strfind( fileread( fn_rx ), 'UT1,10,20,1.5,45,120,0.833,UMa,BS1' ) ) );   %#ok<STREMP> the units of the file: degrees

%% Optional inputs: compact = false writes every cell (a longer file with the same content); [] is the default
quadriga_lib.layout_csv_write( fn_tx, fn_rx, L, false );
assertTrue( numel( fileread( fn_rx ) ) > size_compact );
assertTrue( isequaln( quadriga_lib.layout_csv_read( fn_tx, fn_rx ), L ) );
quadriga_lib.layout_csv_write( fn_tx, fn_rx, L, [] );
assertEqual( numel( fileread( fn_rx ) ), size_compact );
assertTrue( isequaln( quadriga_lib.layout_csv_read( fn_tx, fn_rx ), L ) );

%% One side only: the other file is not written
delete_files( { fn_tx, fn_rx } );
quadriga_lib.layout_csv_write( '', fn_rx, L );
assertTrue( exist( fn_rx, 'file' ) == 2 && exist( fn_tx, 'file' ) == 0 );
assertEqual( numel( fileread( fn_rx ) ), size_compact );

%% Antenna arrays without a line are exported to a QDANT file next to the CSV file
ant = quadriga_lib.generate_antenna( 'omni', 10 );
Lq = quadriga_lib.set_antenna_array( L, ant, true, 2 );
quadriga_lib.layout_csv_write( fn_tx, fn_rx, Lq );
assertTrue( exist( fn_q, 'file' ) == 2 );
Lb = quadriga_lib.layout_csv_read( fn_tx, fn_rx );
assertEqual( Lb.rx.i_antenna, uint32( [ 0; 3 ] ) );
assertEqual( Lb.antenna{3}.e_theta_re, ant.e_theta_re );
quadriga_lib.layout_csv_write( fn_tx, fn_rx, L );                               % no longer referenced: removed
assertTrue( exist( fn_q, 'file' ) == 0 );

%% Errors
assert_throws( @() quadriga_lib.layout_csv_write( fn_tx, fn_rx ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.layout_csv_write( fn_tx, fn_rx, L, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out1( fn_tx, fn_rx, L ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.layout_csv_write( 1, fn_rx, L ), 'Input ''fn_tx'' must be a string' );
assert_throws( @() quadriga_lib.layout_csv_write( fn_tx, fn_rx, 1 ), 'Input ''layout'' must be a 1x1 struct.' );
Lx = L;
Lx.foo = 1;
assert_throws( @() quadriga_lib.layout_csv_write( fn_tx, fn_rx, Lx ), 'Unknown field ''foo'' in ''layout''.' );
Lx = L;
Lx.tx.name{1} = 'a,b';
assert_throws( @() quadriga_lib.layout_csv_write( fn_tx, fn_rx, Lx ), 'not valid' );                              % C++
assert_throws( @() quadriga_lib.layout_csv_write( fn_tx, fn_tx, L ), 'different' );                              % C++

end

% Request 1 output so that the output-count guard fires
function out1(varargin)
[~] = quadriga_lib.layout_csv_write( varargin{:} ); %#ok<ASGLU>
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
