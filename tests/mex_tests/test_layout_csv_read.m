% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_layout_csv_read
% MEX interface test for quadriga_lib.layout_csv_read.
% Scope: argument and output handling, struct fields, index conversion, defaults, error paths.
% The format, the rules of the columns and the antenna composition are tested in the Catch2 suite (test_layout_csv.cpp).

%% Setup: 3 Tx nodes (2 sites, one with an extra sector, a 3GPP and a dipole antenna), 2 Rx terminals
fn_tx = 'test_layout_csv_read_tx.csv';
fn_rx = 'test_layout_csv_read_rx.csv';
fn_rx2 = 'test_layout_csv_read_rx2.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx, fn_rx2 } ) ); %#ok<NASGU>
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
d2r = pi / 180;

%% Basic call: fields, sizes, classes and values
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertEqual( fieldnames( L )', { 'name', 'tx', 'rx', 'links', 'seed', 'center_freq', 'antenna', 'antenna_cols', 'antenna_args', 'wrap', 'pairing' } );
assertEqual( L.name, '' );
assertEqual( L.seed, int64( -1 ) );
assertEqual( L.center_freq, [ 2e9; 3.5e9 ] );
T = L.tx;
assertEqual( T.name, { 'BS1'; 'BS1'; 'BS2' } );
assertEqual( T.sector, uint32( [ 0; 1; 0 ] ) );
assertEqual( T.pos, [ 0, 0, 500; 0, 0, 0; 25, 25, 25 ] );
assertElementsAlmostEqual( T.orientation, [ 0, 0, 0; 10, 10, 0; 30, 150, 90 ] * d2r, 'absolute', 1e-14 );   % radians
assertEqual( T.speed, zeros( 0, 0 ) );                                          % optional members stay empty
assertEqual( T.extra_names, { 'note' } );
assertEqual( T.extra, { { 'a'; ''; 'b' } } );
assertTrue( isstruct( T.scenpar_override ) );
R = L.rx;
assertEqual( R.name, { 'UT1'; 'UT2' } );
assertEqual( size( R.pos ), [ 3, 5 ] );
assertEqual( R.pos(:, 5), [ 2; 1; 1.5 ] );
assertEqual( R.speed, [ 0.833; 0; 0; 0; 0 ] );
assertEqual( size( R.direction ), [ 2, 5 ] );
assertElementsAlmostEqual( R.direction(:, 1), [ 120 * d2r; 0 ], 'absolute', 1e-14 );
assertTrue( all( isnan( R.direction(1, 2:5) ) ) );
assertEqual( R.scenario, { 'UMa'; ''; ''; ''; 'UMa_NLOS' } );
assertTrue( isa( L.links, 'struct' ) && isempty( L.links.i_tx ) && isempty( L.links.method ) );

%% Antennas: lines, indices into the lists (no shift), arrays per band
assertEqual( L.antenna_cols, { 'antenna'; 'panel' } );
assertEqual( L.antenna_args, { { '3gpp,"2,2,1,2"' }; { 'dipole,' } } );
assertEqual( T.i_antenna, uint32( [ 1; 1; 2 ] ) );                              % 1-based, 0 = generic antenna
assertEqual( R.i_antenna, uint32( [ 0; 0 ] ) );
assertEqual( size( L.antenna ), [ 2, 1 ] );
assertEqual( size( L.antenna{1} ), [ 1, 2 ] );                                  % spaced array: one per band
assertEqual( size( L.antenna{2} ), [ 1, 1 ] );                                  % co-located elements: all bands
assertEqual( size( L.antenna{1}(2).element_pos ), [ 3, 8 ] );                   % panel 2 x 2 x 1 x 2
assertEqual( L.antenna{1}(2).center_freq, 3.5e9 );

%% Indices: pairing is 1-based, (site, terminal) pairs; a terminal without entry is linked to all sites
assertEqual( L.pairing, uint64( [ 1, 1, 2; 1, 2, 2 ] ) );
assertEqual( T.offset, uint64( [ 1; 2; 3 ] ) );
assertEqual( R.offset, uint64( [ 1; 2 ] ) );

%% Optional inputs: center_freq and eval_antennas
L3 = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ], false );      % lines and indices are kept, no arrays
assertEqual( size( L3.antenna ), [ 0, 1 ] );
assertEqual( L3.antenna_args, L.antenna_args );
assertEqual( L3.tx.i_antenna, T.i_antenna );
assertEqual( quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ], true ), L );
write_text( fn_tx, 'name,x,y,z,center_freq\nA,0,0,10,1e9\nA,,,,2e9\n' );          % frequencies from the file
Lf = quadriga_lib.layout_csv_read( fn_tx, '' );
assertEqual( Lf.center_freq, [ 1e9; 2e9 ] );
Lf = quadriga_lib.layout_csv_read( fn_tx, '', [], false );
assertEqual( Lf.center_freq, [ 1e9; 2e9 ] );
Lf = quadriga_lib.layout_csv_read( fn_tx, '', [ 1e9, 2e9 ] );
assertEqual( Lf.center_freq, [ 1e9; 2e9 ] );

%% Outputs: none and fewer, extra columns
write_text( fn_tx, [ 'name,sector,x,y,z,heading,tilt,antenna,panel,note\n', ...
    'BS1,0,0,0,25,30,10,3gpp,"2,2,1,2",a\n', ...
    'BS1,1,,,,150,,,,\n', ...
    'BS2,0,500,0,25,90,0,dipole,,b\n' ] );
quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertEqual( ans, L );
[ L5, extra_tx, extra_rx ] = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertEqual( L5, L );
assertEqual( extra_tx, { 'note' } );
assertEqual( extra_rx, cell( 0, 1 ) );
[ ~, extra_tx2 ] = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertEqual( extra_tx2, extra_tx );

%% Type handling: single and column frequencies, one side only
assertEqual( quadriga_lib.layout_csv_read( fn_tx, fn_rx, single( [ 2e9; 3.5e9 ] ) ), L );
write_text( fn_rx2, 'name,x,y,z\nU1,1,2,1.5\nU2,3,4,1.5\n' );
Lr = quadriga_lib.layout_csv_read( '', fn_rx2 );
assertEqual( numel( Lr.tx.name ), 0 );
assertEqual( Lr.rx.name, { 'U1'; 'U2' } );
assertEqual( Lr.pairing, zeros( 0, 0, 'uint64' ) );                              % all pairs

%% Errors
assert_throws( @() quadriga_lib.layout_csv_read( fn_tx ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.layout_csv_read( fn_tx, fn_rx, [], true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( fn_tx, fn_rx ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.layout_csv_read( 1, fn_rx ), 'Input ''fn_tx'' must be a string' );
assert_throws( @() quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 3.5e9, 2e9 ] ), 'center_freq' );                   % C++
assert_throws( @() quadriga_lib.layout_csv_read( fn_tx, fn_rx ), 'frequenc' );                                       % C++: array spaced in wavelengths
assert_throws( @() quadriga_lib.layout_csv_read( 'test_layout_csv_read_missing.csv', '' ), 'test_layout_csv_read_missing.csv' );   % C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.layout_csv_read( varargin{:} ); %#ok<ASGLU>
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
