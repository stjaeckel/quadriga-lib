% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_sample_track
% MEX interface test for quadriga_lib.sample_track.
% Scope: argument and output handling, 1-based indices, orientation of the outputs, error paths.
% The interpolation rules, the mounted antennas and the edge cases are tested in the Catch2 suite (test_generate_track.cpp).

%% Setup: a track A (3 rows, 1 m/s), a point drop B (heading 90 degrees, 2 m/s) and a track C without time and speed
fn_rx = 'test_sample_track_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_rx } ) ); %#ok<NASGU>
write_text( fn_rx, [ 'name,x,y,z,heading,speed\n', ...
    'A,0,0,0,0,1\n', ...
    'A,10,0,0,,\n', ...
    'A,10,10,0,,\n', ...
    'B,5,5,1.5,90,2\n', ...
    'C,0,0,0,,\n', ...
    'C,1,0,0,,\n' ] );
L = quadriga_lib.layout_csv_read( '', fn_rx );
nodes = L.rx;

%% Basic call: a track is interpolated between its rows (speed 1 m/s: the rows are at 0, 10 and 20 s)
[ pos, ori, i_row ] = quadriga_lib.sample_track( nodes, 1, [ 0, 5, 10, 15, 20 ] );
assertEqual( size( pos ), [ 3, 5 ] );
assertElementsAlmostEqual( pos, [ 0, 5, 10, 10, 10; 0, 0, 0, 5, 10; 0, 0, 0, 0, 0 ], 'absolute', 1e-12 );
assertEqual( size( ori ), [ 3, 5 ] );
assertEqual( ori, zeros( 3, 5 ) );
assertEqual( i_row, uint64( [ 1; 1; 2; 2; 3 ] ) );                              % 1-based row at or before the snapshot

%% A point drop moves along the orientation with its speed (heading 90 deg = +y)
[ pos, ori, i_row ] = quadriga_lib.sample_track( nodes, 2, [ 0, 1, 2 ] );
assertElementsAlmostEqual( pos, [ 5, 5, 5; 5, 7, 9; 1.5, 1.5, 1.5 ], 'absolute', 1e-12 );
assertElementsAlmostEqual( ori, [ 0, 0, 0; 0, 0, 0; pi / 2, pi / 2, pi / 2 ], 'absolute', 1e-14 );   % radians
assertEqual( i_row, uint64( [ 4; 4; 4 ] ) );                                    % index into the arrays of the nodes, not into the rows of the node

%% Optional times: [] gives one snapshot per row
[ pos, ~, i_row ] = quadriga_lib.sample_track( nodes, 1, [] );
assertEqual( pos, [ 0, 10, 10; 0, 0, 10; 0, 0, 0 ] );
assertEqual( i_row, uint64( [ 1; 2; 3 ] ) );

%% Outputs: none and fewer
quadriga_lib.sample_track( nodes, 1, [ 0, 5 ] );
assertEqual( ans, [ 0, 5; 0, 0; 0, 0 ] );
[ ~, ori2 ] = quadriga_lib.sample_track( nodes, 1, [ 0, 5 ] );
assertEqual( ori2, zeros( 3, 2 ) );

%% Type handling: single times as a column, uint8 index
[ pos, ~, i_row ] = quadriga_lib.sample_track( nodes, uint8( 1 ), single( [ 0; 5; 10; 15; 20 ] ) );
assertEqual( pos(1, :), [ 0, 5, 10, 10, 10 ] );
assertEqual( i_row, uint64( [ 1; 1; 2; 2; 3 ] ) );

%% Errors
assert_throws( @() quadriga_lib.sample_track( nodes, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sample_track( nodes, 1, [], 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( nodes, 1, [] ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sample_track( 1, 1, [] ), 'Input ''nodes'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.sample_track( nodes, [], [] ), 'Input ''i_node'' must not be empty.' );
assert_throws( @() quadriga_lib.sample_track( nodes, 0, [] ), 'Input ''i_node'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.sample_track( nodes, 5, [] ), 'Input ''i_node'' is out of range.' );                 % C++
assert_throws( @() quadriga_lib.sample_track( nodes, 3, [ 0, 1 ] ), 'time' );                                         % C++: no time and no speed
assert_throws( @() quadriga_lib.sample_track( nodes, 1, [ 0, NaN ] ), 'times' );                                      % C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.sample_track( varargin{:} ); %#ok<ASGLU>
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
