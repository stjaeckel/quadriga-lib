% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_generate_track
% MEX interface test for quadriga_lib.generate_track.
% Scope: argument and output handling, 1-based node indices, units, defaults, error paths.
% The shape of the tracks and the random draws are tested in the Catch2 suite (test_generate_track.cpp).

%% Setup: terminal UT1 (one node) and UT2 (two sectors, one position), taken from a CSV file
fn_rx = 'test_generate_track_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_rx } ) ); %#ok<NASGU>
write_text( fn_rx, [ 'name,sector,x,y,z,speed\n', ...
    'UT1,0,0,0,1.5,3\n', ...
    'UT2,0,100,50,1.5,2\n', ...
    'UT2,1,,,,\n' ] );
L = quadriga_lib.layout_csv_read( '', fn_rx );
nodes = L.rx;
assertEqual( nodes.offset, uint64( [ 1; 2; 3 ] ) );

%% Basic call: a linear track of 10 m in the direction 0.5 rad (radians) for node 2; its sector shares the name and the track
out = quadriga_lib.generate_track( nodes, 2, 'linear', 10, 2, 0.5 );
assertEqual( out.offset, uint64( [ 1; 2; 4 ] ) );                               % 1-based
assertEqual( size( out.pos ), [ 3, 5 ] );
end_pos = [ 100 + 10 * cos( 0.5 ); 50 + 10 * sin( 0.5 ); 1.5 ];
assertElementsAlmostEqual( out.pos(:, [ 2, 3, 4, 5 ]), [ [ 100; 50; 1.5 ], end_pos, [ 100; 50; 1.5 ], end_pos ], 'absolute', 1e-12 );
assertEqual( out.pos(:, 1), nodes.pos(:, 1) );                                  % UT1 is not changed
assertElementsAlmostEqual( out.orientation(3, 2:5), 0.5 * ones( 1, 4 ), 'absolute', 1e-12 );   % the orientation follows the motion
assertElementsAlmostEqual( out.speed(2:5), 2 * ones( 4, 1 ), 'absolute', 0 );
assertEqual( nodes.offset, uint64( [ 1; 2; 3 ] ) );                             % the input stays unchanged
assertEqual( size( nodes.pos ), [ 3, 3 ] );

%% Point drop: one row with speed and an explicit direction (radians)
pt = quadriga_lib.generate_track( nodes, 1, 'point', 0, 4, 1.0 );
assertEqual( size( pt.pos ), [ 3, 3 ] );
assertEqual( pt.speed(1), 4 );
assertElementsAlmostEqual( pt.direction(:, 1), [ 1.0; 0 ], 'absolute', 1e-14 );

%% Circular track: 129 points on a circle of radius 5 m that starts in the east of its centre
ci = quadriga_lib.generate_track( nodes, 1, 'circular', 2 * pi * 5, 1, 0 );
assertEqual( size( ci.pos ), [ 3, 129 + 2 ] );
assertElementsAlmostEqual( ci.pos(:, 1), ci.pos(:, 129), 'absolute', 1e-12 );   % closed
dist = sqrt( ( ci.pos(1, 1:129) + 5 ).^2 + ci.pos(2, 1:129).^2 );               % centre at (-5, 0): the start (0, 0) is east of it
assertElementsAlmostEqual( dist, 5 * ones( 1, 129 ), 'absolute', 1e-12 );
assertEqual( ci.pos(3, 1:129), 1.5 * ones( 1, 129 ) );

%% Street: reproducible with a seed; options; no turns give a straight route
s1 = quadriga_lib.generate_track( nodes, 1, 'street', 200, 5, [], NaN, [], 3 );
s2 = quadriga_lib.generate_track( nodes, 1, 'street', 200, 5, [], 1, struct(), 3 );
assertEqual( s1.pos, s2.pos );                                                  % spacing 1 m is the default
s3 = quadriga_lib.generate_track( nodes, 1, 'street', 200, 5, [], NaN, [], 4 );
assertFalse( isequal( s1.pos, s3.pos ) );
st = quadriga_lib.generate_track( nodes, 1, 'street', 200, 5, 0.3, NaN, struct( 'turn_probability', 0 ), 3 );
dx = st.pos(1, 1:201) - st.pos(1, 1);
dy = st.pos(2, 1:201) - st.pos(2, 1);
assertElementsAlmostEqual( dx * sin( 0.3 ) - dy * cos( 0.3 ), zeros( 1, 201 ), 'absolute', 1e-9 );
assertElementsAlmostEqual( sqrt( dx(end)^2 + dy(end)^2 ), 200, 'absolute', 1e-9 );

%% Indices: several nodes, a vector of directions per node, other classes
two = quadriga_lib.generate_track( nodes, uint8( [ 1; 3 ] ), 'linear', 10, [], [ 0; 0.5 ] );   % node 3 = sector of UT2: both UT2 nodes
assertEqual( two.offset, uint64( [ 1; 3; 5 ] ) );
assertElementsAlmostEqual( two.pos(:, 2), [ 10; 0; 1.5 ], 'absolute', 1e-12 );
assertElementsAlmostEqual( two.speed(1:2), [ 3; 3 ], 'absolute', 0 );           % speed = [] : the speed of the first row

%% Outputs: none
quadriga_lib.generate_track( nodes, 2, 'linear', 10, 2, 0.5 );
assertEqual( ans, out );

%% Errors
assert_throws( @() quadriga_lib.generate_track( nodes, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.generate_track( nodes, 1, 'linear', 1, 1, 0, 1, [], 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( nodes, 1, 'linear' ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.generate_track( 1, 1, 'linear' ), 'Input ''nodes_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.generate_track( nodes, 0, 'linear' ), 'Entries in ''i_node'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.generate_track( nodes, 1, 'linear', 1, 1, 0, 1, [], -2 ), 'Input ''seed'' must be -1 or a non-negative integer.' );
assert_throws( @() quadriga_lib.generate_track( nodes, 4, 'linear', 10, 1, 0 ), 'Input ''i_node'' is out of range.' );                  % C++
assert_throws( @() quadriga_lib.generate_track( nodes, 1, 'spiral', 10, 1, 0 ), 'spiral' );                                % C++
assert_throws( @() quadriga_lib.generate_track( nodes, 1, 'linear', 10, 1, 0, NaN, struct( 'foo', 1 ) ), 'foo' );          % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.generate_track( varargin{:} ); %#ok<ASGLU>
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
