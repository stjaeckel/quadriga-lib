% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_set_orientation
% MEX interface test for quadriga_lib.set_orientation.
% Scope: argument and output handling, 1-based node indices, units, defaults, error paths.
% The rules (tilt, closed tracks, relative nodes) are tested in the Catch2 suite (test_generate_track.cpp).

%% Setup: a track T1 along +x and then +y (3 rows) and a point drop P with heading 30 degrees
fn_rx = 'test_set_orientation_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_rx } ) ); %#ok<NASGU>
write_text( fn_rx, [ 'name,x,y,z,heading\n', ...
    'T1,0,0,0,0\n', ...
    'T1,10,0,0,\n', ...
    'T1,10,10,0,\n', ...
    'P,0,0,0,30\n' ] );
L = quadriga_lib.layout_csv_read( '', fn_rx );
nodes = L.rx;
h30 = 30 * pi / 180;
assertElementsAlmostEqual( nodes.orientation, [ 0, 0, 0, 0; 0, 0, 0, 0; 0, 0, 0, h30 ], 'absolute', 1e-14 );

%% Basic call: the orientation follows the motion of tracks (heading from the steps, the last step repeated)
out = quadriga_lib.set_orientation( nodes );
assertElementsAlmostEqual( out.orientation, [ 0, 0, 0, 0; 0, 0, 0, 0; 0, pi / 2, pi / 2, h30 ], 'absolute', 1e-14 );
assertEqual( nodes.orientation(3, 2), 0 );                                      % the input stays unchanged
assertEqual( out.pos, nodes.pos );
assertEqual( out.offset, uint64( [ 1; 4 ] ) );                                  % 1-based

%% Optional inputs: the orientation (bank, tilt, heading in rad) is an offset; i_node (1-based) selects nodes; force
out = quadriga_lib.set_orientation( nodes, [ 0, 0, 0.5 ] );
assertElementsAlmostEqual( out.orientation(3, :), [ 0.5, pi / 2 + 0.5, pi / 2 + 0.5, h30 + 0.5 ], 'absolute', 1e-14 );
out = quadriga_lib.set_orientation( nodes, [], 1 );                             % T1 only
assertElementsAlmostEqual( out.orientation(3, :), [ 0, pi / 2, pi / 2, h30 ], 'absolute', 1e-14 );
out = quadriga_lib.set_orientation( nodes, [ 0.1, 0.2, 0.3 ], 2, true );        % P only, set to the given value
assertEqual( out.orientation(:, 4), [ 0.1; 0.2; 0.3 ] );
assertEqual( out.orientation(:, 1:3), nodes.orientation(:, 1:3) );
out = quadriga_lib.set_orientation( nodes, [ 0.1, 0.2, 0.3 ], [], true );
assertEqual( out.orientation, repmat( [ 0.1; 0.2; 0.3 ], 1, 4 ) );
out = quadriga_lib.set_orientation( nodes, [], [], false );
assertElementsAlmostEqual( out.orientation(3, :), [ 0, pi / 2, pi / 2, h30 ], 'absolute', 1e-14 );

%% Outputs: none
quadriga_lib.set_orientation( nodes );
assertElementsAlmostEqual( ans.orientation(3, :), [ 0, pi / 2, pi / 2, h30 ], 'absolute', 1e-14 );

%% Type handling: single orientation as a column, uint8 index
out = quadriga_lib.set_orientation( nodes, single( [ 0; 0; 0.5 ] ), uint8( [ 1; 2 ] ) );
assertElementsAlmostEqual( out.orientation(3, :), [ 0.5, pi / 2 + 0.5, pi / 2 + 0.5, h30 + 0.5 ], 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.set_orientation(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.set_orientation( nodes, [], [], false, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( nodes ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.set_orientation( 1 ), 'Input ''nodes_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.set_orientation( nodes, [], 0 ), 'Entries in ''i_node'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.set_orientation( nodes, [], 3 ), 'Input ''i_node'' is out of range.' );              % C++
assert_throws( @() quadriga_lib.set_orientation( nodes, [ 0, 1 ] ), 'orientation' );                                 % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.set_orientation( varargin{:} ); %#ok<ASGLU>
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
