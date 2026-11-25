% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_export_rays_qrt
% MEX interface test for quadriga_lib.export_rays_qrt.
% Scope: argument and output handling, the file that is written, the rays as input, error paths.
% The content of the file and its evaluation are tested in the Catch2 suite (test_export_rays_qrt.cpp).

%% Setup: 2 static Tx sites (0 m and 1 km) and 2 Rx terminals (100 m and 900 m), one band
fn_tx = 'test_export_rays_qrt_tx.csv';
fn_rx = 'test_export_rays_qrt_rx.csv';
fn_q1 = 'test_export_rays_qrt_1.qrt';
fn_q2 = 'test_export_rays_qrt_2.qrt';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx, fn_q1, fn_q2, [ fn_q1, '.part' ] } ) ); %#ok<NASGU>
write_text( fn_tx, 'name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n' );
write_text( fn_rx, 'name,x,y,z,indoor\nU1,100,0,1.5,0\nU2,900,0,1.5,0\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, 3.5e9 );
L2 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );

%% Basic call: the shift of the local coordinates (centroid of the sites and the CIR positions in x and y)
shift = quadriga_lib.export_rays_qrt( fn_q1, L2 );
assertEqual( size( shift ), [ 2, 1 ] );
assertElementsAlmostEqual( shift, [ 500; 0 ], 'absolute', 1e-9 );
assertEqual( exist( fn_q1, 'file' ), 2 );
assertEqual( exist( [ fn_q1, '.part' ], 'file' ), 0 );
[ center_freq, tx_pos, ~, rx_pos ] = quadriga_lib.qrt_file_read( fn_q1, [], 1 );   % origin 1: site A
assertEqual( center_freq, 3.5e9 );
assertElementsAlmostEqual( tx_pos, [ -500, -500; 0, 0; 25, 25 ], 'absolute', 1e-4 );
assertElementsAlmostEqual( rx_pos, [ -400, 400; 0, 0; 1.5, 1.5 ], 'absolute', 1e-4 );   % one CIR per terminal; add the shift back
[ ~, tx_pos2 ] = quadriga_lib.qrt_file_read( fn_q1, [], 2 );                    % origin 2: site B
assertElementsAlmostEqual( tx_pos2, [ 500, 500; 0, 0; 25, 25 ], 'absolute', 1e-4 );

%% Optional inputs: the rays of all links from generate_rays give the same file
rays = quadriga_lib.generate_rays( L2 );
shift2 = quadriga_lib.export_rays_qrt( fn_q2, L2, rays );
assertEqual( shift2, shift );
[ ~, ~, ~, ~, ~, ~, ~, pg1, pl1 ] = quadriga_lib.qrt_file_read( fn_q1 );
[ ~, ~, ~, ~, ~, ~, ~, pg2, pl2 ] = quadriga_lib.qrt_file_read( fn_q2 );
assertEqual( pg2, pg1 );
assertEqual( pl2, pl1 );
quadriga_lib.export_rays_qrt( fn_q2, L2, [] );                                  % [] = generated
[ ~, ~, ~, ~, ~, ~, ~, pg3 ] = quadriga_lib.qrt_file_read( fn_q2 );
assertEqual( pg3, pg1 );
re = rays;
for l = 1 : 4
    re{l}.power(2:end) = 0;                                                     % an edited ray set: only the LOS ray (no power) and the first NLOS ray stay
end
quadriga_lib.export_rays_qrt( fn_q2, L2, re );
[ ~, ~, ~, ~, ~, ~, ~, pg4 ] = quadriga_lib.qrt_file_read( fn_q2 );
assertFalse( isequal( pg4, pg1 ) );

%% Outputs: none gives the shift as ans
quadriga_lib.export_rays_qrt( fn_q2, L2 );
assertEqual( ans, shift );

%% Type handling: the rays as a row cell array
assertEqual( quadriga_lib.export_rays_qrt( fn_q2, L2, rays' ), shift );

%% Errors
assert_throws( @() quadriga_lib.export_rays_qrt( fn_q1 ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.export_rays_qrt( fn_q1, L2, rays, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( fn_q1, L2 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.export_rays_qrt( 1, L2 ), 'Input ''fn'' must be a string' );
assert_throws( @() quadriga_lib.export_rays_qrt( fn_q1, 1 ), 'Input ''layout'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.export_rays_qrt( fn_q1, L2, 5 ), 'Input ''rays'' must be a cell array of struct arrays.' );
assert_throws( @() quadriga_lib.export_rays_qrt( fn_q1, L ), 'Call ''init_link_states'' first.' );                      % C++
Lf = quadriga_lib.layout_csv_read( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
Lf = quadriga_lib.init_link_states( Lf, '38.901', [], false, 2e9, '', 5 );
assert_throws( @() quadriga_lib.export_rays_qrt( fn_q1, Lf ), 'exports one frequency' );                                % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.export_rays_qrt( varargin{:} ); %#ok<ASGLU>
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
