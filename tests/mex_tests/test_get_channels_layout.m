% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_get_channels_layout
% MEX interface test for quadriga_lib.get_channels_layout.
% Scope: argument and output handling, 1-based indices, defaults, the channel struct array, error paths.
% The pipeline, the batches and the link gain are tested in the Catch2 suite (test_get_channels_layout.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals (U1 moves north at 2 m/s), two bands; the layout has no links yet
fn_tx = 'test_get_channels_layout_tx.csv';
fn_rx = 'test_get_channels_layout_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
L = make_layout( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertTrue( isempty( L.links.i_tx ) );

%% Basic call: the link states are drawn for this call only; Tx node fastest, then Rx node, then frequency
[ ch, index ] = quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 5 );
assertEqual( size( ch ), [ 1, 8 ] );
assertEqual( { ch.name }, { 'A_U1', 'B_U1', 'A_U2', 'B_U2', 'A_U1', 'B_U1', 'A_U2', 'B_U2' } );
assertEqual( index, uint64( [ 1, 2, 1, 2, 1, 2, 1, 2; 1, 1, 2, 2, 1, 1, 2, 2; 1, 1, 1, 1, 2, 2, 2, 2 ] ) );
assertEqual( ch(1).center_freq, 2e9 );
assertEqual( ch(5).center_freq, 3.5e9 );
assertEqual( ch(2).tx_pos, [ 1000; 0; 25 ] );
assertEqual( size( ch(1).coeff_re, 1 ), 2 );                                    % generic antennas: two ports each
assertEqual( size( ch(1).coeff_re, 2 ), 2 );
assertFalse( isfield( ch, 'path_gain' ) );
assertTrue( isempty( L.links.i_tx ) );                                          % the layout is unchanged
L2 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
ch2 = quadriga_lib.get_channels_layout( L2 );                                   % the same with filled links
assertTrue( isequaln( ch2, ch ) );

%% Optional inputs: times, output, method, mobility, overlap, seed, config_dirs
ch = quadriga_lib.get_channels_layout( L2, [ 0, 1, 2 ] );
assertEqual( size( ch(1).coeff_re, 4 ), 3 );
assertElementsAlmostEqual( ch(1).rx_pos, [ 100, 100, 100; 0, 2, 4; 1.5, 1.5, 1.5 ], 'absolute', 1e-12 );   % U1 moves 2 m/s north
ch = quadriga_lib.get_channels_layout( L2, [], 'rays' );
assertTrue( isfield( ch, 'path_gain' ) );
chp = quadriga_lib.get_channels_layout( L2, [ 0, 1 ], 'taps', '38.901', 'phase', 0.2 );
assertEqual( size( chp(1).coeff_re, 4 ), 2 );
chq = quadriga_lib.get_channels_layout( L, [], 'taps', 'quadriga', 'auto', 0.5, 5 );   % another method needs empty links
assertEqual( size( chq ), [ 1, 8 ] );
chs = quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 6 );
assertFalse( isequal( chs(1).coeff_re, ch2(1).coeff_re ) );                     % another seed, other channels
chd = quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 5, {} );
assertTrue( isequaln( chd, ch2 ) );
folder = [ tempname(), '_conf' ];
mkdir( folder );
cleanup2 = onCleanup( @() rmdir( folder, 's' ) ); %#ok<NASGU>
copyfile( L2.links.conf_files{2}, fullfile( folder, 'MyUMa.conf' ) );
Lc = L;
Lc.tx.scenario = { 'MyUMa'; 'MyUMa' };
chc = quadriga_lib.get_channels_layout( Lc, [], 'taps', '', 'auto', 0.5, 5, folder );   % a user conf file in config_dirs
assertEqual( size( chc ), [ 1, 8 ] );

%% Antennas of the layout: an omni-directional Tx antenna has one port
Lo = quadriga_lib.set_antenna( L, 'omni', [], false );
ch = quadriga_lib.get_channels_layout( Lo, [], 'taps', '', 'auto', 0.5, 5 );
assertEqual( size( ch(1).coeff_re, 2 ), 1 );
assertEqual( size( ch(1).coeff_re, 1 ), 2 );

%% Pairing: 1-based (site, terminal) pairs select the channels of the output
Lp = L2;
Lp.pairing = [ 2, 1; 1, 2 ];                                                    % (B, U1) and (A, U2)
[ ch, index ] = quadriga_lib.get_channels_layout( Lp );
assertEqual( { ch.name }, { 'B_U1', 'A_U2', 'B_U1', 'A_U2' } );                       % terminal U1 first
assertEqual( index, uint64( [ 2, 1, 2, 1; 1, 2, 1, 2; 1, 1, 2, 2 ] ) );

%% Outputs: none and fewer
quadriga_lib.get_channels_layout( L2 );
assertTrue( isequaln( ans, ch2 ) );
[ ~, index2 ] = quadriga_lib.get_channels_layout( L2 );
assertEqual( index2, uint64( [ 1, 2, 1, 2, 1, 2, 1, 2; 1, 1, 2, 2, 1, 1, 2, 2; 1, 1, 1, 1, 2, 2, 2, 2 ] ) );

%% Type handling: single times as a column, int8 seed
ch = quadriga_lib.get_channels_layout( L, single( [ 0; 1; 2 ] ), 'taps', '', 'auto', 0.5, int8( 5 ) );
assertEqual( size( ch(1).coeff_re, 4 ), 3 );

%% Errors
assert_throws( @() quadriga_lib.get_channels_layout(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 5, {}, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out3( L2 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.get_channels_layout( 1 ), 'Input ''layout'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, -2 ), 'Input ''seed'' must be -1 or a non-negative integer.' );
assert_throws( @() quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 5, 5 ), 'Input ''config_dirs'' must be a cell array of strings.' );
assert_throws( @() quadriga_lib.get_channels_layout( L2, [], 'foo' ), 'Input ''output'' must be ''taps'' or ''rays''.' );               % C++
assert_throws( @() quadriga_lib.get_channels_layout( L2, [], 'taps', 'quadriga' ), 'but the links of the layout were drawn for ''38.901''' );   % C++
assert_throws( @() quadriga_lib.get_channels_layout( Lc, [], 'taps', '', 'auto', 0.5, 5 ), 'scenario ''MyUMa'' is neither a scenario family nor a conf file' );   % C++
Lp.pairing = [ 0; 1 ];
assert_throws( @() quadriga_lib.get_channels_layout( Lp ), 'Entries in ''layout.pairing'' cannot be 0 (1-based index).' );

%% A moved installation: the wrapper finds the shipped conf folder next to its own MEX file and uses it before the folder compiled in
root = tempname();
mkdir( fullfile( root, '+quadriga_lib' ) );
cleanup3 = onCleanup( @() remove_install( root ) ); %#ok<NASGU>
for name = { 'get_channels_layout', 'conf_file_list' }
    copyfile( which( [ 'quadriga_lib.', name{1} ] ), fullfile( root, '+quadriga_lib' ) );
end
mkdir( fullfile( root, 'data' ) );
copyfile( fileparts( L2.links.conf_files{2} ), fullfile( root, 'data', 'config' ) );
addpath( root );
chm = quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 5 );
assertTrue( isequaln( chm, ch2 ) );
% The shipped folder of the copy is searched first: a changed NLOS conf in it changes the channels
fn_conf = fullfile( root, 'data', 'config', '3GPP_38.901_v19_UMa_NLOS.conf' );
txt = fileread( fn_conf );
txt_new = regexprep( txt, '(?m)^DS_mu\s*=\s*-6\.47', 'DS_mu = -7.0' );
assertFalse( strcmp( txt, txt_new ) );                                          % the edit took effect (else the conf changed)
write_text( fn_conf, strrep( txt_new, '%', '%%' ) );
chn = quadriga_lib.get_channels_layout( L, [], 'taps', '', 'auto', 0.5, 5 );
assertFalse( isequaln( chn, ch2 ) );

end

% Layout: BS A (origin) and B (1 km east) with the scenario UMa; UT U1 moves north at 2 m/s, UT U2 is static
function L = make_layout(fn_tx, fn_rx, freq)
write_text( fn_tx, 'name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n' );
write_text( fn_rx, 'name,x,y,z,indoor,speed,direction_az\nU1,100,0,1.5,0,2,90\nU2,900,0,1.5,0,,\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, freq );
end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.get_channels_layout( varargin{:} ); %#ok<ASGLU>
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

% Remove a copy of the installation from the path and from the disk
function remove_install(root)
rmpath( root );
state = warning( 'off', 'all' );                                                % a loaded MEX file locks the folder on Windows
try
    rmdir( root, 's' );
catch
end
warning( state );
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
