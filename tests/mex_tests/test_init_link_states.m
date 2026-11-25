% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_init_link_states
% MEX interface test for quadriga_lib.init_link_states.
% Scope: argument and output handling, 1-based link indices, the fields of the links, conf folders, error paths.
% The draws, the segments and the random streams are tested in the Catch2 suite (test_init_link_states.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals (outdoor), two bands
fn_tx = 'test_init_link_states_tx.csv';
fn_rx = 'test_init_link_states_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
L = make_layout( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
assertTrue( isempty( L.links.i_tx ) );

%% Basic call: one link per site and terminal, terminal-major, 1-based indices
L2 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
k = L2.links;
assertEqual( L2.seed, int64( 5 ) );
assertEqual( k.seed, int64( 5 ) );
assertEqual( k.i_tx, uint32( [ 1; 2; 1; 2 ] ) );
assertEqual( k.i_rx, uint32( [ 1; 1; 2; 2 ] ) );
assertEqual( k.i_segment, uint32( [ 1; 1; 1; 1 ] ) );
assertEqual( k.image, zeros( 4, 1, 'uint32' ) );
assertEqual( k.conf_names, { 'UMa_LOS'; 'UMa_NLOS'; 'UMa_LOS_O2I'; 'UMa_NLOS_O2I' } );
assertTrue( isa( k.i_conf, 'uint32' ) && all( k.i_conf >= 1 & k.i_conf <= 4 ) );   % 1-based index into conf_names
for l = 1 : 4                                                                    % the LOS state selects the conf; indoor = 0: no O2I conf
    name = k.conf_names{ k.i_conf(l) };
    assertEqual( ~isempty( regexp( name, '_LOS$', 'once' ) ), logical( k.los(l) ) );
    assertEqual( ~isempty( regexp( name, 'O2I', 'once' ) ), false );

end
assertEqual( k.indoor, zeros( 4, 1, 'int32' ) );                                % fixed by the layout
assertEqual( k.floor_id, zeros( 4, 1, 'uint32' ) );
assertEqual( k.d2d_in, zeros( 4, 1 ) );
assertEqual( size( k.o2i_rnd ), [ 2, 4 ] );                                     % one row per terminal
assertTrue( isa( k.o2i_model, 'uint32' ) && numel( k.o2i_model ) == 2 && all( k.o2i_model >= 1 ) );   % 1-based
assertEqual( class( k.gbsm_seed ), 'int64' );
assertEqual( k.method, '38.901' );
assertEqual( k.sos_preset, 'Exp256' );
assertEqual( k.anchor_freq, 2e9 );
assertEqual( k.pol_variability, false );
assertEqual( numel( k.conf_files ), 4 );
assertEqual( exist( k.conf_files{2}, 'file' ), 2 );                             % the shipped folder was found
assertEqual( class( k.layout_hash ), 'uint64' );
assertTrue( k.layout_hash ~= 0 );
assertEqual( L2.tx, L.tx );                                                     % the rest is unchanged
assertTrue( isempty( L.links.i_tx ) );                                          % the input stays unchanged

%% The links are a function of the seed; a drawn seed is stored
L3 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
assertTrue( isequaln( L3, L2 ) );
L4 = quadriga_lib.init_link_states( L );
assertTrue( L4.seed >= 0 );
assertEqual( L4.links.seed, L4.seed );
L5 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', L4.seed );
assertTrue( isequaln( L5.links, L4.links ) );
L6 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 6 );
assertFalse( isequal( L6.links.gbsm_seed, L2.links.gbsm_seed ) );

%% Optional inputs: method, o2i_low, pol_variability, anchor_freq, sos_preset
L7 = quadriga_lib.init_link_states( L, '38.901_MF', 0.5, true, 3e9, 'Comb256', 5 );
assertEqual( L7.links.method, '38.901_MF' );
assertEqual( L7.links.pol_variability, true );
assertEqual( L7.links.anchor_freq, 3e9 );
assertEqual( L7.links.sos_preset, 'Comb256' );
L8 = quadriga_lib.init_link_states( L, 'quadriga', [], false, 2e9, '', 5 );
assertEqual( L8.links.method, 'quadriga' );
assertEqual( L8.links.sos_preset, 'Comb256' );                                  % the default of this method
assertEqual( L2.links.sos_preset, 'Exp256' );

%% Conf folders: a user conf file is found in config_dirs (string or cell array), the shipped folder is added by the wrapper
folder = [ tempname(), '_conf' ];
mkdir( folder );
cleanup2 = onCleanup( @() rmdir( folder, 's' ) ); %#ok<NASGU>
copyfile( k.conf_files{2}, fullfile( folder, 'MyUMa.conf' ) );
Lc = L;
Lc.tx.scenario = { 'MyUMa'; 'MyUMa' };
Lc2 = quadriga_lib.init_link_states( Lc, '38.901', [], false, 2e9, '', 5, folder );
assertEqual( Lc2.links.conf_names, { 'MyUMa' } );
assertEqual( numel( Lc2.links.conf_files ), 1 );
assertEqual( exist( Lc2.links.conf_files{1}, 'file' ), 2 );
assertTrue( ~isempty( strfind( Lc2.links.conf_files{1}, 'MyUMa.conf' ) ) );     %#ok<STREMP>
Lc3 = quadriga_lib.init_link_states( Lc, '38.901', [], false, 2e9, '', 5, { 'does_not_exist_folder', folder } );
assertTrue( isequaln( Lc3, Lc2 ) );
Lc4 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5, {} );   % the scenario family UMa needs the shipped folder only
assertTrue( isequaln( Lc4, L2 ) );

%% Outputs: none
quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
assertTrue( isequaln( ans, L2 ) );

%% Type handling: single and uint8 arguments
L9 = quadriga_lib.init_link_states( L, '38.901', single( [] ), 0, single( 2e9 ), '', uint8( 5 ) );
assertTrue( isequaln( L9, L2 ) );

%% Errors
assert_throws( @() quadriga_lib.init_link_states(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5, {}, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( L ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.init_link_states( 1 ), 'Input ''layout_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', -2 ), 'Input ''seed'' must be -1 or a non-negative integer.' );
assert_throws( @() quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5, 5 ), 'Input ''config_dirs'' must be a cell array of strings.' );
assert_throws( @() quadriga_lib.init_link_states( L, 'foo' ), 'Input ''method'' must be ''38.901'', ''38.901_MF'' or ''quadriga''.' );   % C++
assert_throws( @() quadriga_lib.init_link_states( L, '38.901', 2 ), 'Input ''o2i_low'' must be empty or one value in [0, 1].' );   % C++
assert_throws( @() quadriga_lib.init_link_states( Lc, '38.901', [], false, 2e9, '', 5 ), 'scenario ''MyUMa'' is neither a scenario family nor a conf file' );   % C++
L0 = quadriga_lib.layout_csv_read( fn_tx, fn_rx );
assert_throws( @() quadriga_lib.init_link_states( L0 ), 'Member ''center_freq'' of ''layout'' must not be empty' );   % C++

%% A moved installation: the wrapper finds the shipped conf folder next to its own MEX file and uses it before the folder compiled in
root = tempname();
mkdir( fullfile( root, '+quadriga_lib' ) );
cleanup3 = onCleanup( @() remove_install( root ) ); %#ok<NASGU>
for name = { 'init_link_states', 'conf_file_list' }
    copyfile( which( [ 'quadriga_lib.', name{1} ] ), fullfile( root, '+quadriga_lib' ) );
end
mkdir( fullfile( root, 'data' ) );
copyfile( fileparts( k.conf_files{2} ), fullfile( root, 'data', 'config' ) );
addpath( root );
Lm = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
assertEqual( strncmp( Lm.links.conf_files{2}, root, numel( root ) ), true );
assertTrue( isequaln( Lm.links.conf_names, L2.links.conf_names ) );
assertEqual( Lm.links.i_conf, L2.links.i_conf );

end

% Layout: BS A (origin) and B (1 km east) with the scenario UMa; UT U1 moves north at 2 m/s, UT U2 is static
function L = make_layout(fn_tx, fn_rx, freq)
write_text( fn_tx, 'name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n' );
write_text( fn_rx, 'name,x,y,z,indoor,speed,direction_az\nU1,100,0,1.5,0,2,90\nU2,900,0,1.5,0,,\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, freq );
end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.init_link_states( varargin{:} ); %#ok<ASGLU>
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
