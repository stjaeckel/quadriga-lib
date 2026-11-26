% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_get_channels_from_rays
% MEX interface test for quadriga_lib.get_channels_from_rays.
% Scope: argument and output handling, 1-based indices, the channel struct array, the rays as input, error paths.
% The link gain, the segments and the mobility are tested in the Catch2 suite (test_init_link_states.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals (U1 moves north at 2 m/s), two bands, the rays of the 4 links
fn_tx = 'test_get_channels_from_rays_tx.csv';
fn_rx = 'test_get_channels_from_rays_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
L = make_layout( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
L2 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
rays = quadriga_lib.generate_rays( L2 );

%% Basic call: Tx node fastest, then Rx node, then frequency; 1-based indices
[ ch, index ] = quadriga_lib.get_channels_from_rays( L2, rays );
assertEqual( size( ch ), [ 1, 8 ] );
assertEqual( { ch.name }, { 'A_U1', 'B_U1', 'A_U2', 'B_U2', 'A_U1', 'B_U1', 'A_U2', 'B_U2' } );
assertEqual( index, uint64( [ 1, 2, 1, 2, 1, 2, 1, 2; 1, 1, 2, 2, 1, 1, 2, 2; 1, 1, 1, 1, 2, 2, 2, 2 ] ) );
assertEqual( ch(1).center_freq, 2e9 );
assertEqual( ch(5).center_freq, 3.5e9 );
assertEqual( ch(2).tx_pos, [ 1000; 0; 25 ] );
assertEqual( ch(3).rx_pos, [ 900; 0; 1.5 ] );
n_path = size( ch(1).coeff_re, 3 );
assertEqual( size( ch(1).coeff_re ), [ 2, 2, n_path ] );                        % the generic antenna has two ports; 1 snapshot
assertEqual( size( ch(1).delay ), [ 2, 2, n_path ] );
assertFalse( isfield( ch, 'path_gain' ) );                                      % taps
assertEqual( ch(1).delay(1, 1, 1), 0 );                                         % the LOS tap is the delay reference
assertTrue( any( ch(1).delay(:) > 0 ) && all( isfinite( ch(1).coeff_re(:) ) ) );

%% Rays and taps of one call are those of the single call of all stages
[ ch2, index2 ] = quadriga_lib.get_channels_layout( L2 );
assertTrue( isequaln( ch2, ch ) );
assertEqual( index2, index );

%% Optional inputs: i_link (1-based, the links of rays), times, output, mobility, oxygen, use_absolute_delays, overlap
[ ch, index ] = quadriga_lib.get_channels_from_rays( L2, rays(1:2), [ 1; 2 ] );        % links 1 and 2: terminal U1 with both sites
assertEqual( size( ch ), [ 1, 4 ] );
assertEqual( index, uint64( [ 1, 2, 1, 2; 1, 1, 1, 1; 1, 1, 2, 2 ] ) );
ch = quadriga_lib.get_channels_from_rays( L2, rays([ 3, 1 ]), [ 3, 1 ] );              % links in another order: the channels follow the nodes
assertEqual( { ch.name }, { 'A_U1', 'A_U2', 'A_U1', 'A_U2' } );
ch = quadriga_lib.get_channels_from_rays( L2, rays, [], [ 0, 1, 2 ] );                % U1 moves 2 m/s north: positions at 0, 1, 2 s
assertEqual( size( ch(1).coeff_re, 4 ), 3 );
assertElementsAlmostEqual( ch(1).rx_pos, [ 100, 100, 100; 0, 2, 4; 1.5, 1.5, 1.5 ], 'absolute', 1e-12 );
assertEqual( ch(3).rx_pos, repmat( [ 900; 0; 1.5 ], 1, 3 ) );                    % a static terminal is repeated
ch = quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'rays' );
assertTrue( isfield( ch, 'path_gain' ) );
assertEqual( size( ch(1).path_gain, 1 ), size( ch(1).coeff_re, 3 ) );
rs = quadriga_lib.generate_rays( L2, [], true );
chd = quadriga_lib.get_channels_from_rays( L2, rs, [], [ 0, 1, 2 ], 'taps', 'drift' );  % needs the scatterers
assertEqual( size( chd(1).coeff_re, 4 ), 3 );
chp = quadriga_lib.get_channels_from_rays( L2, rs, [], [ 0, 1, 2 ], 'taps', 'phase' );
assertEqual( size( chp(1).coeff_re ), size( chd(1).coeff_re ) );
cho = quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'taps', 'auto', true );   % oxygen absorption
assertEqual( size( cho(1).coeff_re ), [ 2, 2, n_path ] );
cha = quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'taps', 'auto', false, true );   % LOS delay in all delays
dist = sqrt( 100^2 + 23.5^2 );
assertElementsAlmostEqual( cha(1).delay(1, 1, 1), dist / 299792458, 'absolute', 1e-15 );
chv = quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'taps', 'auto', false, false, 0.2 );   % overlap
assertEqual( size( chv ), [ 1, 8 ] );
chg = quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'taps', 'auto', false, false, 0.5, false );   % without link gain: far more power
assertTrue( sum( chg(1).coeff_re(:).^2 + chg(1).coeff_im(:).^2 ) > 1e4 * sum( ch2(1).coeff_re(:).^2 + ch2(1).coeff_im(:).^2 ) );
assertTrue( isequaln( quadriga_lib.get_channels_layout( L2, [], 'taps', '', 'auto', 0.5, -1, {}, false ), chg ) );
L2p = L2;
L2p.pairing = [ 2; 1 ];                                                         % only the pair (site B, terminal U1): 1-based
[ ch, index ] = quadriga_lib.get_channels_from_rays( L2p, rays );
assertEqual( { ch.name }, { 'B_U1', 'B_U1' } );
assertEqual( index, uint64( [ 2, 2; 1, 1; 1, 2 ] ) );

%% Edited rays: powers of 0 give channels without power
re = rays;
re{1}(1).power(:) = 0;
ch = quadriga_lib.get_channels_from_rays( L2, re, [], [], 'rays' );
assertEqual( ch(1).path_gain, zeros( size( ch(1).path_gain ) ) );
assertTrue( any( ch(3).path_gain(:) > 0 ) );

%% Outputs: none and fewer
quadriga_lib.get_channels_from_rays( L2, rays );
assertTrue( isequaln( ans, ch2 ) );
chn = quadriga_lib.get_channels_from_rays( L2, rays );
assertTrue( isequaln( chn, ch2 ) );

%% Type handling: int32 links, single times as a column, rays with single entries
ch = quadriga_lib.get_channels_from_rays( L2, rays(1:2), int32( [ 1; 2 ] ), single( [ 0; 1; 2 ] ) );
assertEqual( size( ch ), [ 1, 4 ] );
assertEqual( size( ch(1).coeff_re, 4 ), 3 );
rsg = rays;
rsg{1}(1).power = single( rsg{1}(1).power );
rsg{1}(1).path_type = int8( rsg{1}(1).path_type );
ch = quadriga_lib.get_channels_from_rays( L2, rsg );
assertEqual( size( ch ), [ 1, 8 ] );

%% Errors
assert_throws( @() quadriga_lib.get_channels_from_rays( L2 ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'taps', 'auto', false, false, 0.5, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out3( L2, rays ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.get_channels_from_rays( 1, rays ), 'Input ''layout'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, 5 ), 'Input ''rays'' must be a cell array of struct arrays.' );
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, { 1 } ), 'Input ''rays{1}'' must be a struct.' );
bad = rays;
bad{1} = rmfield( bad{1}, 'delay' );
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, bad ), 'Field ''delay'' is missing in ''rays{1}''.' );
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, rays, 0 ), 'Entries in ''i_link'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, rays, 1 ), 'Input ''rays'' must have one entry per link of ''i_link''.' );               % C++
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'foo' ), 'Input ''output'' must be ''taps'' or ''rays''.' );               % C++
assert_throws( @() quadriga_lib.get_channels_from_rays( L2, rays, [], [], 'taps', 'drift' ), 'Mobility ''drift'' needs ''rays.fbs_pos''' );          % C++
assert_throws( @() quadriga_lib.get_channels_from_rays( L, rays ), 'Call ''init_link_states'' first.' );                                          % C++
L2p.pairing = [ 0; 1 ];
assert_throws( @() quadriga_lib.get_channels_from_rays( L2p, rays ), 'Entries in ''layout.pairing'' cannot be 0 (1-based index).' );

end

% Layout: BS A (origin) and B (1 km east) with the scenario UMa; UT U1 moves north at 2 m/s, UT U2 is static
function L = make_layout(fn_tx, fn_rx, freq)
write_text( fn_tx, 'name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n' );
write_text( fn_rx, 'name,x,y,z,indoor,speed,direction_az\nU1,100,0,1.5,0,2,90\nU2,900,0,1.5,0,,\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, freq );
end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.get_channels_from_rays( varargin{:} ); %#ok<ASGLU>
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
