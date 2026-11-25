% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_generate_rays
% MEX interface test for quadriga_lib.generate_rays.
% Scope: argument and output handling, 1-based link indices, the cell of struct arrays, error paths.
% The ray statistics, the chain and the batch independence are tested in the Catch2 suite (test_init_link_states.cpp).

%% Setup: 2 Tx sites and 2 Rx terminals (4 propagation links), two bands
fn_tx = 'test_generate_rays_tx.csv';
fn_rx = 'test_generate_rays_rx.csv';
cleanup = onCleanup( @() delete_files( { fn_tx, fn_rx } ) ); %#ok<NASGU>
L = make_layout( fn_tx, fn_rx, [ 2e9, 3.5e9 ] );
L2 = quadriga_lib.init_link_states( L, '38.901', [], false, 2e9, '', 5 );
fields = { 'method', 'tx_pos', 'rx_pos', 'center_freq', 'delay', 'power', 'aod', 'eod', 'aoa', 'eoa', 'polmat', 'path_type', 'tap', ...
    'gr_eps_re', 'gr_eps_im', 'fbs_pos', 'lbs_pos', 'fallback' };

%% Basic call: a cell per link with a struct array of one entry per band (TR methods)
rays = quadriga_lib.generate_rays( L2 );
assertTrue( iscell( rays ) );
assertEqual( size( rays ), [ 4, 1 ] );
assertEqual( size( rays{1} ), [ 1, 2 ] );
assertEqual( fieldnames( rays{1} )', fields );
r = rays{1}(2);                                                                 % link 1 (site A, terminal U1), band 3.5 GHz
n_ray = numel( r.delay );
assertTrue( n_ray > 1 );
assertEqual( r.method, '38.901' );
assertEqual( r.tx_pos, [ 0; 0; 25 ] );
assertEqual( r.rx_pos, [ 100; 0; 1.5 ] );
assertEqual( r.center_freq, 3.5e9 );
assertEqual( size( r.power ), [ n_ray, 1 ] );
assertEqual( size( r.polmat ), [ 8, n_ray ] );
assertEqual( size( r.aod ), [ n_ray, 1 ] );
assertTrue( isa( r.path_type, 'uint32' ) && isa( r.tap, 'uint32' ) && isa( r.fallback, 'uint32' ) );
assertEqual( r.path_type(1), uint32( 0 ) );                                     % the LOS ray comes first, its delay is the reference
assertEqual( r.delay(1), 0 );
assertTrue( all( r.path_type <= 2 ) && all( r.power >= 0 ) );
assertTrue( isempty( r.fbs_pos ) && isempty( r.lbs_pos ) );                     % no scatterers requested
assertEqual( rays{2}(1).tx_pos, [ 1000; 0; 25 ] );                              % link 2: site B, terminal U1
assertEqual( rays{3}(1).rx_pos, [ 900; 0; 1.5 ] );                              % link 3: site A, terminal U2
assertEqual( rays{1}(1).center_freq, 2e9 );
assertEqual( rays{1}(1).path_type, r.path_type );                               % the rays of the bands share the geometry of the clusters

%% Optional inputs: 1-based links in the given order; scatterers
rs = quadriga_lib.generate_rays( L2, [ 3; 2 ] );
assertEqual( size( rs ), [ 2, 1 ] );
assertTrue( isequaln( rs{1}, rays{3} ) );                                       % the values of a link do not depend on the other links
assertTrue( isequaln( rs{2}, rays{2} ) );
rs = quadriga_lib.generate_rays( L2, [], true );
assertEqual( size( rs{1}(1).fbs_pos ), [ 3, numel( rs{1}(1).delay ) ] );
assertEqual( size( rs{1}(1).lbs_pos ), size( rs{1}(1).fbs_pos ) );
assertEqual( rs{1}(1).delay, rays{1}(1).delay );
rs = quadriga_lib.generate_rays( L2, 2, true );
assertEqual( size( rs ), [ 1, 1 ] );
assertEqual( rs{1}(2).tx_pos, [ 1000; 0; 25 ] );
rs = quadriga_lib.generate_rays( L2, [], false );
assertTrue( isequaln( rs, rays ) );

%% quadriga method: one object with all frequencies per link, with scatterers
Lq = quadriga_lib.init_link_states( L, 'quadriga', [], false, 2e9, '', 5 );
rq = quadriga_lib.generate_rays( Lq, 1 );
assertEqual( size( rq ), [ 1, 1 ] );
assertEqual( size( rq{1} ), [ 1, 1 ] );
assertEqual( rq{1}.method, 'quadriga' );
assertEqual( rq{1}.center_freq, [ 2e9; 3.5e9 ] );
assertEqual( size( rq{1}.power ), [ numel( rq{1}.delay ), 2 ] );
assertEqual( size( rq{1}.fbs_pos ), [ 3, numel( rq{1}.delay ) ] );

%% Outputs: none
quadriga_lib.generate_rays( L2, 1 );
assertTrue( isequaln( ans, rays(1) ) );

%% Type handling: int32 column and single logical
rs = quadriga_lib.generate_rays( L2, int32( [ 3; 2 ] ), single( 0 ) );
assertTrue( isequaln( rs{1}, rays{3} ) );

%% Errors
assert_throws( @() quadriga_lib.generate_rays(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.generate_rays( L2, [], false, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( L2 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.generate_rays( 1 ), 'Input ''layout'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.generate_rays( L2, 0 ), 'Entries in ''i_link'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.generate_rays( L2, 5 ), 'Input ''i_link'' has an index outside the links.' );          % C++
assert_throws( @() quadriga_lib.generate_rays( L2, [ 1, 1 ] ), 'Input ''i_link'' must not repeat a link.' );             % C++
assert_throws( @() quadriga_lib.generate_rays( L ), 'Call ''init_link_states'' first.' );                              % C++
Lm = L2;
Lm.tx.pos(1, 1) = 5;
assert_throws( @() quadriga_lib.generate_rays( Lm ), 'The layout has changed since ''init_link_states''' );            % C++

end

% Layout: BS A (origin) and B (1 km east) with the scenario UMa; UT U1 moves north at 2 m/s, UT U2 is static
function L = make_layout(fn_tx, fn_rx, freq)
write_text( fn_tx, 'name,x,y,z,scenario\nA,0,0,25,UMa\nB,1000,0,25,UMa\n' );
write_text( fn_rx, 'name,x,y,z,indoor,speed,direction_az\nU1,100,0,1.5,0,2,90\nU2,900,0,1.5,0,,\n' );
L = quadriga_lib.layout_csv_read( fn_tx, fn_rx, freq );
end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.generate_rays( varargin{:} ); %#ok<ASGLU>
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
