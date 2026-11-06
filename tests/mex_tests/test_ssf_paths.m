function test_ssf_paths
% MEX interface test for quadriga_lib.ssf_paths.
% Scope: argument and output handling, optional inputs, index conversion, error paths.
% Physics validation lives in the Catch2 suite (test_ssf_paths.cpp).

%% Setup: UMi LOS with ground reflection, '38.901' rays of one link at 60 GHz, solved
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
sp.GR_enabled = 1;
tx_pos = [ 0, 0, 10 ];
rx_pos = [ 60, 10, 1.5 ];
fc = 60e9;
phase = quadriga_lib.gbsm_init_phases( sp, 'lsf', 5 );
[ lsp, ~, gre, gim, ~, mu, sig ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, ...
    quadriga_lib.gbsm_draw( sp, 'lsf', phase, tx_pos, rx_pos ) );
rnd = quadriga_lib.gbsm_draw( sp, 'cluster', quadriga_lib.gbsm_init_phases( sp, 'cluster', 5 ), tx_pos, rx_pos );
rf = quadriga_lib.gbsm_draw( sp, 'cluster', quadriga_lib.gbsm_init_phases( sp, 'cluster', 5, 1 ), tx_pos, rx_pos, 1 );
rr = quadriga_lib.gbsm_draw( sp, 'ray', quadriga_lib.gbsm_init_phases( sp, 'ray', 5, 1 ), tx_pos, rx_pos, 1 );
[ d, p, aod, eod, aoa, eoa ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, '38.901', rf, [], [], [], ...
    gre, gim );
rays = quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc, lsp, mu, sig, d, p, aod, eod, aoa, eoa, rr, '38.901', 1, ...
    [], [], gre, gim );
rays = quadriga_lib.ssf_scatterers( rays );
kept = find( any( rays.power > 0, 2 ) | rays.path_type < 2 );                 % LOS, GR and NLOS rays with power
n_path = numel( kept );

%% Basic call and values ('phase', the default of '38.901')
[ pg, pl, M, aod_p, eod_p, aoa_p, eoa_p, fbs, lbs, pt, ri ] = quadriga_lib.ssf_paths( rays, -90 );
assertEqual( size(pg), [n_path, 1] );
assertEqual( size(pl), [n_path, 1] );
assertEqual( size(M), [8, n_path] );
assertEqual( size(aod_p), [n_path, 1] );
assertEqual( size(eoa_p), [n_path, 1] );
assertEqual( size(fbs), [3, 0] );
assertEqual( size(lbs), [3, 0] );
assertTrue( isa(pt, 'uint64') && isa(ri, 'uint64') && isa(pg, 'double') );
assertEqual( ri, uint64( kept ) );                                            % 1-based (+1)
assertEqual( ri(1:2), uint64( [1; 2] ) );
assertEqual( pt, rays.path_type( ri ) );                                      % codes, not shifted
assertEqual( pt(1:3), uint64( [0; 1; 2] ) );
assertEqual( rays.tap( ri(1:2) ), uint64( [0; 3] ) );
assertElementsAlmostEqual( pg, rays.power( kept ) * 1e-9, 'absolute', 1e-20 );
assertElementsAlmostEqual( pl, norm( rx_pos - tx_pos ) + 299792458 * rays.delay( kept ), 'absolute', 1e-9 );
assertEqual( M, rays.M( :, kept ) );                                          % no displacement
assertEqual( aod_p, rays.aod( kept ) );
assertEqual( eoa_p, rays.eoa( kept ) );

%% Optional inputs: mobility, tx_pos, rx_pos, kf_ratio, oxygen
[ g2, l2, M2 ] = quadriga_lib.ssf_paths( rays, -90, [], [], [], [], [] );
assertEqual( g2, pg );
assertEqual( M2, M );
[ g2, l2, M2, ~, ~, ~, ~, f2, b2, ~, r2 ] = quadriga_lib.ssf_paths( rays, -90, 'drift' );
assertEqual( size(f2), [3, n_path] );
assertEqual( f2, rays.fbs_pos( :, r2 ) );
assertEqual( b2, rays.lbs_pos( :, r2 ) );
assertTrue( max( abs( l2 - pl ) ) > 1e-6 );                                   % fallback rays: longer polyline
[ g3, l3, M3 ] = quadriga_lib.ssf_paths( rays, -90, 'phase', tx_pos, rx_pos + [ 0.01, 0, 0 ] );
assertEqual( g3, pg );
assertEqual( l3, pl );
assertTrue( max( abs( M3(:) - M(:) ) ) > 1e-3 );                              % displacement phase
[ ~, ~, M4 ] = quadriga_lib.ssf_paths( rays, -90, 'phase', tx_pos', ( rx_pos + [ 0.01, 0, 0 ] )' );
assertEqual( M4, M3 );                                                        % row or column
[ ~, ~, M4 ] = quadriga_lib.ssf_paths( rays, -90, 'phase', [], rx_pos + [ 0.01, 0, 0 ] );
assertEqual( M4, M3 );
g4 = quadriga_lib.ssf_paths( rays, -90, [], [], [], 0 );
assertEqual( g4(1:2), [0; 0] );                                               % kf_ratio 0: no LOS and GR
assertTrue( all( g4(3:end) >= pg(3:end) ) );
g5 = quadriga_lib.ssf_paths( rays, -90, [], [], [], [], true );
assertTrue( all( g5 < pg ) );                                                 % oxygen at 60 GHz
g6 = quadriga_lib.ssf_paths( rays, -90, [], [], [], [], 1 );
assertEqual( g6, g5 );
g7 = quadriga_lib.ssf_paths( rays, -80 );
assertElementsAlmostEqual( g7, pg * 10, 'absolute', 1e-18 );

%% Outputs: 0 to 11, fewer outputs give the same values
quadriga_lib.ssf_paths( rays, -90 );
[ g8, l8 ] = quadriga_lib.ssf_paths( rays, -90 );
assertEqual( g8, pg );
assertEqual( l8, pl );
[ ~, ~, ~, ~, ~, ~, ~, ~, ~, p8 ] = quadriga_lib.ssf_paths( rays, -90 );
assertEqual( p8, pt );

%% Type handling: single link gain, struct with a double 'path_type' and without optional fields
s = rmfield( rays, { 'gr_eps_re', 'gr_eps_im', 'fallback' } );
s.path_type = double( s.path_type );
g9 = quadriga_lib.ssf_paths( s, single(-90) );
assertEqual( g9, pg );
for cls = { 'uint8', 'int8', 'uint16', 'int16' }                              % integer fields of any real class
    s.path_type = cast( rays.path_type, cls{1} );
    s.tap = cast( rays.tap, cls{1} );
    s.fallback = logical( rays.fallback );
    assertEqual( quadriga_lib.ssf_paths( s, -90 ), pg );
end

%% No path: empty outputs keep their shapes
s = rays;
for f = { 'delay', 'power', 'aod', 'eod', 'aoa', 'eoa', 'path_type', 'tap', 'fallback' }
    s.(f{1}) = s.(f{1})( [], : );
end
s.M = zeros( 8, 0 );
s.fbs_pos = zeros( 3, 0 );
s.lbs_pos = zeros( 3, 0 );
[ g0, l0, M0, a0, ~, ~, ~, f0, ~, p0, r0 ] = quadriga_lib.ssf_paths( s, -90, 'drift' );
assertEqual( { size(g0), size(l0), size(M0), size(a0), size(f0), size(p0), size(r0) }, ...
    { [0, 1], [0, 1], [8, 0], [0, 1], [3, 0], [0, 1], [0, 1] } );
assertTrue( isa(p0, 'uint64') && isa(r0, 'uint64') );

%% Errors
assert_throws( @() quadriga_lib.ssf_paths( rays ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.ssf_paths( rays, 0, 'auto', [], [], [], false, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out12( rays, 0 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.ssf_paths( [ rays, rays ], 0 ), 'Input ''rays'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.ssf_paths( rmfield( rays, 'tap' ), 0 ), 'Field ''tap'' is missing in ''rays''.' );
assert_throws( @() quadriga_lib.ssf_paths( rays, 0, 'foo' ), 'Input ''mobility'' must be ''phase'', ''drift'' or ''auto''.' );
assert_throws( @() quadriga_lib.ssf_paths( rays, [0, 0] ), 'Input ''link_gain_dB'' must have n_freq elements.' );
s = rays;
s.fbs_pos = [];
s.lbs_pos = [];
assert_throws( @() quadriga_lib.ssf_paths( s, 0, 'drift' ), 'Mobility ''drift'' needs ''rays.fbs_pos''' );

end

% Request 12 outputs so that the output-count guard fires
function out12(varargin)
[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.ssf_paths( varargin{:} ); %#ok<ASGLU>
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
