function test_ssf_rays
% MEX interface test for quadriga_lib.ssf_rays.
% Scope: argument and output handling, ray struct fields, index conversion, error paths.
% Physics validation lives in the Catch2 suite (test_ssf_rays.cpp, test_ssf_rays_quadriga.cpp).

%% Setup: UMi LOS with ground reflection, 3 links, two frequencies
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
sp.GR_enabled = 1;
tx_pos = [ 0, 0, 10 ];
rx_pos = [ 60, 10, 1.5; -30, 80, 1.5; 120, -40, 7.5 ];
fc = [ 6e9; 30e9 ];
[ lsp, mu, sig, gre, gim ] = lsps( sp, fc, tx_pos, rx_pos );
rnd = quadriga_lib.gbsm_draw( sp, 'cluster', quadriga_lib.gbsm_init_phases( sp, 'cluster', 7 ), tx_pos, rx_pos );
N = sp.NumClusters - 1;
M = sp.NumSubPaths;
n_ray = 2 + N * M;                                                            % LOS, GR, N clusters of M rays
fields = { 'method', 'tx_pos', 'rx_pos', 'center_frequency', 'delay', 'power', 'aod', 'eod', 'aoa', 'eoa', ...
    'M', 'path_type', 'tap', 'gr_eps_re', 'gr_eps_im', 'fbs_pos', 'lbs_pos', 'fallback' };

% 'quadriga': clusters of all frequencies, the shared "ray" set
[ d, p, aod, eod, aoa, eoa ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], ...
    [], gre, gim );
rq = quadriga_lib.gbsm_draw( sp, 'ray', quadriga_lib.gbsm_init_phases( sp, 'ray', 7 ), tx_pos, rx_pos, 0, ...
    [], [], fc );

% '38.901_MF': per-frequency cluster and "ray" sets, anchor LSPs at 2 GHz
rf = cat( 3, cset( sp, 'cluster', 1, tx_pos, rx_pos ), cset( sp, 'cluster', 2, tx_pos, rx_pos ) );
rr = cat( 3, cset( sp, 'ray', 1, tx_pos, rx_pos ), cset( sp, 'ray', 2, tx_pos, rx_pos ) );
la = lsps( sp, 2e9, tx_pos, rx_pos );
[ dm, pm, aodm, eodm, aoam, eoam ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, '38.901_MF', rf, la, ...
    [], [], gre, gim );
mf = @( varargin ) quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc, lsp, mu, sig, dm, pm, aodm, eodm, aoam, eoam, ...
    rr, '38.901_MF', varargin{:} );

%% Basic call and values: all fields, sizes and classes ('quadriga')
rays = quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc, lsp, mu, sig, d, p, aod, eod, aoa, eoa, rq, 'quadriga', ...
    [], [], [], gre, gim );
assertEqual( size(rays), [1, 3] );
assertEqual( fieldnames(rays)', fields );
for l = 1 : 3
    r = rays(l);
    assertEqual( r.method, 'quadriga' );
    assertEqual( r.tx_pos, tx_pos' );
    assertEqual( r.rx_pos, rx_pos(l, :)' );
    assertEqual( r.center_frequency, fc );
    assertEqual( size(r.delay), [n_ray, 1] );
    assertEqual( size(r.power), [n_ray, 2] );
    assertEqual( size(r.eoa), [n_ray, 1] );
    assertEqual( size(r.M), [8, n_ray, 2] );
    assertTrue( isa(r.path_type, 'uint32') && isa(r.tap, 'uint32') && isa(r.fallback, 'uint32') );
    assertEqual( r.path_type(1:3), uint32( [0; 1; 2] ) );                     % codes, not shifted
    assertEqual( r.tap(1:3), uint32( [0; 3; 6] ) );
    assertEqual( r.gr_eps_re, gre(l, :)' );
    assertEqual( r.gr_eps_im, gim(l, :)' );
    assertTrue( isempty(r.fbs_pos) && isempty(r.lbs_pos) && isempty(r.fallback) );
    assertElementsAlmostEqual( r.power(1, :), reshape( p(l, 1, :), 1, [] ), 'absolute', 1e-14 );
end

% Ground reflection disabled: no GR ray, empty permittivity fields
sn = sp;
sn.GR_enabled = 0;
[ d0, p0, a0, e0, b0, f0 ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp, rnd, 'quadriga' );
r0 = quadriga_lib.ssf_rays( sn, tx_pos, rx_pos, fc, lsp, mu, sig, d0, p0, a0, e0, b0, f0, rq, 'quadriga' );
assertEqual( size(r0(1).delay), [n_ray - 1, 1] );
assertEqual( fieldnames(r0)', fields );
assertTrue( isempty(r0(1).gr_eps_re) && isempty(r0(1).gr_eps_im) );
assertEqual( r0(1).tap(1:2), uint32( [0; 3] ) );

%% Optional inputs: method, i_freq (1-based), angle_model, pol_model, pol_variability, random_phase
r2 = mf( 1, [], [], gre, gim );
assertEqual( r2(1).center_frequency, fc(1) );
r3 = mf( 2, [], [], gre, gim );
assertEqual( r3(1).center_frequency, fc(2) );                                 % rays(1).center_frequency == fc(i_freq)
assertEqual( r3(2).gr_eps_re, gre(2, 2) );
assertEqual( size(r3(1).power), [n_ray, 1] );
assertTrue( max(abs(r3(1).delay - r2(1).delay)) > 0 );
r4 = mf( int32(2), [], [], gre, gim );                                        % integer class for the index
assertEqual( r4(1).delay, r3(1).delay );
assertEqual( r4(1).method, '38.901_MF' );
rt = mf( 1, 'additive', '38.901', gre, gim, false, true );
assertEqual( rt(3).M, r2(3).M );
assertEqual( rt(3).aod, r2(3).aod );
rt = mf( 1, 'rotation', [], gre, gim );
assertTrue( max(abs(rt(3).eod - r2(3).eod)) > 1e-6 );
rt = mf( 1, [], 'quadriga', gre, gim );
assertTrue( max(abs(rt(3).M(:) - r2(3).M(:))) > 1e-6 );
rt = mf( 1, [], [], gre, gim, true );
assertTrue( max(abs(rt(3).M(:) - r2(3).M(:))) > 1e-6 );
assertEqual( rt(3).delay, r2(3).delay );
rt = mf( 1, [], [], gre, gim, [], false );
assertEqual( rt(3).M(2 : 2 : 8, 3 : end), zeros(4, n_ray - 2) );              % initial phases 0: real matrices
r5 = mf( [], [], [], gre, gim, [], [] );
assertEqual( r5(2).M, r2(2).M );

% Default method '38.901' at one frequency; 'rnd_ray' as [n_link, n_gen]
[ d1, p1, a1, e1, b1, f1 ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp(:, 1, :), rnd, '38.901', ...
    rf(:, :, 1), [], [], [], gre(:, 1), gim(:, 1) );
rd = quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc(1), lsp(:, 1, :), mu(:, 1, :), sig(:, 1, :), d1, p1, a1, e1, ...
    b1, f1, rr(:, :, 1), [], [], [], [], gre(:, 1), gim(:, 1) );
assertEqual( rd(1).method, '38.901' );
assertEqual( rd(1).center_frequency, fc(1) );

%% Outputs and type handling
quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc, lsp, mu, sig, d, p, aod, eod, aoa, eoa, rq, 'quadriga', [], [], [], ...
    gre, gim );                                                               % zero outputs
rs = quadriga_lib.ssf_rays( sp, single(tx_pos), rx_pos, fc', lsp, mu, sig, d, p, aod, eod, aoa, eoa, rq, ...
    'quadriga', [], [], [], gre, gim );
assertEqual( rs(2).delay, rays(2).delay );                                    % single 'tx_pos', row 'center_freq'

%% Errors
args = { sp, tx_pos, rx_pos, fc, lsp, mu, sig, d, p, aod, eod, aoa, eoa, rq };
assert_throws( @() quadriga_lib.ssf_rays( args{1:13} ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.ssf_rays( args{:}, 'quadriga', 1, 'additive', '38.901', gre, gim, false, true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( args{:}, 'quadriga', 1, [], [], gre, gim ), 'Wrong number of output arguments.' );
assert_throws( @() mf( 0, [], [], gre, gim ), 'Input ''i_freq'' cannot be 0 (1-based index).' );
assert_throws( @() mf( 3, [], [], gre, gim ), 'Input ''i_freq'' is out of range for 2 frequencies.' );
assert_throws( @() quadriga_lib.ssf_rays( args{1:4}, lsp(:, :, 1:7), args{6:end}, 'quadriga', [], [], [], gre, gim ), ...
    'Input ''lsp'' must have size [n_link, n_freq, 8].' );
assert_throws( @() quadriga_lib.ssf_rays( args{:}, 'quadriga' ), 'are required when ''GR_enabled'' is set.' );
assert_throws( @() quadriga_lib.ssf_rays( args{:}, 'foo' ), 'Input ''method'' must be' );

end

% LSPs, means, std and ground permittivity at the links
function [ lsp, mu, sig, gre, gim ] = lsps( sp, fc, tx_pos, rx_pos )
phase = quadriga_lib.gbsm_init_phases( sp, 'lsf', 7 );
[ lsp, ~, gre, gim, ~, mu, sig ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, ...
    quadriga_lib.gbsm_draw( sp, 'lsf', phase, tx_pos, rx_pos ) );
end

% Per-frequency random set k (1-based set id)
function rnd = cset( sp, set_type, k, tx_pos, rx_pos )
rnd = quadriga_lib.gbsm_draw( sp, set_type, quadriga_lib.gbsm_init_phases( sp, set_type, 7, k ), tx_pos, rx_pos, k );
end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.ssf_rays( varargin{:} ); %#ok<ASGLU>
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
