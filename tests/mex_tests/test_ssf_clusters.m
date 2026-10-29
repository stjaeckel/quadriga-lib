function test_ssf_clusters
% MEX interface test for quadriga_lib.ssf_clusters.
% Scope: argument and output handling, optional inputs, slot layout, error paths.
% Physics validation lives in the Catch2 suite (test_ssf_clusters.cpp, test_ssf_clusters_mf.cpp,
% test_ssf_clusters_quadriga.cpp).

%% Setup: UMi LOS and NLOS files, 4 links, LSPs and random values of the library
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
sn = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_NLOS.conf' ) );
tx_pos = [ 0, 0, 10 ];
rx_pos = [ 60, 10, 1.5; -30, 80, 1.5; 120, -40, 7.5; 15, 25, 1.5 ];
[ lsp, rnd, rnd_f ] = draw( sp, 6e9, tx_pos, rx_pos, 11 );
[ lsp_n, rnd_n, rnd_fn, abs_toa_n, eod_n ] = draw( sn, 6e9, tx_pos, rx_pos, 12 );
N = sp.NumClusters - 1;                                                       % 12 LOS, 19 NLOS

% LOS geometry of slot 1
v = rx_pos - tx_pos;
aod_l = atan2( v(:, 2), v(:, 1) );
eod_l = atan2( v(:, 3), hypot( v(:, 1), v(:, 2) ) );
aoa_l = mod( aod_l + 2 * pi, 2 * pi ) - pi;                                   % aod + pi, wrapped

%% Basic call, all outputs
[ delay, power, aod, eod, aoa, eoa, n_active ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, ...
    '38.901', rnd_f );
assertEqual( size(delay), [4, N + 1] );
assertEqual( size(power), [4, N + 1] );
assertEqual( size(aod), [4, N + 1] );
assertEqual( size(eoa), [4, N + 1] );
assertEqual( size(n_active), [4, 1] );
assertTrue( isa(delay, 'double') && isa(power, 'double') && isa(eod, 'double') && isa(aoa, 'double') );
assertTrue( isa(n_active, 'uint64') );
assertEqual( delay(:, 1), zeros(4, 1) );
assertElementsAlmostEqual( aod(:, 1), aod_l, 'absolute', 1e-14 );
assertElementsAlmostEqual( eod(:, 1), eod_l, 'absolute', 1e-14 );
assertElementsAlmostEqual( aoa(:, 1), aoa_l, 'absolute', 1e-14 );
assertElementsAlmostEqual( eoa(:, 1), -eod_l, 'absolute', 1e-14 );
K = lsp(:, 1, 2);
assertElementsAlmostEqual( power(:, 1), K ./ (K + 1), 'absolute', 1e-14 );     % LOS slot
assertTrue( all( sum(power, 2) <= 1 + 1e-12 ) );                              % -25 dB removal, no renormalisation
assertEqual( n_active, uint64( sum( power(:, 2:end) > 0, 2 ) ) );
assertTrue( all( abs( aod(:) ) <= pi & abs( eod(:) ) <= pi / 2 ) );

% NLOS scenario: zero LOS power, the NLOS powers sum to 1
[ ~, pn ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn );
assertEqual( size(pn), [4, 20] );
assertEqual( pn(:, 1), zeros(4, 1) );
assertTrue( all( sum(pn, 2) <= 1 + 1e-12 & sum(pn, 2) > 0.9 ) );

%% Optional inputs: method
[ d2, p2 ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, [], rnd_f );
assertEqual( d2, delay );
assertEqual( p2, power );
[ dq, pq, ~, ~, ~, ~, nq ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga' );
assertEqual( size(dq), [4, N + 1] );
assertElementsAlmostEqual( sum(pq, 2), ones(4, 1), 'absolute', 1e-12 );      % no removal in 'quadriga'
assertEqual( nq, uint64( N * ones(4, 1) ) );
assertTrue( max(abs(dq(:) - delay(:))) > 1e-9 );

% '38.901_MF' with two frequencies and anchor LSPs: delays and angles per frequency
[ lsp2, ~, rnd_f1 ] = draw( sp, [ 6e9, 30e9 ], tx_pos, rx_pos, 11 );
rnd_f2 = quadriga_lib.ssf_draw( sp, quadriga_lib.ssf_init_phases( sp, 11, 2 ), tx_pos, rx_pos, 2 );
lsp_a = draw( sp, 2e9, tx_pos, rx_pos, 11 );
[ dm, pm, am ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp2, rnd, '38.901_MF', cat( 3, rnd_f1, rnd_f2 ), lsp_a );
assertEqual( size(dm), [4, N + 1, 2] );
assertEqual( size(pm), [4, N + 1, 2] );
assertEqual( size(am), [4, N + 1, 2] );
assertEqual( rnd_f1, rnd_f );

%% Optional inputs: eod_offset_deg (NLOS, 38.901) and abs_toa (NLOS)
[ d0, ~, ~, e0 ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn );
[ ~, ~, ~, e1 ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn, [], eod_n );
[ ~, ~, ~, e2 ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn, [], [] );
assertEqual( e2, e0 );
assertTrue( max(max(abs(e1(:, 2:end) - e0(:, 2:end)))) > 1e-3 );
assertEqual( e1(:, 1), e0(:, 1) );                                            % LOS slot unchanged
[ dt, pt ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn, [], [], abs_toa_n );
assertTrue( all( abs_toa_n > 0 ) );
assertElementsAlmostEqual( dt(:, 2:end) - d0(:, 2:end), repmat( abs_toa_n, 1, 19 ), 'absolute', 1e-15 );
assertEqual( pt, pn );
assertEqual( quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn, [], [], abs_toa_n' ), dt );

%% Optional inputs: ground reflection (slot 2)
sg = sp;
sg.GR_enabled = 1;
eps_re = [ 5; 10; 15; 20 ];
eps_im = [ 0.5; 1; 0; 2 ];
[ dg, pg, ag, eg, ~, eag ] = quadriga_lib.ssf_clusters( sg, tx_pos, rx_pos, lsp, rnd, '38.901', rnd_f, [], [], ...
    [], eps_re, eps_im );
assertEqual( size(dg), [4, N + 2] );
assertEqual( pg(:, [1, 3:end]), power );                                      % other slots unchanged
assertTrue( all( pg(:, 2) > 0 ) );
assertElementsAlmostEqual( ag(:, 2), aod_l, 'absolute', 1e-14 );
e_gr = -atan2( tx_pos(3) + rx_pos(:, 3), hypot( v(:, 1), v(:, 2) ) );
assertElementsAlmostEqual( [ eg(:, 2), eag(:, 2) ], [ e_gr, e_gr ], 'absolute', 1e-14 );
d_gr = sqrt( sum( v(:, 1:2).^2, 2 ) + ( tx_pos(3) + rx_pos(:, 3) ).^2 );
assertElementsAlmostEqual( dg(:, 2), ( d_gr - sqrt( sum( v.^2, 2 ) ) ) / 299792458, 'absolute', 1e-15 );
[ ~, pg2 ] = quadriga_lib.ssf_clusters( sg, tx_pos, rx_pos, lsp, rnd, '38.901', rnd_f, [], [], [], eps_re, ...
    eps_im + 1 );
assertTrue( max(abs(pg2(:, 2) - pg(:, 2))) > 1e-6 );                          % gr_eps_im reaches C++

%% Optional inputs: o2i (38.901, NLOS)
[ ~, ~, ~, ~, ~, eo ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn, [], [], ...
    [], [], [], [ false; true; false; false ] );
[ ~, ~, ~, ~, ~, ea ] = quadriga_lib.ssf_clusters( sn, tx_pos, rx_pos, lsp_n, rnd_n, '38.901', rnd_fn );
assertEqual( eo([1, 3, 4], :), ea([1, 3, 4], :) );
assertTrue( max(abs(eo(2, 2:end) - ea(2, 2:end))) > 1e-3 );
eo2 = ssf_eoa( sn, tx_pos, rx_pos, lsp_n, rnd_n, rnd_fn, int32([ 0, 2, 0, 0 ]) );                % numeric, row
assertEqual( eo2, eo );
assertEqual( ssf_eoa( sn, tx_pos, rx_pos, lsp_n, rnd_n, rnd_fn, [] ), ea );

%% Optional inputs: n_cluster (counts, no index shift)
nc = [ 12; 6; 9; 12 ];
[ dc, pc, ac, ~, ~, ~, nac ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], ...
    [], [], [], [], nc );
assertEqual( pc(2, 8:end), zeros(1, 6) );                                     % slots 8 ... 13 inactive
assertEqual( dc(2, 8:end), zeros(1, 6) );
assertElementsAlmostEqual( ac(2, 8:end), repmat( aod_l(2), 1, 6 ), 'absolute', 1e-14 );
assertEqual( pc(3, 11:end), zeros(1, 3) );
assertEqual( nac, uint64(nc) );
assertEqual( quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], [], [], [], [], ...
    uint32(nc') ), dc );
assertEqual( quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], [], [], [], [], ...
    [] ), dq );

%% Outputs: none, and fewer than the maximum
quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, '38.901', rnd_f );
assertEqual( quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, '38.901', rnd_f ), delay );
[ ~, ~, ~, ~, aoa3 ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, '38.901', rnd_f );
assertEqual( aoa3, aoa );

%% NumClusters = 1: LOS path only
s1 = sp;
s1.NumClusters = 1;
s1.NumClusters_min = 1;
[ d1, p1, ~, ~, ~, ~, n1 ] = quadriga_lib.ssf_clusters( s1, tx_pos, rx_pos, lsp, zeros(4, 0) );
assertEqual( size(d1), [4, 1] );
assertEqual( p1, ones(4, 1) );
assertEqual( n1, uint64( zeros(4, 1) ) );

%% Type handling
[ ds, ps ] = quadriga_lib.ssf_clusters( sp, single(tx_pos), single(rx_pos), lsp, rnd, '38.901', rnd_f );
assertEqual( ds, delay );                                                      % positions exact in single
assertEqual( ps, power );

%% Errors
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], [], [], [], ...
    [], [], 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out8( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga' ), 'Wrong number of output arguments.' );
msg_nc = 'Entries in ''n_cluster'' must be non-negative integers.';
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], [], [], [], ...
    [], [ 12; 6.5; 9; 12 ] ), msg_nc );
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], [], [], [], ...
    [], [ 12; -6; 9; 12 ] ), msg_nc );
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], [], [], [], ...
    [], [ 12; 3; 9; 12 ] ), 'Entries of ''n_cluster'' must be in [6, 12].' );                     % C++
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd ), ...
    'Input ''rnd_freq'' is required for method ''38.901''.' );                                       % C++
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'other' ), ...
    'Input ''method'' must be ''38.901'', ''38.901_MF'' or ''quadriga''.' );                         % C++
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp2, rnd, '38.901', rnd_f ), ...
    'Method ''38.901'' supports one frequency only' );                                              % C++
assert_throws( @() quadriga_lib.ssf_clusters( sg, tx_pos, rx_pos, lsp, rnd, 'quadriga' ), ...
    'Inputs ''gr_eps_re'' and ''gr_eps_im'' are required when ''GR_enabled'' is set.' );              % C++
assert_throws( @() quadriga_lib.ssf_clusters( sp, tx_pos, zeros(0, 3), lsp(1:0, :, :), rnd(1:0, :), ...
    'quadriga' ), 'rx_pos' );                                                                       % zero links

end

% LSPs and the random values of one scenario at the links
function [ lsp, rnd, rnd_f, abs_toa, eod_offset_deg ] = draw( sp, fc, tx_pos, rx_pos, seed )
phase = quadriga_lib.lsf_init_phases( sp, seed, [], [], [], false );
[ lsp, abs_toa, ~, ~, eod_offset_deg ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, ...
    quadriga_lib.lsf_draw( sp, phase, tx_pos, rx_pos ) );
rnd = quadriga_lib.ssf_draw( sp, quadriga_lib.ssf_init_phases( sp, seed ), tx_pos, rx_pos );
rnd_f = quadriga_lib.ssf_draw( sp, quadriga_lib.ssf_init_phases( sp, seed, 1 ), tx_pos, rx_pos, 1 );
end

% EoA of the '38.901' method with an O2I flag
function eoa = ssf_eoa( sp, tx_pos, rx_pos, lsp, rnd, rnd_f, o2i )
[ ~, ~, ~, ~, ~, eoa ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, '38.901', rnd_f, [], [], [], ...
    [], [], o2i );
end

% Request 8 outputs so that the output-count guard fires
function out8(varargin)
[~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.ssf_clusters( varargin{:} ); %#ok<ASGLU>
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
