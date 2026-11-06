function test_gbsm_single_link_channel
% MEX interface test for quadriga_lib.gbsm_single_link_channel.
% Scope: argument and output handling, optional inputs, index conversion, error paths.
% Physics validation lives in the Catch2 suite (test_gbsm_single_link_channel.cpp).

%% Setup: UMi LOS with ground reflection, 'quadriga' rays of one link at two frequencies, solved
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
sp.GR_enabled = 1;
tx_pos = [ 0, 0, 10 ];
rx_pos = [ 60, 10, 1.5 ];
fc = [ 3.5e9; 28e9 ];
phase = quadriga_lib.gbsm_init_phases( sp, 'lsf', 9 );
[ lsp, ~, gre, gim, ~, mu, sig ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, ...
    quadriga_lib.gbsm_draw( sp, 'lsf', phase, tx_pos, rx_pos ) );
rnd = quadriga_lib.gbsm_draw( sp, 'cluster', quadriga_lib.gbsm_init_phases( sp, 'cluster', 9 ), tx_pos, rx_pos );
[ d, p, aod, eod, aoa, eoa ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], [], ...
    [], gre, gim );
rr = quadriga_lib.gbsm_draw( sp, 'ray', quadriga_lib.gbsm_init_phases( sp, 'ray', 9 ), tx_pos, rx_pos, 0, ...
    [], [], fc );
rays = quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc, lsp, mu, sig, d, p, aod, eod, aoa, eoa, rr, 'quadriga', ...
    [], [], [], gre, gim );
rays = quadriga_lib.ssf_scatterers( rays );
tx = quadriga_lib.arrayant_generate( 'omni' );
rx = quadriga_lib.arrayant_generate( 'xpol' );
taps = unique( rays.tap );
n_tap = numel( taps );
n_path = sum( any( rays.power > 0, 2 ) | rays.path_type < 2 );

%% Basic call and values ('drift', the default of 'quadriga'; 'taps')
[ chan, par ] = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90 );
assertEqual( size(chan), [1, 1] );
assertEqual( size(chan.coeff_re), [2, 1, n_tap] );
assertEqual( size(chan.delay), [2, 1, n_tap] );
assertEqual( chan.center_frequency, fc(1) );
assertEqual( chan.tx_position, tx_pos' );
assertEqual( chan.rx_position, rx_pos' );
assertEqual( fieldnames(par)', { 'tap', 'n_ray_tap', 'fbs_pos', 'lbs_pos', 'gr_pos', 'link_gain_dB', 'b_med', ...
    'b_eff', 'limit', 'ratio', 'ratio_total' } );
assertTrue( isa(par.tap, 'uint64') && isa(par.n_ray_tap, 'uint64') );
assertEqual( par.tap, taps );                                                 % codes, not shifted
assertEqual( par.tap(1:2), uint64( [0; 3] ) );
assertEqual( par.link_gain_dB, -90 );
assertEqual( par.ratio, [0; 0] );
assertEqual( size(par.fbs_pos), [3, n_path] );

%% Optional inputs: mobility, output, i_freq (1-based), positions, orientations, kf_ratio, flags
[ c2, p2 ] = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], [], [], [], [], [], [], [], [] );
assertEqual( c2.coeff_re, chan.coeff_re );
assertEqual( p2, par );
[ c2, p2 ] = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, 'phase' );
assertTrue( ~isfield( p2, 'fbs_pos' ) );
assertEqual( size(c2.coeff_re), size(chan.coeff_re) );
[ c3, p3 ] = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, 'drift', 'rays' );
assertEqual( size(c3.coeff_re), [2, 1, n_path] );
assertEqual( size(c3.path_gain), [n_path, 1] );
assertEqual( size(c3.fbs_pos), [3, n_path] );
assertEqual( numel(p3.tap), n_path );
assertTrue( ~isfield( p3, 'n_ray_tap' ) );
c4 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], 2 );
assertEqual( c4.center_frequency, fc(2) );
assertTrue( max( abs( c4.coeff_re(:) - chan.coeff_re(:) ) ) > 1e-9 );
c5 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], int32(2) );
assertEqual( c5.coeff_re, c4.coeff_re );
c6 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], 1, tx_pos, [ rx_pos; rx_pos + [ 1, 0, 0 ] ] );
assertEqual( size(c6.coeff_re), [2, 1, n_tap, 2] );
assertEqual( c6.rx_position, [ rx_pos', rx_pos' + [ 1; 0; 0 ] ] );
assertElementsAlmostEqual( c6.coeff_re(:, :, :, 1), chan.coeff_re, 'absolute', 1e-12 );
[ c7, p7 ] = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, [ -90; -80 ], [], [], 1, [], ...
    [ rx_pos; rx_pos + [ 1, 0, 0 ] ] );
assertEqual( p7.link_gain_dB, [ -90; -80 ] );
assertTrue( p7.ratio(2) > 0 && p7.ratio(1) == 0 );
c8 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], [], [], [], [ 0, 0, 1 ], [ 0, 0.5, 0 ] );
assertEqual( c8.tx_orientation, [ 0; 0; 1 ] );
assertEqual( c8.rx_orientation, [ 0; 0.5; 0 ] );
assertTrue( max( abs( c8.coeff_re(:) - chan.coeff_re(:) ) ) > 1e-9 );
c9 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], [], [], [], [], [], 0 );
assertTrue( max( abs( c9.coeff_re(:) - chan.coeff_re(:) ) ) > 1e-9 );        % kf_ratio 0
s = rays;
s.center_frequency(2) = 60e9;                                                 % oxygen absorption needs > 52 GHz
c9 = quadriga_lib.gbsm_single_link_channel( tx, rx, s, -90, [], 'rays', 2 );
c0 = quadriga_lib.gbsm_single_link_channel( tx, rx, s, -90, [], 'rays', 2, [], [], [], [], [], true );
assertTrue( all( c0.path_gain < c9.path_gain ) );
c9 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90, [], [], [], [], [], [], [], [], [], 1 );
assertElementsAlmostEqual( c9.delay - chan.delay, norm( rx_pos - tx_pos ) / 299792458 * ones( 2, 1, n_tap ), ...
    'absolute', 1e-15 );                                                      % use_absolute_delays

%% Outputs: 0 to 2
quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90 );
c1 = quadriga_lib.gbsm_single_link_channel( tx, rx, rays, -90 );
assertEqual( c1.coeff_im, chan.coeff_im );

%% Errors
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, rays ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, rays, 0, [], [], [], [], [], [], [], [], [], [], 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out3( tx, rx, rays, 0 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, [ rays, rays ], 0 ), ...
    'Input ''rays'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, rays, 0, [], [], 0 ), ...
    'Input ''i_freq'' cannot be 0 (1-based index).' );
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, rays, 0, [], [], 3 ), ...
    'Input ''i_freq'' is out of range for 2 frequencies.' );
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, rays, 0, [], [], [], [], rx_pos' ), ...
    'Input ''rx_pos'' must have 3 columns.' );
assert_throws( @() quadriga_lib.gbsm_single_link_channel( tx, rx, rays, 0, [], 'foo' ), ...
    'Input ''output'' must be ''taps'' or ''rays''.' );
bad = tx;
bad.e_theta_re = bad.e_theta_re(:, 1:2);
assert_throws( @() quadriga_lib.gbsm_single_link_channel( bad, rx, rays, 0 ), 'Transmit antenna: ' );

end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.gbsm_single_link_channel( varargin{:} ); %#ok<ASGLU>
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
