function test_ssf_scatterers
% MEX interface test for quadriga_lib.ssf_scatterers.
% Scope: argument and output handling, struct copy rules, validity scales, error paths.
% Physics validation lives in the Catch2 suite (test_gbsm_rays_solve_scatterers.cpp, test_gbsm_rays.cpp).

%% Setup: UMi LOS, 'quadriga' rays of 2 links at two frequencies
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
tx_pos = [ 0, 0, 10 ];
rx_pos = [ 60, 10, 1.5; -30, 80, 1.5 ];
fc = [ 3.5e9; 28e9 ];
rays = make_rays( sp, tx_pos, rx_pos, fc );
n_ray = numel( rays(1).delay );
fields = fieldnames( rays )';

%% Basic call and values: all fields, solver fields new, the other fields copied
[ r, b_med, b_eff ] = quadriga_lib.ssf_scatterers( rays(1) );
assertEqual( fieldnames(r)', fields );
assertEqual( size(r), [1, 1] );
assertEqual( size(r.fbs_pos), [3, n_ray] );
assertEqual( size(r.lbs_pos), [3, n_ray] );
assertEqual( size(r.fallback), [n_ray, 1] );
assertTrue( isa(r.fallback, 'uint32') && all( r.fallback <= 1 ) && any( r.fallback == 0 ) );
for f = fields( 1 : 15 )
    assertEqual( r.(f{1}), rays(1).(f{1}) );
end
assertElementsAlmostEqual( r.fbs_pos(:, 1), ( rays(1).tx_pos + rays(1).rx_pos ) / 2, 'absolute', 1e-12 ); % LOS
assertTrue( isempty(rays(1).fbs_pos) );                                       % input unchanged

% Validity scales [2, n_freq], equal per column to the report of gbsm_single_link_channel
assertEqual( size(b_med), [2, 2] );
assertEqual( size(b_eff), [2, 2] );
assertTrue( all( b_med(:) > 0 & b_eff(:) > 0 ) );
ant = quadriga_lib.arrayant_generate( 'omni' );
for k = 1 : 2
    [ ~, par ] = quadriga_lib.gbsm_single_link_channel( ant, ant, r, 0, 'drift', 'taps', k );
    assertEqual( b_med(:, k), par.b_med );
    assertEqual( b_eff(:, k), par.b_eff );
end

% Round trip in a struct array (identical fields)
rays(2) = quadriga_lib.ssf_scatterers( rays(2) );
assertEqual( size(rays(2).fbs_pos), [3, n_ray] );
assertTrue( isempty(rays(1).fbs_pos) );

%% Optional input: d_min; an already solved input stays unchanged
[ r2, m2, e2 ] = quadriga_lib.ssf_scatterers( rays(1), [] );
assertEqual( r2, r );
assertEqual( m2, b_med );
r2 = quadriga_lib.ssf_scatterers( rays(1), 2 );
assertEqual( r2.fbs_pos, r.fbs_pos );
fbs0 = r.fbs_pos + 0;                                                         % copies, a plain copy shares the data
lbs0 = r.lbs_pos + 0;
fb0 = r.fallback + 0;
r5 = quadriga_lib.ssf_scatterers( r, 5 );                                     % 2 rays fall back at 5 m, none at 2 m
assertEqual( r.fbs_pos, fbs0 );
assertEqual( r.lbs_pos, lbs0 );
assertEqual( r.fallback, fb0 );
assertTrue( max(abs(r5.fbs_pos(:) - fbs0(:))) > 1e-3 );
assertTrue( sum(r5.fallback) > sum(fb0) );
assertEqual( r5.delay, r.delay );

%% Outputs: 0 to 3
quadriga_lib.ssf_scatterers( rays(1) );
r1 = quadriga_lib.ssf_scatterers( rays(1) );
assertEqual( r1, r );
[ ~, m1 ] = quadriga_lib.ssf_scatterers( rays(1) );
assertEqual( m1, b_med );
assertEqual( e2, b_eff );

%% Type handling: optional fields missing, integer fields of another class
s = rmfield( rays(1), { 'gr_eps_re', 'gr_eps_im', 'fbs_pos', 'lbs_pos', 'fallback' } );
s.path_type = double( s.path_type );
s.tap = int32( s.tap );
rs = quadriga_lib.ssf_scatterers( s, single(2) );
assertEqual( fieldnames(rs)', fields );                                       % all fields on output
assertTrue( isempty(rs.gr_eps_re) && isempty(rs.gr_eps_im) );
assertEqual( rs.fbs_pos, r.fbs_pos );
assertEqual( rs.path_type, double( r.path_type ) );                           % copied as given
s = r;
s.path_type = int8( s.path_type );
s.tap = uint16( s.tap );
s.fallback = logical( s.fallback );
rs = quadriga_lib.ssf_scatterers( s );
assertEqual( rs.fbs_pos, r.fbs_pos );
assertTrue( isa(rs.path_type, 'int8') && isa(rs.tap, 'uint16') && isa(rs.fallback, 'uint32') );
assertEqual( rs.tap, s.tap );
s.tap = mod( (1 : numel(s.tap))', 2 ) == 1;                                   % logical labels
rs = quadriga_lib.ssf_scatterers( s );
assertTrue( islogical(rs.tap) );
assertEqual( rs.tap, s.tap );
s = rays(1);                                                                  % empty cell or struct: empty
s.gr_eps_re = cell( 0, 1 );
s.gr_eps_im = struct( [] );
rs = quadriga_lib.ssf_scatterers( s );
assertEqual( rs.fbs_pos, r.fbs_pos );
assertEqual( { class(rs.gr_eps_re), size(rs.gr_eps_re), class(rs.gr_eps_im), size(rs.gr_eps_im) }, ...
    { 'double', [0, 1], 'double', [0, 0] } );

%% Errors
assert_throws( @() quadriga_lib.ssf_scatterers(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.ssf_scatterers( rays(1), 2, 3 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( rays(1) ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.ssf_scatterers( rays ), 'Input ''rays_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.ssf_scatterers( 1 ), 'Input ''rays_in'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.ssf_scatterers( rmfield( rays(1), 'delay' ) ), ...
    'Field ''delay'' is missing in ''rays_in''.' );
s = rays(1);
s.foo = 1;
assert_throws( @() quadriga_lib.ssf_scatterers( s ), 'Unknown field ''foo'' in ''rays_in''.' );
s = rays(1);
s.method = 1;
assert_throws( @() quadriga_lib.ssf_scatterers( s ), 'Field ''method'' of ''rays_in'' must be a string.' );
s = rays(1);
s.tap = double( s.tap );
s.tap(1) = -1;
assert_throws( @() quadriga_lib.ssf_scatterers( s ), 'Entries in ''rays_in.tap'' must be integers' );
s = rays(1);
s.delay(2) = NaN;
assert_throws( @() quadriga_lib.ssf_scatterers( s ), 'Input ''rays.delay'' must be finite.' );
assert_throws( @() quadriga_lib.ssf_scatterers( rays(1), 0 ), 'Input ''d_min'' must be finite and > 0.' );

end

% 'quadriga' rays of a set of links
function rays = make_rays( sp, tx_pos, rx_pos, fc )
phase = quadriga_lib.gbsm_init_phases( sp, 'lsf', 3 );
[ lsp, abs_toa, ~, ~, eod_off, mu, sig ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, ...
    quadriga_lib.gbsm_draw( sp, 'lsf', phase, tx_pos, rx_pos ) );
rnd = quadriga_lib.gbsm_draw( sp, 'cluster', quadriga_lib.gbsm_init_phases( sp, 'cluster', 3 ), tx_pos, rx_pos );
[ d, p, aod, eod, aoa, eoa ] = quadriga_lib.ssf_clusters( sp, tx_pos, rx_pos, lsp, rnd, 'quadriga', [], [], ...
    eod_off, abs_toa );
rr = quadriga_lib.gbsm_draw( sp, 'ray', quadriga_lib.gbsm_init_phases( sp, 'ray', 3 ), tx_pos, rx_pos, 0, ...
    [], [], fc );
rays = quadriga_lib.ssf_rays( sp, tx_pos, rx_pos, fc, lsp, mu, sig, d, p, aod, eod, aoa, eoa, rr, 'quadriga' );
end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.ssf_scatterers( varargin{:} ); %#ok<ASGLU>
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
