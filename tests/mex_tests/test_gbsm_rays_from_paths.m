function test_gbsm_rays_from_paths
% MEX interface test for quadriga_lib.gbsm_rays_from_paths.
% Scope: argument and output handling, ray struct fields, codes, error paths.
% Physics validation lives in the Catch2 suite (test_gbsm_rays_from_paths.cpp).

%% Setup: LOS, GR and NLOS path of one link at two frequencies
c0 = 299792458;
fc = [ 3.5e9; 28e9 ];
tx = [ 0; 0; 10 ];
rx = [ 40; 30; 1.5 ];
gr = [ tx(1:2) + ( rx(1:2) - tx(1:2) ) * tx(3) / ( tx(3) + rx(3) ); 0 ];      % specular point
fb = [ tx / 2 + rx / 2, gr, [ 10; 20; 5 ] ];
lb = [ tx / 2 + rx / 2, gr, [ 30; 35; 3 ] ];
d = norm( rx - tx );
pl = [ d; norm( gr - tx ) + norm( rx - gr ); norm( fb(:, 3) - tx ) + norm( lb(:, 3) - fb(:, 3) ) + norm( rx - lb(:, 3) ) ];
pg = [ 1e-6, 2e-6; 3e-7, 4e-7; 5e-8, 6e-8 ];
M = reshape( 1 : 48, 8, 3, 2 ) / 48;
pt = [ 0; 1; 2 ];
fields = { 'method', 'tx_pos', 'rx_pos', 'center_frequency', 'delay', 'power', 'aod', 'eod', 'aoa', 'eoa', ...
    'M', 'path_type', 'tap', 'gr_eps_re', 'gr_eps_im', 'fbs_pos', 'lbs_pos', 'fallback' };

%% Basic call and values
rays = quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb, lb, pg, pl, M, 'quadriga', pt );
assertEqual( fieldnames(rays)', fields );
assertEqual( size(rays), [1, 1] );
assertEqual( rays.method, 'quadriga' );
assertEqual( rays.tx_pos, tx );
assertEqual( rays.rx_pos, rx );
assertEqual( rays.center_frequency, fc );
assertElementsAlmostEqual( rays.delay, ( pl - d ) / c0, 'absolute', 1e-18 );
assertEqual( rays.power, pg );
assertEqual( rays.M, M );
assertEqual( rays.fbs_pos, fb );
assertEqual( rays.lbs_pos, lb );
assertTrue( isa(rays.path_type, 'uint64') && isa(rays.tap, 'uint64') && isa(rays.fallback, 'uint64') );
assertEqual( rays.path_type, uint64( pt ) );                                  % codes, not shifted
assertEqual( rays.tap, uint64( [0; 3; 6] ) );
assertTrue( isempty(rays.fallback) && isempty(rays.gr_eps_re) && isempty(rays.gr_eps_im) );
v = fb(:, 3) - tx;
assertElementsAlmostEqual( [ rays.aod(3), rays.eod(3) ], [ atan2( v(2), v(1) ), atan2( v(3), norm( v(1:2) ) ) ], ...
    'absolute', 1e-14 );
v = lb(:, 3) - rx;
assertElementsAlmostEqual( [ rays.aoa(3), rays.eoa(3) ], [ atan2( v(2), v(1) ), atan2( v(3), norm( v(1:2) ) ) ], ...
    'absolute', 1e-14 );

% Round trip: the paths of 'drift' at the same positions are the input paths
[ g2, l2, M2, ~, ~, ~, ~, f2, b2, p2 ] = quadriga_lib.ssf_paths( rays, [0; 0], 'drift' );
assertElementsAlmostEqual( g2, pg, 'absolute', 1e-20 );
assertElementsAlmostEqual( l2, pl, 'absolute', 1e-9 );
assertElementsAlmostEqual( M2, M, 'absolute', 1e-14 );
assertElementsAlmostEqual( f2, fb, 'absolute', 1e-12 );
assertEqual( p2, uint64( pt ) );

%% Optional inputs: method, path_type, tap, gr_eps_re, gr_eps_im
r2 = quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb(:, [1 3]), lb(:, [1 3]), pg([1 3], :), pl([1 3]), ...
    M(:, [1 3], :) );
assertEqual( r2.method, 'quadriga' );
assertEqual( r2.path_type, uint64( [0; 2] ) );                                % LOS by geometry
assertEqual( r2.tap, uint64( [0; 6] ) );
r3 = quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb(:, [1 3]), lb(:, [1 3]), pg([1 3], :), pl([1 3]), ...
    M(:, [1 3], :), [], [], [], [], [] );
assertEqual( r3, r2 );
r4 = quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb, lb, pg, pl, M, '38.901', int32( pt ), [ 5; 3; 0 ], ...
    [ 5; 7 ], [ 0.1, 0.2 ] );
assertEqual( r4.method, '38.901' );
assertEqual( r4.tap, uint64( [5; 3; 0] ) );                                   % codes, not shifted
assertEqual( r4.gr_eps_re, [ 5; 7 ] );
assertEqual( r4.gr_eps_im, [ 0.1; 0.2 ] );

%% Outputs and type handling
quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb, lb, pg, pl, M, 'quadriga', pt );
r5 = quadriga_lib.gbsm_rays_from_paths( fc', tx', single( rx ), fb, lb, pg, pl', M, 'quadriga', pt' );
assertEqual( r5.delay, rays.delay );
assertEqual( r5.rx_pos, rx );

%% Errors
assert_throws( @() quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb, lb, pg, pl ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb, lb, pg, pl, M, [], [], [], [], [], 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( fc, tx, rx, fb, lb, pg, pl, M ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb(:, [1 1]), lb(:, [1 1]), pg(1:2, :), [d; d], ...
    M(:, 1:2, :) ), 'More than one path matches the LOS geometry; pass ''path_type''.' );
assert_throws( @() quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, fb, lb, pg, pl, M, [], pt, [], [ 5; 7 ] ), ...
    'Inputs ''gr_eps_re'' and ''gr_eps_im'' must both be given or both be omitted.' );
lz = lb;
lz(3, 2) = 0.01;
assert_throws( @() quadriga_lib.gbsm_rays_from_paths( fc, tx, rx, lz, lz, pg, pl, M, [], pt ), ...
    'The interaction point of the GR path must be on the ground' );
assert_throws( @() quadriga_lib.gbsm_rays_from_paths( 0 * fc, tx, rx, fb, lb, pg, pl, M ), ...
    'Input ''center_frequency'' must be finite and > 0.' );

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.gbsm_rays_from_paths( varargin{:} ); %#ok<ASGLU>
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
