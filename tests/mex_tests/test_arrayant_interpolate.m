function test_arrayant_interpolate
% MEX interface test for quadriga_lib.arrayant_interpolate.
% Scope: argument and output handling, index conversion, single- and multi-frequency mode, kernels, error paths.
% Physics validation lives in the Catch2 suites (test_arrayant_interpolate.cpp, test_arrayant_interpolate_multi.cpp).

%% Setup: two elements with two azimuth samples (0 and pi) each
% Element 1: e-theta in opposite phase (linear), e-phi in phase; element 2: e-theta with a 90 degree phase step
ant = struct( 'e_theta_re', cat(3, [-2, 2], [1, 0]), 'e_theta_im', cat(3, [-1, 1], [0, 1]), ...
    'e_phi_re', cat(3, [3, 1], [0, 0]), 'e_phi_im', cat(3, [6, 2], [0, 0]), ...
    'azimuth_grid', [0, pi], 'elevation_grid', 0 );
az = [0, 1, 2] * pi/4;
el = [-0.5, 0, 0.5];
c = cos([0, 1, 2] * pi/8);
s = sin([0, 1, 2] * pi/8);

%% Basic call and values
[ V_re, V_im, H_re, H_im, dist, az_loc, el_loc, gamma ] = quadriga_lib.arrayant_interpolate( ant, az, el );
assertEqual( size(V_re), [2, 3] );
assertTrue( isa(V_re, 'double') && isa(gamma, 'double') );
assertElementsAlmostEqual( V_re, [ -2, -1, 0; c ], 'absolute', 1e-14 );
assertElementsAlmostEqual( V_im, [ -1, -0.5, 0; s ], 'absolute', 1e-14 );
assertElementsAlmostEqual( H_re, [ 3, 2.5, 2; 0, 0, 0 ], 'absolute', 1e-14 );
assertElementsAlmostEqual( H_im, [ 6, 5, 4; 0, 0, 0 ], 'absolute', 1e-14 );
assertElementsAlmostEqual( dist, zeros(2, 3), 'absolute', 1e-14 );
assertElementsAlmostEqual( az_loc, [ az; az ], 'absolute', 1e-14 );
assertElementsAlmostEqual( el_loc, [ el; el ], 'absolute', 1e-14 );
assertElementsAlmostEqual( gamma, zeros(2, 3), 'absolute', 1e-14 );

%% Optional inputs: element (1-based, permuted, repeated), [] and omitted give all elements
[ V_e, ~, ~, H_e ] = quadriga_lib.arrayant_interpolate( ant, az, el, [2, 1, 2] );
assertEqual( V_e, V_re([2, 1, 2], :) );
assertEqual( H_e, H_im([2, 1, 2], :) );
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, [] ), V_re );
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, [2; 1; 2] ), V_e );        % Column vector
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, uint32([2, 1, 2]) ), V_e ); % Integer class

%% Optional inputs: orientation (heading -pi/8 adds pi/8 to the local azimuth)
[ V_o, ~, ~, ~, ~, az_o ] = quadriga_lib.arrayant_interpolate( ant, az, el, [], [0; 0; -pi/8] );
assertElementsAlmostEqual( az_o, [ az; az ] + pi/8, 'absolute', 1e-14 );
assertElementsAlmostEqual( V_o(2, :), cos([0, 1, 2] * pi/8 + pi/16), 'absolute', 1e-14 );
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, [], [] ), V_re );
ori = zeros(3, 2, 3);
ori(3, 2, :) = -pi/8;                                                                    % Per output and angle
[ ~, ~, ~, ~, ~, az_o2 ] = quadriga_lib.arrayant_interpolate( ant, az, el, [], ori );
assertElementsAlmostEqual( az_o2, [ az; az + pi/8 ], 'absolute', 1e-14 );

%% Optional inputs: element_pos (distance from the wavefront plane, minus the projection onto the direction)
pos = [ 1, 0; 0, 2; 0, 0 ];
[ ~, ~, ~, ~, dist_p ] = quadriga_lib.arrayant_interpolate( ant, [0, pi/2], [0, 0], [], [], pos );
assertElementsAlmostEqual( dist_p, [ -1, 0; 0, -2 ], 'absolute', 1e-14 );
ant_p = ant;
ant_p.element_pos = pos;
[ ~, ~, ~, ~, dist_a ] = quadriga_lib.arrayant_interpolate( ant_p, [0, pi/2], [0, 0] );  % Positions of the arrayant
assertEqual( dist_a, dist_p );
[ ~, ~, ~, ~, dist_e ] = quadriga_lib.arrayant_interpolate( ant_p, [0, pi/2], [0, 0], [], [], [] );
assertEqual( dist_e, dist_p );

%% Optional inputs: use_kernel
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [], [] ), V_re );
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [], 1 ), V_re );
assertEqual( quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [], true ), V_re ); % Logical 1 = GENERIC
has_avx2 = true;
try
    V_k = quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [], 2 );
catch ME
    has_avx2 = false;
    assertTrue( ~isempty(strfind(ME.message, 'AVX2 kernel requested but not available')) ); %#ok<STREMP>
end
if has_avx2
    assertElementsAlmostEqual( V_k, V_re, 'absolute', 1e-6 );                           % Single precision
end

%% Multi-frequency mode: struct array with two entries (the second one is twice the first)
ant_m = ant;
ant_m.center_freq = 1e9;
ant_m(2) = ant_m(1);
ant_m(2).e_theta_re = 2 * ant.e_theta_re;
ant_m(2).e_theta_im = 2 * ant.e_theta_im;
ant_m(2).e_phi_re = 2 * ant.e_phi_re;
ant_m(2).e_phi_im = 2 * ant.e_phi_im;
ant_m(2).center_freq = 2e9;
[ MV_re, MV_im, MH_re, MH_im ] = quadriga_lib.arrayant_interpolate( ant_m, az, el, [], [], [], [1e9, 1.5e9, 2e9] );
assertEqual( size(MV_re), [2, 3, 3] );
assertElementsAlmostEqual( MV_re, cat(3, V_re, 1.5 * V_re, 2 * V_re), 'absolute', 1e-12 ); % In phase: linear
assertElementsAlmostEqual( MV_im, cat(3, V_im, 1.5 * V_im, 2 * V_im), 'absolute', 1e-12 );
assertElementsAlmostEqual( MH_re, cat(3, H_re, 1.5 * H_re, 2 * H_re), 'absolute', 1e-12 );
assertElementsAlmostEqual( MH_im, cat(3, H_im, 1.5 * H_im, 2 * H_im), 'absolute', 1e-12 );
MV_d = quadriga_lib.arrayant_interpolate( ant_m, az, el );                              % Default: center frequencies
assertEqual( MV_d, MV_re(:, :, [1, 3]) );
assertEqual( quadriga_lib.arrayant_interpolate( ant_m, az, el, [], [], [], [] ), MV_d );
MV_e = quadriga_lib.arrayant_interpolate( ant_m, az, el, [2, 1, 2], [], [], 1.5e9 );     % Element indices
assertEqual( MV_e, MV_re([2, 1, 2], :, 2) );
MV_1 = quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [1e9, 2e9] );        % 1x1 struct with frequency
assertEqual( MV_1, cat(3, V_re, V_re) );

%% Outputs: none, and fewer than the maximum
quadriga_lib.arrayant_interpolate( ant, az, el );
[ V_2, V_i2 ] = quadriga_lib.arrayant_interpolate( ant, az, el );
assertEqual( V_2, V_re );
assertEqual( V_i2, V_im );
[ ~, ~, ~, ~, dist_5 ] = quadriga_lib.arrayant_interpolate( ant_p, az, el );
[ ~, ~, ~, ~, dist_8, ~, ~, gamma_8 ] = quadriga_lib.arrayant_interpolate( ant_p, az, el );
assertEqual( dist_5, dist_8 );
assertEqual( gamma_8, gamma );

%% Type handling and shapes: single precision inputs, empty angle input
[ V_s, ~, ~, H_s ] = quadriga_lib.arrayant_interpolate( ant, single(az), single(el) );
assertTrue( isa(V_s, 'double') );
assertElementsAlmostEqual( V_s, V_re, 'absolute', 1e-6 );                               % Angles rounded to single
assertElementsAlmostEqual( H_s, H_im, 'absolute', 1e-6 );
[ V_0, ~, ~, ~, dist_0 ] = quadriga_lib.arrayant_interpolate( ant, zeros(1, 0), zeros(1, 0) );
assertEqual( size(V_0), [2, 0] );
assertEqual( size(dist_0), [2, 0] );
MV_0 = quadriga_lib.arrayant_interpolate( ant_m, zeros(2, 0), zeros(2, 0) );
assertEqual( size(MV_0), [2, 0, 2] );

%% Errors
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, az ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [], 0, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.arrayant_interpolate( [], az, el, [], [], [], [], [-2, 2], [-1, 1], [3, 1], ...
    [6, 2], [0, pi], 0 ), 'Wrong number of input arguments.' );                          % Separate fields: removed
assert_throws( @() out9( ant, az, el ), 'Wrong number of output arguments.' );
assert_throws( @() out5( ant_m, az, el ), 'Multi-frequency mode supports at most 4 outputs' );
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, az, el, [0, 1] ), 'cannot be 0 (1-based index)' );
assert_throws( @() quadriga_lib.arrayant_interpolate( ant_m([]), az, el ), 'Input ''arrayant'' cannot be empty.' );
assert_throws( @() quadriga_lib.arrayant_interpolate( 1, az, el ), 'Input must be a struct.' );
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, [], [] ), 'Input ''azimuth'' must have 1 or n_out rows.' ); % C++
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, az, el(1:2) ), ...
    'Input ''elevation'' must have the same size as ''azimuth''.' );                   % C++
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, az, el, 3 ), 'exceed the number of elements' ); % C++
assert_throws( @() quadriga_lib.arrayant_interpolate( ant, az, el, [], [], [], [], 3 ), ...
    'Input ''use_kernel'' must be 0, 1 or 2.' );                                       % C++
assert_throws( @() quadriga_lib.arrayant_interpolate( ant_m, az, el, [], [], [], [1e9, NaN] ), ...
    'Input ''frequency'' must contain finite values.' );                               % C++, multi-frequency
assert_throws( @() out0( ant, az, el(1:2) ), 'must have the same size' );                % C++ called with 0 outputs

end

% Call without outputs
function out0(varargin)
quadriga_lib.arrayant_interpolate( varargin{:} );
end

% Request one output more than the wrapper allows
function out9(varargin)
[~, ~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.arrayant_interpolate( varargin{:} ); %#ok<ASGLU>
end

% Request 5 outputs in the multi-frequency mode
function out5(varargin)
[~, ~, ~, ~, ~] = quadriga_lib.arrayant_interpolate( varargin{:} ); %#ok<ASGLU>
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
