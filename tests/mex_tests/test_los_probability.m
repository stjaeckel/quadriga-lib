function test_los_probability
% MEX interface test for quadriga_lib.los_probability.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_los_probability.cpp).

%% Setup: 4 links
d = [ 5; 10; 50; 200 ];
umi = @(x) 18 ./ x + exp( -x / 36 ) .* ( 1 - 18 ./ x );                       % TR UMi for d > 18 m

%% Basic call and values
p = quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5 );
assertEqual( size(p), [4, 1] );
assertTrue( isa(p, 'double') );
assertEqual( p(1:2), [1; 1] );                                                 % d <= 18 m
assertElementsAlmostEqual( p(3:4), umi( d(3:4) ), 'absolute', 1e-14 );
assertEqual( quadriga_lib.los_probability( 'NLOS', d, 10, 1.5 ), zeros(4, 1) );
assertEqual( quadriga_lib.los_probability( 'LOS', d, 10, 1.5 ), ones(4, 1) );

% h_bs and h_ut: scalar or one value per link (QuaDRiGa_Industrial uses hB, UMa uses hU)
h_bs = [ 2; 4; 6; 10 ];
p_ind = quadriga_lib.los_probability( 'QuaDRiGa_Industrial', d, h_bs, 1.5 );
assertElementsAlmostEqual( p_ind, min( 1, exp( -(d + 20) / 80 ) .* exp( min(h_bs, 8) / 5 - 1 ) ), 'absolute', 1e-14 );
p_ind4 = quadriga_lib.los_probability( 'QuaDRiGa_Industrial', d, 6, 1.5 );
assertEqual( p_ind(3), p_ind4(3) );
h_ut = [ 1.5; 1.5; 22.5; 22.5 ];
p_uma = quadriga_lib.los_probability( '38.901_UMa', d, 25, h_ut );
p_uma1 = quadriga_lib.los_probability( '38.901_UMa', d, 25, 1.5 );
p_uma2 = quadriga_lib.los_probability( '38.901_UMa', d, 25, 22.5 );
assertEqual( p_uma, [ p_uma1(1:2); p_uma2(3:4) ] );
assertTrue( p_uma(4) > p_uma1(4) + 1e-3 );

%% Optional inputs: clutter arguments (omitted, [], NaN and the TR defaults give the same result)
p_sl = quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5 );
assertElementsAlmostEqual( p_sl, exp( -d * (-log(1 - 0.2) / 10) ), 'absolute', 1e-14 );
assertEqual( quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5, [], [], [] ), p_sl );
assertEqual( quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5, NaN, NaN, NaN ), p_sl );
assertElementsAlmostEqual( quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5, 10, 0.2 ), p_sl, 'absolute', 1e-15 );
p_d = quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5, 20 );        % d_clutter
assertElementsAlmostEqual( p_d, exp( -d * (-log(1 - 0.2) / 20) ), 'absolute', 1e-14 );
p_r = quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5, [], 0.4 );   % r_clutter
assertElementsAlmostEqual( p_r, exp( -d * (-log(1 - 0.4) / 10) ), 'absolute', 1e-14 );
p_sh = quadriga_lib.los_probability( '38.901_InF_SH', d, 8, 1.5 );           % h_c = 2 m above the UT
assertTrue( all( p_sh(2:end) < 1 ) );
assertEqual( quadriga_lib.los_probability( '38.901_InF_SH', d, 8, 1.5, [], [], 1 ), ones(4, 1) );  % UT above h_c

%% Optional inputs and outputs: soft LOS state
[ p1, s1 ] = quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5 );
assertEqual( p1, p );
assertEqual( s1, [] );                                                         % no rnd_g, no fc
rnd_g = [ 0.1; -0.2; 0.3; 0 ];
fc = [ 1e9, 30e9 ];
[ ~, s2 ] = quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, [], [], [], rnd_g );
assertEqual( s2, [] );                                                         % no fc
[ ~, s3 ] = quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, [], [], [], [], fc );
assertEqual( s3, [] );                                                         % no rnd_g
[ p4, s4 ] = quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, [], [], [], rnd_g, fc );
assertEqual( p4, p );
assertEqual( size(s4), [4, 2] );
assertEqual( s4(1:2, :), ones(2, 2) );                                         % p = 1 gives 1
lambda = 299792458 ./ fc;
F = sqrt(2) * erfinv( 2 * p(3:4) - 1 );                                        % inverse normal CDF
ref = 0.5 + atan( sqrt( 20 ./ lambda ) .* ( rnd_g(3:4) + F ) ) / pi;
assertElementsAlmostEqual( s4(3:4, :), ref, 'absolute', 1e-8 );
[ ~, s5 ] = quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, NaN, NaN, NaN, rnd_g', fc' );  % row / column
assertEqual( s5, s4 );
assertEqual( quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, [], [], [], rnd_g, fc ), p );

%% Outputs: none
quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5 );

%% Type handling and orientation
p_t = quadriga_lib.los_probability( '38.901_UMi', single(d'), int32(10), single(1.5) );
assertTrue( isa(p_t, 'double') );
assertEqual( size(p_t), [4, 1] );
assertElementsAlmostEqual( p_t, p, 'absolute', 1e-14 );                       % values exact in single
p_t2 = quadriga_lib.los_probability( '38.901_InF_SL', d, 8, 1.5, int32(20), single(0.2) );
assertElementsAlmostEqual( p_t2, p_d, 'absolute', 1e-7 );                     % r = 0.2 rounded to single
assertEqual( quadriga_lib.los_probability( '38.901_UMi', zeros(0, 1), 10, 1.5 ), zeros(0, 1) );   % no link

%% Errors
assert_throws( @() quadriga_lib.los_probability( '38.901_UMi', d, 10 ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, [], [], [], [], [], 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out3( '38.901_UMi', d, 10, 1.5 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.los_probability( 'foo', d, 10, 1.5 ), 'Model ''foo'' is not supported.' );   % C++
assert_throws( @() quadriga_lib.los_probability( '38.901_UMi', d, [ 10; 20 ], 1.5 ), ...
    'Number of elements in ''h_bs'' must be 1 or match ''d2d_out''.' );                              % C++
assert_throws( @() quadriga_lib.los_probability( '38.901_UMi', d, 10, 1.5, [], 1 ), ...
    'Input ''r_clutter'' must be NaN or in the range from 0 to 1, 1 excluded.' );                    % C++
assert_throws( @() out2( '38.901_UMi', d, 10, 1.5, [], [], [], [ 0; 1 ], fc ), ...
    'Number of elements in ''rnd_g'' does not match number of elements in ''d2d_out''.' );          % C++

end

% Request 2 outputs: the soft LOS state is computed
function out2(varargin)
[~, ~] = quadriga_lib.los_probability( varargin{:} ); %#ok<ASGLU>
end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.los_probability( varargin{:} ); %#ok<ASGLU>
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
