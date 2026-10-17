function test_sos_calc_mse
% MEX interface test for quadriga_lib.sos_calc_mse.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_calc_mse.cpp).

%% Setup: 1-D, two sinusoids; the target is their own ACF, so there is no error inside the range
freq = [0.04; 0.13];
dist = 0:0.5:20;
acf = 0.5 * ( cos( 2*pi * 0.04 * dist ) + cos( 2*pi * 0.13 * dist ) );
other = exp( -dist / 5 );

%% Basic call, 1-D
[ mse, mse_core, mse_all, acf_tail, test_dir ] = quadriga_lib.sos_calc_mse( freq, acf, dist );
assertTrue( isa(mse, 'double') && isscalar(mse) && isscalar(mse_core) && isscalar(mse_all) );
assertTrue( mse > 250 && mse_core > 250 );
assertTrue( mse_all > 0 && mse_all < 30 );
assertEqual( size(acf_tail), [4, 1] );
assertTrue( isempty(test_dir) );
tail = 0.5 * ( cos( 2*pi * 0.04 * 20 * [2; 4; 10; 20] ) + cos( 2*pi * 0.13 * 20 * [2; 4; 10; 20] ) );
assertElementsAlmostEqual( acf_tail, tail, 'absolute', 1e-11 );

% A wrong target: error from the definition
mse2 = quadriga_lib.sos_calc_mse( freq, other, dist );
assertElementsAlmostEqual( mse2, -10 * log10( mean( ( other - acf ).^2 ) ), 'absolute', 1e-9 );

%% 3-D: test directions and the optional input n_dir
f3 = [0.03, 0.05, -0.02; -0.07, 0.01, 0.04];
[ m3, c3, a3, t3, dir3 ] = quadriga_lib.sos_calc_mse( f3, other, dist );
assertEqual( size(dir3), [10, 2] );                                            % default n_dir = 10
assertElementsAlmostEqual( sin( dir3(:, 2) ), ( (0:9)' + 0.5 ) / 10, 'absolute', 1e-13 );
assertTrue( isscalar(c3) && isscalar(a3) && c3 > 0 && a3 > 0 );
assertEqual( size(t3), [4, 1] );
[ m4, ~, ~, ~, dir4 ] = quadriga_lib.sos_calc_mse( f3, other, dist, [] );
assertEqual( m4, m3 );
assertEqual( dir4, dir3 );
[ m5, ~, ~, ~, dir5 ] = quadriga_lib.sos_calc_mse( f3, other, dist, 7 );
assertEqual( size(dir5), [7, 2] );
assertTrue( abs(m5 - m3) > 1e-6 );

%% 2-D: one column of azimuth angles
[ ~, ~, ~, ~, dir6 ] = quadriga_lib.sos_calc_mse( f3(:, 1:2), other, dist, 4 );
assertEqual( size(dir6), [4, 1] );
assertElementsAlmostEqual( dir6, pi * (0:3)' / 4, 'absolute', 1e-14 );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_calc_mse( f3, other, dist );
m7 = quadriga_lib.sos_calc_mse( f3, other, dist );
assertEqual( m7, m3 );
[ ~, c8 ] = quadriga_lib.sos_calc_mse( f3, other, dist );
assertEqual( c8, c3 );
[ ~, ~, a9 ] = quadriga_lib.sos_calc_mse( f3, other, dist );
assertEqual( a9, a3 );

%% Type handling and orientation
m10 = quadriga_lib.sos_calc_mse( single(f3), single(other'), single(dist'), uint32(10) );
assertTrue( isa(m10, 'double') );
assertElementsAlmostEqual( m10, m3, 'absolute', 1e-3 );

%% Errors
assert_throws( @() quadriga_lib.sos_calc_mse( freq, acf ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_calc_mse( freq, acf, dist, 10, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out6( freq, acf, dist ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_calc_mse( freq, acf, dist(1:5) ), 'dist' );       % thrown by C++
assert_throws( @() quadriga_lib.sos_calc_mse( freq, acf, dist, 0 ), 'n_dir' );        % thrown by C++

% n_dir is validated by the wrapper before the conversion to an unsigned type
assert_cpp_error( @() quadriga_lib.sos_calc_mse( freq, acf, dist, -1 ), 'Input ''n_dir'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_calc_mse( freq, acf, dist, NaN ), 'Input ''n_dir'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_calc_mse( freq, acf, dist, Inf ), 'Input ''n_dir'' must be an integer >= 1.' );

end

% Request 6 outputs so that the output-count guard fires
function out6(varargin)
[~, ~, ~, ~, ~, ~] = quadriga_lib.sos_calc_mse( varargin{:} ); %#ok<ASGLU>
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

% Assert that fn() raises quadriga_lib:CPPerror with a message that contains expected (substring).
function assert_cpp_error(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, 'quadriga_lib:CPPerror') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: quadriga_lib:CPPerror "', expected, '", GOT: ', ...
            ME.identifier, ' "', ME.message, '"']);
    end
end
end
