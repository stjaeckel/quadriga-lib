function test_sos_acf_approx
% MEX interface test for quadriga_lib.sos_acf_approx.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_acf_approx.cpp).

%% Setup: one sinusoid with three different components; 4 samples with 2 m spacing -> 13 grid points
freq = [0.03, 0.05, -0.02];
dist = [0, 2, 4, 6];
ax = (-12:2:12)';

%% Basic call, 3-D: planes x-y, x-z, y-z; column = first axis, row = second axis
[ acf_approx, dist_2d ] = quadriga_lib.sos_acf_approx( freq, dist );
assertEqual( size(acf_approx), [13, 13, 3] );
assertEqual( size(dist_2d), [13, 1] );
assertTrue( isa(acf_approx, 'double') );
assertElementsAlmostEqual( dist_2d, ax, 'absolute', 1e-13 );
[c, r] = meshgrid( ax, ax );                                                   % c: column axis, r: row axis
assertElementsAlmostEqual( acf_approx(:, :, 1), cos( 2*pi * ( 0.03 * c + 0.05 * r ) ), 'absolute', 1e-12 );
assertElementsAlmostEqual( acf_approx(:, :, 2), cos( 2*pi * ( 0.03 * c - 0.02 * r ) ), 'absolute', 1e-12 );
assertElementsAlmostEqual( acf_approx(:, :, 3), cos( 2*pi * ( 0.05 * c - 0.02 * r ) ), 'absolute', 1e-12 );

%% 2-D and 1-D shapes
a2 = quadriga_lib.sos_acf_approx( freq(1:2), dist );
assertEqual( size(a2), [13, 13] );
assertElementsAlmostEqual( a2, acf_approx(:, :, 1), 'absolute', 1e-12 );
a1 = quadriga_lib.sos_acf_approx( freq(1), dist );
assertEqual( size(a1), [13, 1] );
assertElementsAlmostEqual( a1, cos( 2*pi * 0.03 * ax ), 'absolute', 1e-12 );

%% Several sinusoids are averaged
a4 = quadriga_lib.sos_acf_approx( [0.03; 0.07], dist );
assertElementsAlmostEqual( a4, 0.5 * ( cos( 2*pi * 0.03 * ax ) + cos( 2*pi * 0.07 * ax ) ), 'absolute', 1e-12 );

%% Outputs: none
quadriga_lib.sos_acf_approx( freq, dist );

%% Type handling and orientation
a5 = quadriga_lib.sos_acf_approx( single(freq), single(dist') );
assertTrue( isa(a5, 'double') );
assertElementsAlmostEqual( a5, acf_approx, 'absolute', 1e-5 );

%% Errors
assert_throws( @() quadriga_lib.sos_acf_approx( freq ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_acf_approx( freq, dist, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out3( freq, dist ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_acf_approx( [freq, 1], dist ), 'freq' );          % thrown by C++
assert_throws( @() quadriga_lib.sos_acf_approx( freq, [0, 2, 5] ), 'equally spaced' ); % thrown by C++
assert_throws( @() quadriga_lib.sos_acf_approx( freq, 0 ), 'dist' );                  % thrown by C++

end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.sos_acf_approx( varargin{:} ); %#ok<ASGLU>
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
