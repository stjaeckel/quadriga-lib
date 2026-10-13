function test_sos_acf_2d
% MEX interface test for quadriga_lib.sos_acf_2d.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_acf_2d.cpp).

%% Setup: 3 samples with 1 m spacing -> grid of 9 points from -4 m to 4 m, center (5, 5)
acf = [1, 0.5, 0.25];
dist = [0, 1, 2];

%% Basic call and values
[ acf_2d, dist_2d ] = quadriga_lib.sos_acf_2d( acf, dist );
assertEqual( size(acf_2d), [9, 9] );
assertEqual( size(dist_2d), [9, 1] );
assertTrue( isa(acf_2d, 'double') && isa(dist_2d, 'double') );
assertElementsAlmostEqual( dist_2d, (-4:4)', 'absolute', 1e-14 );
assertElementsAlmostEqual( acf_2d(5, 5), 1, 'absolute', 1e-14 );
assertElementsAlmostEqual( acf_2d(5, 6), 0.5, 'absolute', 1e-14 );
assertElementsAlmostEqual( acf_2d(6, 6), 0.5 - (sqrt(2) - 1) * 0.25, 'absolute', 1e-14 );
assertElementsAlmostEqual( acf_2d(5, 7), 0.25, 'absolute', 1e-14 );
assertElementsAlmostEqual( acf_2d(5, 8), 0, 'absolute', 1e-14 );
assertElementsAlmostEqual( acf_2d, acf_2d', 'absolute', 1e-14 );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_acf_2d( acf, dist );
a2 = quadriga_lib.sos_acf_2d( acf, dist );
assertEqual( a2, acf_2d );

%% Type handling and orientation
[ a3, d3 ] = quadriga_lib.sos_acf_2d( single(acf'), single(dist' * 2.5) );
assertTrue( isa(a3, 'double') );
assertElementsAlmostEqual( a3, acf_2d, 'absolute', 1e-6 );
assertElementsAlmostEqual( d3, 2.5 * (-4:4)', 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.sos_acf_2d( acf ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_acf_2d( acf, dist, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out3( acf, dist ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_acf_2d( acf, [0, 1, 3] ), 'equally spaced' );     % thrown by C++
assert_throws( @() quadriga_lib.sos_acf_2d( 1, 0 ), 'acf' );                          % thrown by C++

end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.sos_acf_2d( varargin{:} ); %#ok<ASGLU>
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
