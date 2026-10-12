function test_sos_preset
% MEX interface test for quadriga_lib.sos_preset.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_preset.cpp).

%% Basic call, all outputs
[ freq, acf, dist ] = quadriga_lib.sos_preset( 'Comb256' );
assertEqual( size(freq), [256, 3] );
assertEqual( size(acf), [200, 1] );
assertEqual( size(dist), [200, 1] );
assertTrue( isa(freq, 'double') && isa(acf, 'double') && isa(dist, 'double') );
assertElementsAlmostEqual( dist([1, 2, 200]), [0; 0.25; 49.75], 'absolute', 1e-13 );
assertElementsAlmostEqual( acf([1, 41]), [1; exp(-1)], 'absolute', 1e-14 );   % R(0) and R(10 m)
assertTrue( max(abs(freq(:))) > 0.01 );

%% Name selects type and size
freq_e = quadriga_lib.sos_preset( 'Exp128' );
assertEqual( size(freq_e), [128, 3] );
[ ~, acf_g ] = quadriga_lib.sos_preset( 'Gauss1024' );
assertElementsAlmostEqual( acf_g(81), exp(-4), 'absolute', 1e-14 );             % Gauss at 20 m
assertElementsAlmostEqual( acf(81), exp(-2), 'absolute', 1e-14 );               % Comb at 20 m

%% Optional input: omitted, empty, explicit
[ freq2, acf2, dist2 ] = quadriga_lib.sos_preset( 'Comb256', [] );
assertEqual( freq2, freq );
assertEqual( acf2, acf );
assertEqual( dist2, dist );
[ freq3, acf3, dist3 ] = quadriga_lib.sos_preset( 'Comb256', 25 );
assertElementsAlmostEqual( dist3, dist * 2.5, 'absolute', 1e-12 );
assertElementsAlmostEqual( freq3 * 2.5, freq, 'absolute', 1e-14 );
assertElementsAlmostEqual( acf3, acf, 'absolute', 1e-14 );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_preset( 'Comb256' );
freq4 = quadriga_lib.sos_preset( 'Comb256' );
assertEqual( freq4, freq );
[ ~, acf5 ] = quadriga_lib.sos_preset( 'Comb256' );
assertEqual( acf5, acf );

%% Type handling
freq6 = quadriga_lib.sos_preset( 'Comb256', single(25) );
assertEqual( freq6, freq3 );
freq7 = quadriga_lib.sos_preset( 'Comb256', int32(25) );
assertEqual( freq7, freq3 );

%% Errors
assert_throws( @() quadriga_lib.sos_preset(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_preset( 'Comb256', 10, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( 'Comb256' ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_preset( 'Comb300' ), 'name' );              % thrown by C++
assert_throws( @() quadriga_lib.sos_preset( 'Comb256', 0 ), 'dist_decorr' );    % thrown by C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.sos_preset( varargin{:} ); %#ok<ASGLU>
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
