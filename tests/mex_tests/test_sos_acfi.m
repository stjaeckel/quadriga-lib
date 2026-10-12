function test_sos_acfi
% MEX interface test for quadriga_lib.sos_acfi.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_acfi.cpp).

%% Setup: table with unequal spacing
acf = [1; 0.5; 0.2];
dist = [0; 1; 3];

%% Basic call and values
val = quadriga_lib.sos_acfi( acf, dist, [0, 0.5, 1, 2, 3, 10] );
assertEqual( size(val), [6, 1] );                                              % returned as column
assertTrue( isa(val, 'double') );
assertElementsAlmostEqual( val, [1; 0.75; 0.5; 0.35; 0.2; 0.2], 'absolute', 1e-14 );

%% Orientation: rows and columns are accepted
val2 = quadriga_lib.sos_acfi( acf', dist', [0; 0.5; 1; 2; 3; 10] );
assertEqual( val2, val );
val3 = quadriga_lib.sos_acfi( acf, dist, [] );
assertTrue( isempty(val3) );

%% Outputs: none
quadriga_lib.sos_acfi( acf, dist, 0.5 );

%% Type handling
val4 = quadriga_lib.sos_acfi( single(acf), single(dist), single([0.5, 2]) );
assertTrue( isa(val4, 'double') );
assertElementsAlmostEqual( val4, [0.75; 0.35], 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.sos_acfi( acf, dist ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_acfi( acf, dist, 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( acf, dist, 1 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_acfi( acf, [0; 1], 1 ), 'dist' );           % thrown by C++
assert_throws( @() quadriga_lib.sos_acfi( acf, dist, -1 ), 'd' );               % thrown by C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.sos_acfi( varargin{:} ); %#ok<ASGLU>
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
