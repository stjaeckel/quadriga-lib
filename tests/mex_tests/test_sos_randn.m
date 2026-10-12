function test_sos_randn
% MEX interface test for quadriga_lib.sos_randn.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_rand.cpp).

%% Setup
dd  = [5, 20, 0];
pos = [0, 0, 0; 3, 0, 0; 0, 4, 1.5; 100, -50, 2];
pos_b = pos(:, [2, 1, 3]) + 1;

%% Basic call
val = quadriga_lib.sos_randn( dd, pos, [], [], 'Comb256', 7 );
assertEqual( size(val), [4, 3] );                                              % [n_pos, n_gen]
assertTrue( isa(val, 'double') );
assertTrue( max(abs(val(:))) > 0.1 && max(abs(val(:))) < 6 );

%% The result is the documented composition of preset, phases and evaluation
[ freq, acf, dist ] = quadriga_lib.sos_preset( 'Comb256', 20 );
phase = quadriga_lib.sos_init_phases( 256, 6, 7 );
ref = quadriga_lib.sos_val( freq, phase(:, 2), pos, [], [], [], [], [], false );
assertElementsAlmostEqual( val(:, 2), ref, 'absolute', 1e-12 );
freq0 = quadriga_lib.sos_preset( 'Comb256', 0.1 );                             % 0 is replaced by 0.1 m
ref0 = quadriga_lib.sos_val( freq0, phase(:, 3), pos, [], [], [], [], [], false );
assertElementsAlmostEqual( val(:, 3), ref0, 'absolute', 1e-12 );

%% Optional inputs: omitted, empty, explicit
val2 = quadriga_lib.sos_randn( dd, pos, [], [], [], 7 );                            % default preset
assertEqual( val2, val );
val3 = quadriga_lib.sos_randn( dd, pos, [], [], 'Exp128', 7 );
assertTrue( max(abs(val3(:) - val(:))) > 0.01 );
r1 = quadriga_lib.sos_randn( dd, pos );                                             % seed -1: new realisation
assertEqual( size(r1), [4, 3] );
r2 = quadriga_lib.sos_randn( dd, pos );
assertTrue( max(abs(r1(:) - r2(:))) > 0.01 );

% Two devices, same map
val4 = quadriga_lib.sos_randn( dd, pos, pos_b, [], [], 7 );
ref4 = quadriga_lib.sos_val( freq, phase(:, 2), pos, pos_b, [], acf, dist, [], false );
assertElementsAlmostEqual( val4(:, 2), ref4, 'absolute', 1e-12 );
val5 = quadriga_lib.sos_randn( dd, pos, pos_b, dd, [], 7 );                         % equal distances: same map
assertEqual( val5, val4 );

% Two devices, different decorrelation distance for generator 2: two maps; the phases of the second device
% come from a second stream with the seed + 2^62, which is exact only as int64
phase_b = quadriga_lib.sos_init_phases( 256, 3, int64(7) + int64(4611686018427387904) );
val6 = quadriga_lib.sos_randn( dd, pos, pos_b, [5, 8, 0], [], 7 );
ref6 = quadriga_lib.sos_val( freq, phase(:, 2), pos, pos_b, phase_b(:, 2), [], [], 20/8, false );
assertElementsAlmostEqual( val6(:, 2), ref6, 'absolute', 1e-12 );
assertEqual( val6(:, [1, 3]), val4(:, [1, 3]) );

%% Outputs: none
quadriga_lib.sos_randn( dd, pos );

%% Type handling and orientation
val7 = quadriga_lib.sos_randn( single(dd'), single(pos), [], [], 'Comb256', int32(7) );
assertTrue( isa(val7, 'double') );
assertElementsAlmostEqual( val7, val, 'absolute', 1e-4 );

%% Errors
assert_throws( @() quadriga_lib.sos_randn( dd ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_randn( dd, pos, [], [], 'Comb256', 7, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( dd, pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_randn( [5, -1], pos ), 'dist_decorr' );                     % thrown by C++
assert_throws( @() quadriga_lib.sos_randn( dd, pos, [], dd ), 'dist_decorr_b' );                % thrown by C++
assert_throws( @() quadriga_lib.sos_randn( dd, pos, [], [], 'Comb300' ), 'name' );              % thrown by C++
assert_throws( @() quadriga_lib.sos_randn( dd, pos, [], [], [], NaN ), 'Input ''seed''' );            % wrapper
assert_throws( @() quadriga_lib.sos_randn( dd, pos, [], [], [], -2 ), 'seed' );                        % thrown by C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.sos_randn( varargin{:} ); %#ok<ASGLU>
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
