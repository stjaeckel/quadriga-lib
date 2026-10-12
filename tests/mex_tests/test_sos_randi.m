function test_sos_randi
% MEX interface test for quadriga_lib.sos_randi.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_rand.cpp).

%% Setup
dd  = [5, 20, 0];
pos = [0, 0, 0; 3, 0, 0; 0, 4, 1.5; 100, -50, 2];
pos_b = pos(:, [2, 1, 3]) + 1;

%% Basic call
val = quadriga_lib.sos_randi( dd, pos, 6, [], [], 'Comb256', 7 );
assertEqual( size(val), [4, 3] );                                              % [n_pos, n_gen]
assertTrue( isa(val, 'double') );
assertTrue( all(val(:) >= 1) && all(val(:) <= 6) && all(val(:) == round(val(:))) );

%% The result is the documented composition of preset, phases and evaluation
[ freq, acf, dist ] = quadriga_lib.sos_preset( 'Comb256', 20 );
phase = quadriga_lib.sos_init_phases( 256, 6, 7 );
ref = quadriga_lib.sos_val( freq, phase(:, 2), pos, [], [], [], [], [], true );
assertElementsAlmostEqual( val(:, 2), ceil( 6 * ref ), 'absolute', 1e-12 );
freq0 = quadriga_lib.sos_preset( 'Comb256', 0.1 );                             % 0 is replaced by 0.1 m
ref0 = quadriga_lib.sos_val( freq0, phase(:, 3), pos, [], [], [], [], [], true );
assertElementsAlmostEqual( val(:, 3), ceil( 6 * ref0 ), 'absolute', 1e-12 );

%% Optional inputs: omitted, empty, explicit
val2 = quadriga_lib.sos_randi( dd, pos, 6, [], [], [], 7 );                            % default preset
assertEqual( val2, val );
val3 = quadriga_lib.sos_randi( dd, pos, 6, [], [], 'Exp128', 7 );
assertEqual( size(val3), [4, 3] );
r1 = quadriga_lib.sos_randi( dd, pos, 6 );                                             % seed -1: new realisation
assertEqual( size(r1), [4, 3] );
assertTrue( all(r1(:) >= 1) && all(r1(:) <= 6) );
v1 = quadriga_lib.sos_randi( dd, pos, 1, [], [], [], 7 );                      % imax reaches the library
assertEqual( v1, ones(4, 3) );
vb = quadriga_lib.sos_randi( dd, pos, 1000, [], [], [], 7 );
ub = quadriga_lib.sos_rand( dd, pos, [], [], [], 7 );
assertEqual( vb, ceil( 1000 * ub ) );

% Two devices, same map
val4 = quadriga_lib.sos_randi( dd, pos, 6, pos_b, [], [], 7 );
ref4 = quadriga_lib.sos_val( freq, phase(:, 2), pos, pos_b, [], acf, dist, [], true );
assertElementsAlmostEqual( val4(:, 2), ceil( 6 * ref4 ), 'absolute', 1e-12 );
val5 = quadriga_lib.sos_randi( dd, pos, 6, pos_b, dd, [], 7 );                         % equal distances: same map
assertEqual( val5, val4 );

% Two devices, different decorrelation distance for generator 2: two maps, phases from column n_gen + 2
val6 = quadriga_lib.sos_randi( dd, pos, 6, pos_b, [5, 8, 0], [], 7 );
ref6 = quadriga_lib.sos_val( freq, phase(:, 2), pos, pos_b, phase(:, 5), [], [], 20/8, true );
assertElementsAlmostEqual( val6(:, 2), ceil( 6 * ref6 ), 'absolute', 1e-12 );
assertEqual( val6(:, [1, 3]), val4(:, [1, 3]) );

%% Outputs: none
quadriga_lib.sos_randi( dd, pos, 6 );

%% Type handling and orientation
val7 = quadriga_lib.sos_randi( single(dd'), single(pos), uint32(6), [], [], 'Comb256', int32(7) );
assertTrue( isa(val7, 'double') );
assertEqual( size(val7), [4, 3] );
assertTrue( mean(val7(:) == val(:)) >= 11/12 );                                % single-precision positions may move one value across a step

%% Errors
assert_throws( @() quadriga_lib.sos_randi( dd ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_randi( dd, pos, 6, [], [], 'Comb256', 7, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( dd, pos, 6 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_randi( [5, -1], pos, 6 ), 'dist_decorr' );                     % thrown by C++
assert_throws( @() quadriga_lib.sos_randi( dd, pos, 6, [], dd ), 'dist_decorr_b' );                % thrown by C++
assert_throws( @() quadriga_lib.sos_randi( dd, pos, 6, [], [], 'Comb300' ), 'name' );              % thrown by C++
assert_throws( @() quadriga_lib.sos_randi( dd, pos, 0 ), 'imax' );                           % thrown by C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.sos_randi( varargin{:} ); %#ok<ASGLU>
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
