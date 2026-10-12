function test_sos_val
% MEX interface test for quadriga_lib.sos_val.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_val.cpp).

%% Setup: two sinusoids along x and y with 4 m wavelength, amplitude sqrt(2/2) = 1
% Generator 1: cos( pi/2 x ) + cos( pi/2 y );  generator 2: -sin( pi/2 x ) + cos( pi/2 y )
freq  = [0.25, 0, 0; 0, 0.25, 0];
phase = [0, pi/2; 0, 0];
pos   = [0, 0, 0; 1, 0, 0; 0, 2, 0];
ref   = [2, 1; 1, 0; 0, -1];

%% Basic call and values
val = quadriga_lib.sos_val( freq, phase, pos );
assertEqual( size(val), [3, 2] );                                              % [n_pos, n_gen]
assertTrue( isa(val, 'double') );
assertElementsAlmostEqual( val, ref, 'absolute', 1e-6 );                       % AVX2 kernel by default

%% Optional inputs: omitted, empty, explicit
val2 = quadriga_lib.sos_val( freq, phase, pos, [], [], [], [], [], [], [] );
assertEqual( val2, val );

val3 = quadriga_lib.sos_val( freq, phase, pos, [], [], [], [], [], [], 1 );     % use_kernel = GENERIC
assertElementsAlmostEqual( val3, ref, 'absolute', 1e-14 );

val4 = quadriga_lib.sos_val( freq, phase, pos, [], [], [], [], [], true, 1 );   % uniform
assertElementsAlmostEqual( val4, 0.5 * erfc( -ref / sqrt(2) ), 'absolute', 1e-14 );

% Two devices on the same map: ( s(a) + s(b) ) / sqrt( 2 + 2 A(D) )
acf = [1; 0.5; 0]; dist = [0; 1; 2];
pos_b = [0, 0, 0; 0, 0, 0; 0, 0, 0];                                           % distances 0, 1, 2 -> A = 1, 0.5, 0
ref5 = [ (2+2)/2, (1+1)/2; (1+2)/sqrt(3), (0+1)/sqrt(3); (0+2)/sqrt(2), (-1+1)/sqrt(2) ];
val5 = quadriga_lib.sos_val( freq, phase, pos, pos_b, [], acf, dist, [], [], 1 );
assertElementsAlmostEqual( val5, ref5, 'absolute', 1e-14 );

% Two devices on different maps: the second device uses phase_b, here the two generators swapped
phase_b = phase(:, [2, 1]);
ref6 = [ (2+1)/sqrt(2), (1+2)/sqrt(2); (1+1)/sqrt(2), (0+2)/sqrt(2); (0+1)/sqrt(2), (-1+2)/sqrt(2) ];
val6 = quadriga_lib.sos_val( freq, phase, pos, pos_b, phase_b, [], [], [], [], 1 );
assertElementsAlmostEqual( val6, ref6, 'absolute', 1e-14 );

% Frequency scale of the second device: at x = 1 m a scale of 2 turns the quarter wave into a half wave
pos_b7 = [1, 0, 0; 1, 0, 0; 1, 0, 0];
val7 = quadriga_lib.sos_val( freq, phase, pos, pos_b7, phase, [], [], 2, [], 1 );
ref7 = ( ref + repmat( [0, 1], 3, 1 ) ) / sqrt(2);                             % s_b = cos(pi)+1 = 0 and -sin(pi)+1 = 1
assertElementsAlmostEqual( val7, ref7, 'absolute', 1e-14 );

%% Outputs: none
quadriga_lib.sos_val( freq, phase, pos );

%% Type handling
val8 = quadriga_lib.sos_val( single(freq), single(phase), single(pos), [], [], [], [], single(1), false, int32(1) );
assertTrue( isa(val8, 'double') );
assertElementsAlmostEqual( val8, ref, 'absolute', 1e-6 );

%% Orientation and shape: positions with fewer columns, frequencies with fewer columns
val9 = quadriga_lib.sos_val( freq, phase, pos(:, 1:2), [], [], [], [], [], [], 1 );
assertElementsAlmostEqual( val9, ref, 'absolute', 1e-14 );
val10 = quadriga_lib.sos_val( freq(1, 1), 0, [0; 1; 2] );                       % 1-D: sqrt(2) cos( pi/2 x )
assertEqual( size(val10), [3, 1] );
assertElementsAlmostEqual( val10, sqrt(2) * [1; 0; -1], 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.sos_val( freq, phase ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_val( freq, phase, pos, [], [], [], [], 1, 0, 0, 0 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( freq, phase, pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_val( freq, phase(1, :), pos ), 'phase' );                          % thrown by C++
assert_throws( @() quadriga_lib.sos_val( freq, phase, pos, pos_b ), 'acf' );                           % thrown by C++
assert_throws( @() quadriga_lib.sos_val( freq, phase, pos, pos_b, [], acf, dist, 2 ), 'freq_scale_b' ); % thrown by C++
assert_throws( @() quadriga_lib.sos_val( freq, phase, pos, [], [], [], [], [], [], 3 ), 'use_kernel' ); % thrown by C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.sos_val( varargin{:} ); %#ok<ASGLU>
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
