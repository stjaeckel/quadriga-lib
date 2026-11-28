% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_sos_val
% MEX interface test for quadriga_lib.sos_val.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_val.cpp).

%% Setup: two sinusoids along x and y with 4 m wavelength, amplitude sqrt(2/2) = 1
% Generator 1: cos( pi/2 x ) + cos( pi/2 y );  generator 2: -sin( pi/2 x ) + cos( pi/2 y )
spatial_freq  = [0.25, 0, 0; 0, 0.25, 0];
phase = [0, pi/2; 0, 0];
pos   = [0, 0, 0; 1, 0, 0; 0, 2, 0];
ref   = [2, 1; 1, 0; 0, -1];

%% Basic call and values
val = quadriga_lib.sos_val( spatial_freq, phase, pos );
assertEqual( size(val), [3, 2] );                                              % [n_pos, n_gen]
assertTrue( isa(val, 'double') );
assertElementsAlmostEqual( val, ref, 'absolute', 1e-6 );                       % AVX2 kernel by default

%% Optional inputs: omitted, empty, explicit
val2 = quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], [] );
assertEqual( val2, val );

val3 = quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1 );             % use_kernel = GENERIC
assertElementsAlmostEqual( val3, ref, 'absolute', 1e-14 );
assertEqual( quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], false, [], 1 ), val3 );  % reciprocal: no effect for one device

val4 = quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], true, 1 );           % uniform
assertElementsAlmostEqual( val4, 0.5 * erfc( -ref / sqrt(2) ), 'absolute', 1e-14 );

% Two devices, reciprocal (default): the value does not change when the devices swap positions
pos_b = [0.5, 1, 0; 2, 0, 0; 1, 3, 0];
val5 = quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, [], [], [], 1 );
assertEqual( size(val5), [3, 2] );
assertEqual( quadriga_lib.sos_val( spatial_freq, phase, pos_b, pos, [], [], [], 1 ), val5 );

% Two devices, not reciprocal: with two sinusoids the permutation swaps them,
% s(a, b) = cos( pi/2 (a_x + b_y) + phi_1 ) + cos( pi/2 (a_y + b_x) + phi_2 )
ref6 = cos( pi/2 * ( pos(:, 1) + pos_b(:, 2) ) + phase(1, :) ) + cos( pi/2 * ( pos(:, 2) + pos_b(:, 1) ) + phase(2, :) );
val6 = quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, [], false, [], 1 );
assertElementsAlmostEqual( val6, ref6, 'absolute', 1e-14 );
assertTrue( max(abs(val6(:) - val5(:))) > 0.1 );

% Second device at the origin, not reciprocal: the value of one device, with any frequency scale
val7 = quadriga_lib.sos_val( spatial_freq, phase, pos, zeros(3, 3), [], false, [], 1 );
assertElementsAlmostEqual( val7, val3, 'absolute', 1e-14 );
val7 = quadriga_lib.sos_val( spatial_freq, phase, pos, zeros(3, 3), 2, false, [], 1 );
assertElementsAlmostEqual( val7, val3, 'absolute', 1e-14 );

% Frequency scale of the second device: a scale of 2 equals doubled positions of the second device
val7 = quadriga_lib.sos_val( spatial_freq, phase, zeros(3, 3), pos_b, 2, false, [], 1 );
ref7 = quadriga_lib.sos_val( spatial_freq, phase, zeros(3, 3), 2 * pos_b, [], false, [], 1 );
assertElementsAlmostEqual( val7, ref7, 'absolute', 1e-14 );

%% Outputs: none
quadriga_lib.sos_val( spatial_freq, phase, pos );

%% Type handling
val8 = quadriga_lib.sos_val( single(spatial_freq), single(phase), single(pos), [], single(1), true, false, int32(1) );
assertTrue( isa(val8, 'double') );
assertElementsAlmostEqual( val8, ref, 'absolute', 1e-6 );

%% Orientation and shape: positions with fewer columns, frequencies with fewer columns
val9 = quadriga_lib.sos_val( spatial_freq, phase, pos(:, 1:2), [], [], [], [], 1 );
assertElementsAlmostEqual( val9, ref, 'absolute', 1e-14 );
val10 = quadriga_lib.sos_val( spatial_freq(1, 1), 0, [0; 1; 2] );                       % 1-D: sqrt(2) cos( pi/2 x )
assertEqual( size(val10), [3, 1] );
assertElementsAlmostEqual( val10, sqrt(2) * [1; 0; -1], 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, pos, [], 1, true, false, 0, 0 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( spatial_freq, phase, pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase(1, :), pos ), 'phase' );                          % thrown by C++
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b(1:2, :) ), 'pos_b' );                  % thrown by C++
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, pos, [], 2, false ), ...
    'Input ''spatial_freq_scale_b'' other than 1 requires ''pos_b''.' );                                         % thrown by C++
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, 2 ), ...
    'Input ''spatial_freq_scale_b'' other than 1 requires ''reciprocal = false''.' );                            % thrown by C++
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 3 ), 'use_kernel' );         % thrown by C++

%% Type handling of the shared MEX helpers, pinned so that a change is noticed
% Complex arrays are rejected, also when their class matches the type the wrapper reads in place
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, complex(pos, pos + 1) ), 'is complex; complex inputs are not supported.' );
assert_throws( @() quadriga_lib.sos_val( spatial_freq, complex(phase, 1), pos ), 'is complex; complex inputs are not supported.' );
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, complex(single(pos), 1) ), 'is complex; complex inputs are not supported.' );
% All real classes are read: 8- and 16-bit integers and logical arrays as their double values
assertEqual( quadriga_lib.sos_val( spatial_freq, phase, uint8(pos) ), quadriga_lib.sos_val( spatial_freq, phase, double(uint8(pos)) ) );
assertEqual( quadriga_lib.sos_val( spatial_freq, phase, pos > 0 ), quadriga_lib.sos_val( spatial_freq, phase, double(pos > 0) ) );
assert_cpp_error( @() quadriga_lib.sos_val( spatial_freq, phase, @sin ), 'Input ''pos'' has an unsupported data type.' );  % no data pointer
assert_cpp_error( @() quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], @sin ), ...
    'Input ''use_kernel'' has an unsupported data type.' );
vi = quadriga_lib.sos_val( spatial_freq, phase, int32(pos), [], [], [], [], 1 );
assertEqual( vi, quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1 ) );
% Scalars: complex and non-integer values for integer arguments are rejected, uint8 is read
assert_throws( @() quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, complex(1, 1), false ), 'is complex; complex inputs are not supported.' );
assertEqual( quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], uint8(1) ), ...
    quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1 ) );
assert_cpp_error( @() quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1.7 ), ...
    'Input ''use_kernel'' must be an integer from -2147483648 to 2147483647.' );
vk = quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1.0 );    % use_kernel = 1
assertEqual( vk, quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1 ) );

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
