% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_calc_rotation_matrix
% MEX interface test for quadriga_lib.calc_rotation_matrix.

try
    quadriga_lib.calc_rotation_matrix;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% 3 inputs should throw error
try
    [~] = quadriga_lib.calc_rotation_matrix(1,1,1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% 2 outputs should throw error
try
    [~,~] = quadriga_lib.calc_rotation_matrix([0;0;0]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% empty input
try
    [~] = quadriga_lib.calc_rotation_matrix([]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orientation'' cannot be empty.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.calc_rotation_matrix(eye(2));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orientation'' must have 3 rows.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% int32 input is cast to double
R = quadriga_lib.calc_rotation_matrix(int32([0;0;0]));
assert( isa(R,'double') );
assertElementsAlmostEqual( R, [1,0,0, 0,1,0, 0,0,1]', 'absolute', 1e-5 );

% transposed flag: numeric and logical values are accepted
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],0);
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],2);
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],false);
[~] = quadriga_lib.calc_rotation_matrix([0;0;0],true);

c = cos(pi/8);
s = sin(pi/8);

R = quadriga_lib.calc_rotation_matrix([0;0;0]);
assert( isa(R,'double') );
assertElementsAlmostEqual( R, [1,0,0, 0,1,0, 0,0,1]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix(single([pi/8;0;0]));
assert( isa(R,'double') );
assertElementsAlmostEqual( R, [1,0,0,0,c,s,0,-s,c]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix([0,pi/8,0;0,0,pi/8]');
assertElementsAlmostEqual( R(:,1), [c,0,-s,0,1,0,s,0,c]', 'absolute', 1e-5 );
assertElementsAlmostEqual( R(:,2), [c,s,0,-s,c,0,0,0,1]', 'absolute', 1e-5 );

R = quadriga_lib.calc_rotation_matrix([0;-pi/8;0],1);
assertElementsAlmostEqual( R, [c,0,-s,0,1,0,s,0,c]', 'absolute', 1e-5 );

%% Malformed input is rejected by the MEX helpers
assert_cpp_error( @() quadriga_lib.calc_rotation_matrix( sparse([0;0;1]) ), 'Input ''orientation'' is sparse; sparse inputs are not supported.' );
assert_cpp_error( @() quadriga_lib.calc_rotation_matrix( [0;0;1], sparse(true) ), 'Input ''transposed'' is sparse' );
assert_cpp_error( @() quadriga_lib.calc_rotation_matrix( zeros(3,1,1,1,2) ), 'Input ''orientation'' has more than 4 dimensions.' );
assert_cpp_error( @() quadriga_lib.calc_rotation_matrix( [0;0;1], [true, false] ), 'Input ''transposed'' must be a scalar.' );

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
