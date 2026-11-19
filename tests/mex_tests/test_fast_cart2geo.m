% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_fast_cart2geo
% MEX interface test for quadriga_lib.fast_cart2geo.

[x, y, z] = deal(rand(6,2), rand(6,2), rand(6,2));
v = 1/sqrt(2);

%% Argument count errors

assert_cpp_error( @() quadriga_lib.fast_cart2geo, 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.fast_cart2geo(x), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.fast_cart2geo(x, y), 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.fast_cart2geo(x, y, z, 0, 0), 'Wrong number of input arguments.' );

try
    [~,~,~,~] = quadriga_lib.fast_cart2geo(x, y, z);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Shape errors, also with an equal number of elements

assert_cpp_error( @() quadriga_lib.fast_cart2geo(rand(4,1), rand(5,1), rand(4,1)), 'Input ''y'' must have the same size as ''x''.' );
assert_cpp_error( @() quadriga_lib.fast_cart2geo(rand(4,1), rand(4,1), rand(1,4)), 'Input ''z'' must have the same size as ''x''.' );

% y and z cannot be empty
assert_cpp_error( @() quadriga_lib.fast_cart2geo(rand(3,6,2), [], []), 'Input ''y'' must have the same size as ''x''.' );

%% Known values

[az,el,le] = quadriga_lib.fast_cart2geo([-v;0], [-v;2*v], [0;2*v]);
assertElementsAlmostEqual( az, [-3*pi/4; pi/2], 'absolute', 1e-5 );
assertElementsAlmostEqual( el, [0; pi/4],       'absolute', 1e-5 );
assertElementsAlmostEqual( le, [1; 2],          'absolute', 1e-5 );

%% Outputs preserve the input shape and type

[az,el,le] = quadriga_lib.fast_cart2geo(x, y, z);
quadriga_lib.fast_cart2geo(x, y, z);   % no output: ans is az
assertEqual( ans, az );
assert( isa(az,'double') && isa(el,'double') && isa(le,'double') );
assert( isequal(size(az), [6,2]) && isequal(size(el), [6,2]) && isequal(size(le), [6,2]) );

[az1,el1,le1] = quadriga_lib.fast_cart2geo(x', y', z');
assertEqual( az1, az' );
assertEqual( el1, el' );
assertEqual( le1, le' );

% A 3D input keeps its size; the values equal those of the column-vector call
x3 = rand(2,3,4); y3 = rand(2,3,4); z3 = rand(2,3,4);
[az1,el1,le1] = quadriga_lib.fast_cart2geo(x3, y3, z3, 1);
[az2,el2,le2] = quadriga_lib.fast_cart2geo(x3(:), y3(:), z3(:), 1);
assertEqual( size(az1), [2,3,4] );
assertEqual( [az1(:),el1(:),le1(:)], [az2,el2,le2] );
assert_cpp_error( @() quadriga_lib.fast_cart2geo(x3, reshape(y3, 2, 12), z3), 'Input ''y'' must have the same size as ''x''.' );

% Single precision input is converted
[~,~,le1] = quadriga_lib.fast_cart2geo(single(x), y, z);
assertElementsAlmostEqual( le1, sqrt(double(single(x)).^2 + y.^2 + z.^2), 'absolute', 1e-12 );

% Empty input gives empty outputs, as in C++
[az1,el1,le1] = quadriga_lib.fast_cart2geo(zeros(0,3), zeros(0,3), zeros(0,3));
assert( isequal(size(az1), [0,3]) && isequal(size(el1), [0,3]) && isequal(size(le1), [0,3]) );

%% Outputs equal the reference formulas element-wise (GENERIC kernel)

[az1,el1,le1] = quadriga_lib.fast_cart2geo(x, y, z, 1);
le_ref = sqrt(x.^2 + y.^2 + z.^2);
assertElementsAlmostEqual( az1, atan2(y, x),                          'absolute', 1e-12 );
assertElementsAlmostEqual( el1, asin(min(max(z ./ le_ref, -1), 1)),   'absolute', 1e-12 );
assertElementsAlmostEqual( le1, le_ref,                               'absolute', 1e-12 );

%% use_kernel variants

[az0,el0,le0] = quadriga_lib.fast_cart2geo(x, y, z, 0);  % auto
assertElementsAlmostEqual( az0, az1, 'absolute', 1e-5 );
assertElementsAlmostEqual( el0, el1, 'absolute', 1e-5 );
assertElementsAlmostEqual( le0, le1, 'absolute', 1e-5 );

% use_kernel=[] and an omitted use_kernel fall back to the default (0, auto)
assertEqual( quadriga_lib.fast_cart2geo(x, y, z, []), az0 );
assertEqual( quadriga_lib.fast_cart2geo(x, y, z), az0 );

% Kernel values outside 0, 1, 2 are rejected
for k = [3, 99]
    assert_cpp_error( @() quadriga_lib.fast_cart2geo(x, y, z, k), 'Input ''use_kernel'' must be 0, 1 or 2.' );
end

end

% Assert that fn() raises quadriga_lib:CPPerror with a message that contains expected (substring).
function assert_cpp_error(fn, expected)
try
    [~] = fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, 'quadriga_lib:CPPerror') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: quadriga_lib:CPPerror "', expected, '", GOT: ', ...
            ME.identifier, ' "', ME.message, '"']);
    end
end
end
