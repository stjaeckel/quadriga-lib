function test_fast_geo2cart

e = 2*pi*(rand(2,6)-0.5);

% --- Argument count errors ---

assert_cpp_error( @() quadriga_lib.fast_geo2cart, 'Wrong number of input arguments.' );
assert_cpp_error( @() quadriga_lib.fast_geo2cart(e), 'Wrong number of input arguments.' );

% The split flag is gone: a 5th input is rejected
assert_cpp_error( @() quadriga_lib.fast_geo2cart(e, e, e, 0, true), 'Wrong number of input arguments.' );

try
    [~,~,~,~,~,~,~,~] = quadriga_lib.fast_geo2cart(e, e);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- Shape errors ---

assert_cpp_error( @() quadriga_lib.fast_geo2cart([], e), 'Input ''el'' must have the same size as ''az''.' );
assert_cpp_error( @() quadriga_lib.fast_geo2cart(e, []), 'Input ''el'' must have the same size as ''az''.' );
assert_cpp_error( @() quadriga_lib.fast_geo2cart(e(1,:), e), 'Input ''el'' must have the same size as ''az''.' );
assert_cpp_error( @() quadriga_lib.fast_geo2cart(e, reshape(e, 6, 2)), 'Input ''el'' must have the same size as ''az''.' );
assert_cpp_error( @() quadriga_lib.fast_geo2cart(e, e, e(1,:)), 'Input ''len'' must have the same size as ''az''.' );

% --- Outputs preserve the input shape; single precision input is converted ---

[x,y,z] = quadriga_lib.fast_geo2cart(single(e), e, e);
assert( isa(x,'double') && isa(y,'double') && isa(z,'double') );
assert( isequal(size(x), [2,6]) && isequal(size(y), [2,6]) && isequal(size(z), [2,6]) );

% A 3D input keeps its size; the values equal those of the column-vector call
a3 = rand(2,3,4); e3 = rand(2,3,4); l3 = rand(2,3,4);
[x1,y1,z1,s1] = quadriga_lib.fast_geo2cart(a3, e3, l3, 1);
[x2,y2,z2,s2] = quadriga_lib.fast_geo2cart(a3(:), e3(:), l3(:), 1);
assertEqual( size(x1), [2,3,4] );
assertEqual( size(s1), [2,3,4] );
assertEqual( [x1(:),y1(:),z1(:),s1(:)], [x2,y2,z2,s2] );
assert_cpp_error( @() quadriga_lib.fast_geo2cart(a3, e3, reshape(l3, 6, 4)), 'Input ''len'' must have the same size as ''az''.' );

% Empty input gives empty outputs, as in C++
[x,y,z] = quadriga_lib.fast_geo2cart(zeros(0,3), zeros(0,3));
assert( isequal(size(x), [0,3]) && isequal(size(y), [0,3]) && isequal(size(z), [0,3]) );

% --- Known values; len omitted or [] = unit length ---

[x,y,z] = quadriga_lib.fast_geo2cart([0;pi/4;0], [0;0;pi/4], [1;sqrt(2);sqrt(2)]);
assertElementsAlmostEqual( [x,y,z], [1,0,0; 1,1,0; 1,0,1], 'absolute', 1e-5 );

[x,y,z] = quadriga_lib.fast_geo2cart([pi;-pi/2], [pi/4;pi/4]);
assertElementsAlmostEqual( [x,y,z], [-1,0,1; 0,-1,1]/sqrt(2), 'absolute', 1e-5 );
[x1,y1,z1] = quadriga_lib.fast_geo2cart([pi;-pi/2], [pi/4;pi/4], []);
assertEqual( [x1,y1,z1], [x,y,z] );

% --- All seven outputs equal the reference formulas element-wise (GENERIC kernel) ---

el = (pi/2)*(rand(2,6)-0.5);
le = rand(2,6) + 0.5;
[x,y,z,sin_az,cos_az,sin_el,cos_el] = quadriga_lib.fast_geo2cart(e, el, le, 1);
assertElementsAlmostEqual( x, cos(el).*cos(e).*le, 'absolute', 1e-12 );
assertElementsAlmostEqual( y, cos(el).*sin(e).*le, 'absolute', 1e-12 );
assertElementsAlmostEqual( z, sin(el).*le,         'absolute', 1e-12 );
assertElementsAlmostEqual( sin_az, sin(e),  'absolute', 1e-12 );
assertElementsAlmostEqual( cos_az, cos(e),  'absolute', 1e-12 );
assertElementsAlmostEqual( sin_el, sin(el), 'absolute', 1e-12 );
assertElementsAlmostEqual( cos_el, cos(el), 'absolute', 1e-12 );

quadriga_lib.fast_geo2cart(e, el, le, 1);   % no output: ans is x
assertEqual( ans, x );

% Fewer outputs give the same values
[x1,y1] = quadriga_lib.fast_geo2cart(e, el, le, 1);
assertEqual( [x1,y1], [x,y] );

% --- Round trip with fast_cart2geo ---

[az1,el1,le1] = quadriga_lib.fast_cart2geo(x, y, z, 1);
[x1,y1,z1] = quadriga_lib.fast_geo2cart(az1, el1, le1, 1);
assertElementsAlmostEqual( [x1,y1,z1], [x,y,z], 'absolute', 1e-12 );

% --- use_kernel ---

assertEqual( quadriga_lib.fast_geo2cart(e, el, le), quadriga_lib.fast_geo2cart(e, el, le, 0) );   % default kernel: 0 (auto)
assertElementsAlmostEqual( quadriga_lib.fast_geo2cart(e, el, le, 0), x, 'absolute', 1e-5 );

% Kernel values outside 0, 1, 2 are rejected (they used to run the auto kernel silently)
for k = [3, 99]
    assert_cpp_error( @() quadriga_lib.fast_geo2cart(0.5, 0.2, 1, k), 'Input ''use_kernel'' must be 0, 1 or 2.' );
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
