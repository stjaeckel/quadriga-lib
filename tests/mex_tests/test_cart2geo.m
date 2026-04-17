function test_cart2geo

e = rand(3,6,2);
v = 1/sqrt(2);

% --- Argument count errors ---

try
    quadriga_lib.cart2geo;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~,~,~] = quadriga_lib.cart2geo(e);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- Input format errors ---

try
    quadriga_lib.cart2geo(e,0,[]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong input argument format.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    quadriga_lib.cart2geo(rand(2,2,2,2));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong input argument format.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% n_rows != 3, no y/z provided -> wrong format
try
    quadriga_lib.cart2geo(rand(2,4));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong input argument format.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Size mismatch between x, y, z
try
    quadriga_lib.cart2geo(rand(4,1), rand(5,1), rand(4,1));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong input argument format.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- Cube input path ---

[az,el] = quadriga_lib.cart2geo(e);
assert( isa(az,'double') );
assert( isa(el,'double') );
assert( isequal(size(az), [6,2]) );
assert( isequal(size(el), [6,2]) );

[az,el,le] = quadriga_lib.cart2geo([-v;-v;0]);
assertElementsAlmostEqual( az, -3*pi/4, 'absolute', 1e-5 );
assertElementsAlmostEqual( el, 0,        'absolute', 1e-5 );
assertElementsAlmostEqual( le, 1,        'absolute', 1e-5 );

[az,el,le] = quadriga_lib.cart2geo([0;2*v;2*v]);
assertElementsAlmostEqual( az, pi/2, 'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4, 'absolute', 1e-5 );
assertElementsAlmostEqual( le, 2,    'absolute', 1e-5 );

% [3,2] matrix treated as [3,2,1] cube -> outputs [2,1]
[az,el,le] = quadriga_lib.cart2geo([0,v;1,v;1,-1]);
assertElementsAlmostEqual( az, [pi/2;pi/4],    'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4*[1;-1],    'absolute', 1e-5 );
assertElementsAlmostEqual( le, sqrt(2)*[1;1],  'absolute', 1e-5 );
assert( isequal(size(az), [2,1]) );

% [3,3,2] cube -> outputs [3,2]
[az2,el2,le2] = quadriga_lib.cart2geo(rand(3,3,2));
assert( isequal(size(az2), [3,2]) );
assert( isequal(size(el2), [3,2]) );
assert( isequal(size(le2), [3,2]) );

% --- Separate x, y, z input path ---

xv = [0;v]; yv = [1;v]; zv = [1;-1];
[az,el,le] = quadriga_lib.cart2geo(xv, yv, zv);
assertElementsAlmostEqual( az, [pi/2;pi/4],   'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4*[1;-1],   'absolute', 1e-5 );
assertElementsAlmostEqual( le, sqrt(2)*[1;1], 'absolute', 1e-5 );
assert( isequal(size(az), [2,1]) );

% Separate [n,m] matrices -> outputs preserve shape
xm = rand(3,4); ym = rand(3,4); zm = rand(3,4);
[azm,elm,lem] = quadriga_lib.cart2geo(xm, ym, zm);
assert( isequal(size(azm), [3,4]) );
assert( isequal(size(elm), [3,4]) );
assert( isequal(size(lem), [3,4]) );

% Separate path matches cube path
c = [xv';yv';zv'];   % [3,1,2] equivalent via cube
[az_c,el_c,le_c] = quadriga_lib.cart2geo([xv,yv,zv]');  % [3,2] cube
assertElementsAlmostEqual( az_c, az, 'absolute', 1e-5 );
assertElementsAlmostEqual( el_c, el, 'absolute', 1e-5 );
assertElementsAlmostEqual( le_c, le, 'absolute', 1e-5 );

% --- use_kernel variants ---

[az1,el1,le1] = quadriga_lib.cart2geo(e, [], [], 0);  % auto
[az2,el2,le2] = quadriga_lib.cart2geo(e, [], [], 1);  % GENERIC
assertElementsAlmostEqual( az1, az2, 'absolute', 1e-5 );
assertElementsAlmostEqual( el1, el2, 'absolute', 1e-5 );
assertElementsAlmostEqual( le1, le2, 'absolute', 1e-5 );

% use_kernel=[] falls back to default (1)
[az3,el3,le3] = quadriga_lib.cart2geo(e, [], [], []);
assertElementsAlmostEqual( az1, az3, 'absolute', 1e-5 );

end