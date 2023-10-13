function test_cart2geo

e = rand(3,6,2);

f = @() quadriga_lib.cart2geo;
assertExceptionThrown( f, 'quadriga_lib:cart2geo:no_input')

f = @() quadriga_lib.cart2geo(e,[]);
assertExceptionThrown( f, 'quadriga_lib:cart2geo:no_input')

f = @() quadriga_lib.cart2geo(e);
assertExceptionThrown( f, 'quadriga_lib:cart2geo:no_output')

% empty
try
    [~,~] = quadriga_lib.cart2geo([]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% size_mismatch
try
    [~,~] = quadriga_lib.cart2geo(rand(2,2,2,2));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~,~] = quadriga_lib.cart2geo(rand(2,2,2));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% wrong_type
try
    [~,~] = quadriga_lib.cart2geo(int32([1;1;1]));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

[az,el] = quadriga_lib.cart2geo(e);
assert( isa(az,'double') );
assert( isa(el,'double') );

[az,el,le] = quadriga_lib.cart2geo(single(e));
assert( isa(az,'single') );
assert( isa(el,'single') );
assert( isa(le,'single') );

v = 1/sqrt(2);
[az,el,le] = quadriga_lib.cart2geo([-v;-v;0]);
assertElementsAlmostEqual( az, -3*pi/4, 'absolute', 1e-5 );
assertElementsAlmostEqual( el, 0, 'absolute', 1e-5 );
assertElementsAlmostEqual( le, 1, 'absolute', 1e-5 );

[az,el,le] = quadriga_lib.cart2geo([0;2*v;2*v]);
assertElementsAlmostEqual( az, pi/2, 'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4, 'absolute', 1e-5 );
assertElementsAlmostEqual( le, 2, 'absolute', 1e-5 );

[az,el,le] = quadriga_lib.cart2geo([0,v;1,v;1,-1]);
assertElementsAlmostEqual( az, [pi/2;pi/4], 'absolute', 1e-5 );
assertElementsAlmostEqual( el, pi/4*[1;-1], 'absolute', 1e-5 );
assertElementsAlmostEqual( le, sqrt(2)*[1;1], 'absolute', 1e-5 );
