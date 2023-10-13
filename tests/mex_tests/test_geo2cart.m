function test_geo2cart

e = 2*pi*(rand(2,6)-0.5);

f = @() quadriga_lib.geo2cart;
assertExceptionThrown( f, 'quadriga_lib:geo2cart:no_input')

f = @() quadriga_lib.geo2cart(e);
assertExceptionThrown( f, 'quadriga_lib:geo2cart:no_input')

f = @() quadriga_lib.geo2cart(e,e);
assertExceptionThrown( f, 'quadriga_lib:geo2cart:no_output')

% empty
try
    [~] = quadriga_lib.geo2cart([],e);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart(e,[]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart([],[]);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% size_mismatch
try
    [~] = quadriga_lib.geo2cart(e(1,:),e);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart(e,e(1,:));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart(e,e,e(1,:));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

% wrong_type
try
    [~] = quadriga_lib.geo2cart(single(e),e);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart(e,single(e));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart(e,e,single(e));
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    [~] = quadriga_lib.geo2cart(single(e),single(e),e);
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

c = quadriga_lib.geo2cart(single(e),single(e),single(e));
assertEqual( size(c), [3,2,6] )
assert( isa(c,'single') );

c = quadriga_lib.geo2cart(e,e,e);
assertEqual( size(c), [3,2,6] )
assert( isa(c,'double') );

c = quadriga_lib.geo2cart(0,0);
assertElementsAlmostEqual( c, [1;0;0], 'absolute', 1e-5 );

c = quadriga_lib.geo2cart(pi/4,0,sqrt(2));
assertElementsAlmostEqual( c, [1;1;0], 'absolute', 1e-5 );

c = quadriga_lib.geo2cart(0,pi/4,sqrt(2));
assertElementsAlmostEqual( c, [1;0;1], 'absolute', 1e-5 );

c = quadriga_lib.geo2cart([pi,-pi/2]',[pi/4,pi/4]',[1;1]*sqrt(2));
assertElementsAlmostEqual( c, [-1,0;0,-1;1,1], 'absolute', 1e-5 );

d = quadriga_lib.geo2cart([pi,-pi/2],[pi/4,pi/4],[1,1]*sqrt(2));
assertElementsAlmostEqual( permute(c,[1,3,2]), d, 'absolute', 1e-5 );
