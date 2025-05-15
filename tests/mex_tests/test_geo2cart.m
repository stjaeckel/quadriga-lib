function test_geo2cart

e = 2*pi*(rand(2,6)-0.5);

try
    quadriga_lib.geo2cart;
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Need at least one input argument.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% empty
try
    [~] = quadriga_lib.geo2cart([],e);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs cannot be empty.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.geo2cart(e,[]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs cannot be empty.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.geo2cart([],[]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs cannot be empty.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% size_mismatch
try
    [~] = quadriga_lib.geo2cart(e(1,:),e);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs must have the same size.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.geo2cart(e,e(1,:));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs must have the same size.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.geo2cart(e,e,e(1,:));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs must have the same size.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

c = quadriga_lib.geo2cart(single(e),single(e),single(e));
assertEqual( size(c), [3,2,6] )
assert( isa(c,'single') );

c = quadriga_lib.geo2cart(single(e),e,e);
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

end