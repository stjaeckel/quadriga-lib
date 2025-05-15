function test_interp

I = [ 1,0,-2; 1,0,-2 ];
I(:,:,2) = [2,2,2;4,4,4];
x = [0,1.5,2];
y = [10,20];

O = quadriga_lib.interp( x,y,I,x,y );

assertElementsAlmostEqual( I, O, 'absolute', 1e-13 );

O = quadriga_lib.interp( single(x),single(y),single(I),single(x),single(y) );
assert( isa(O,'single') );

assertElementsAlmostEqual( I, O, 'absolute', 1e-6 );

O = quadriga_lib.interp( x,y,I,[0.75, 1.875],[0.0, 100.0] );

T = [0.5, -1.5;0.5, -1.5];
T(:,:,2) = [2.0, 2.0;4.0, 4.0];
assertElementsAlmostEqual( T, O, 'absolute', 1e-13 );

I = [0.0, 1.0, 2.0, 3.0];
x = [3.0, 2.0, 1.0, 0.0];
xo = [2.5, 2.1, 2.0, 1.9];

O = quadriga_lib.interp( x,[],I,xo );
T = [0.5, 0.9, 1.0, 1.1];
assertElementsAlmostEqual( T, O, 'absolute', 1e-13 );

O = quadriga_lib.interp( x,[],I,xo,[] );
assertElementsAlmostEqual( T, O, 'absolute', 1e-13 );

try
    [~,~] = quadriga_lib.interp( x,y,I,x,y );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    O = quadriga_lib.interp( x,y,I );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Need at least 4 input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    O = quadriga_lib.interp( [],[],I,xo );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Data dimensions must match the given number of sample points.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    O = quadriga_lib.interp( x,[],[],xo );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Data dimensions must match the given number of sample points.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

O = quadriga_lib.interp( x,[],I,[] );
assertTrue( length(O) == 0 );

try
    O = quadriga_lib.interp( x,[],1,xo );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Data dimensions must match the given number of sample points.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end


