function test_arrayant_calc_directivity

ant = quadriga_lib.arrayant_generate('dipole');
directivity = quadriga_lib.arrayant_calc_directivity(ant);
assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

directivity = quadriga_lib.arrayant_calc_directivity(ant, [1,1]);
assertElementsAlmostEqual( directivity, [ 1.760964; 1.760964 ], 'absolute', 1e-6 );

[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_generate('dipole');
directivity = quadriga_lib.arrayant_calc_directivity(A,B,C,D,E,F);
assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

directivity = quadriga_lib.arrayant_calc_directivity(A,B,C,D,E,F, [1,1]);
assertElementsAlmostEqual( directivity, [ 1.760964; 1.760964 ], 'absolute', 1e-6 );

% Errors
try
    directivity = quadriga_lib.arrayant_calc_directivity( A, B );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input must be a struct.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    directivity = quadriga_lib.arrayant_calc_directivity( A, B, C );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.arrayant_calc_directivity( ant, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Element index out of bound.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~] = quadriga_lib.arrayant_calc_directivity( ant );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

