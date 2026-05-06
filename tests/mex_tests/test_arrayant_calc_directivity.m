function test_arrayant_calc_directivity

ant = quadriga_lib.arrayant_generate('dipole');
directivity = quadriga_lib.arrayant_calc_directivity(ant);
assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

directivity = quadriga_lib.arrayant_calc_directivity(ant, [1,1]);
assertElementsAlmostEqual( directivity, [ 1.760964; 1.760964 ], 'absolute', 1e-6 );

[A,B,C,D,E,F,~,~,~,~,~] = quadriga_lib.arrayant_generate('dipole');
directivity = quadriga_lib.arrayant_calc_directivity(A,B,C,D,E,F);
assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

directivity = quadriga_lib.arrayant_calc_directivity(A,B,C,D,E,F, [1,1]);
assertElementsAlmostEqual( directivity, [ 1.760964; 1.760964 ], 'absolute', 1e-6 );

% xpol: 2 cross-polarized isotropic elements -> 0 dBi each
ant_xp = quadriga_lib.arrayant_generate('xpol');
directivity = quadriga_lib.arrayant_calc_directivity(ant_xp);
assertElementsAlmostEqual( directivity, [0; 0], 'absolute', 1e-6 );

% Subset selection
directivity = quadriga_lib.arrayant_calc_directivity(ant_xp, 2);
assertElementsAlmostEqual( directivity, 0, 'absolute', 1e-6 );

% Empty i_element -> all elements (struct mode)
directivity = quadriga_lib.arrayant_calc_directivity(ant_xp, []);
assertElementsAlmostEqual( directivity, [0; 0], 'absolute', 1e-6 );

% Empty i_element -> all elements (split mode)
[A2,B2,C2,D2,E2,F2,~,~,~,~,~] = quadriga_lib.arrayant_generate('xpol');
directivity = quadriga_lib.arrayant_calc_directivity(A2,B2,C2,D2,E2,F2, []);
assertElementsAlmostEqual( directivity, [0; 0], 'absolute', 1e-6 );

% Multi-frequency struct array (n_freq = 2)
ant_mf = [ant, ant];
directivity = quadriga_lib.arrayant_calc_directivity(ant_mf);
assertElementsAlmostEqual( directivity, [1.760964, 1.760964], 'absolute', 1e-6 );

% Multi-frequency with i_element selection
directivity = quadriga_lib.arrayant_calc_directivity(ant_mf, [1,1]);
assertElementsAlmostEqual( directivity, [1.760964, 1.760964; 1.760964, 1.760964], 'absolute', 1e-6 );

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

% Error: i_element = 0 (1-based violation)
try
    [~] = quadriga_lib.arrayant_calc_directivity( ant, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''i_element'' cannot be 0';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: struct + split-mode extras
try
    [~] = quadriga_lib.arrayant_calc_directivity( ant, B, C, D, E, F );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Cannot mix struct input with separate arrayant inputs.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

