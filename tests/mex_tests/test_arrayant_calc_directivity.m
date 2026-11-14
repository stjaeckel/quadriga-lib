function test_arrayant_calc_directivity

ant = quadriga_lib.arrayant_generate('dipole');
directivity = quadriga_lib.arrayant_calc_directivity(ant);
assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

directivity = quadriga_lib.arrayant_calc_directivity(ant, [1,1]);
assertElementsAlmostEqual( directivity, [ 1.760964; 1.760964 ], 'absolute', 1e-6 );

% xpol: 2 cross-polarized isotropic elements -> 0 dBi each
ant_xp = quadriga_lib.arrayant_generate('xpol');
directivity = quadriga_lib.arrayant_calc_directivity(ant_xp);
assertElementsAlmostEqual( directivity, [0; 0], 'absolute', 1e-6 );

% Subset selection
directivity = quadriga_lib.arrayant_calc_directivity(ant_xp, 2);
assertElementsAlmostEqual( directivity, 0, 'absolute', 1e-6 );

% Empty i_element -> all elements
directivity = quadriga_lib.arrayant_calc_directivity(ant_xp, []);
assertElementsAlmostEqual( directivity, [0; 0], 'absolute', 1e-6 );

% Multi-frequency struct array (n_freq = 2)
ant_mf = [ant, ant];
directivity = quadriga_lib.arrayant_calc_directivity(ant_mf);
assertElementsAlmostEqual( directivity, [1.760964, 1.760964], 'absolute', 1e-6 );

% Multi-frequency with i_element selection
directivity = quadriga_lib.arrayant_calc_directivity(ant_mf, [1,1]);
assertElementsAlmostEqual( directivity, [1.760964, 1.760964; 1.760964, 1.760964], 'absolute', 1e-6 );

% Errors

% Split mode (the arrayant fields as separate inputs) is not supported
A = ant.e_theta_re; B = ant.e_theta_im; C = ant.e_phi_re;
D = ant.e_phi_im; E = ant.azimuth_grid; F = ant.elevation_grid;
try
    directivity = quadriga_lib.arrayant_calc_directivity( A, B, C, D, E, F );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    directivity = quadriga_lib.arrayant_calc_directivity( A, B );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''arrayant'' must be an arrayant struct or struct array.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.arrayant_calc_directivity( ant, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''i_element'' contains indices that exceed the number of elements.';
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

% Error: the struct field center_freq is a scalar
try
    ant_cf = ant;
    ant_cf.center_freq = [3.5e9, 28e9];
    [~] = quadriga_lib.arrayant_calc_directivity( ant_cf );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''center_freq'' must be a scalar.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

