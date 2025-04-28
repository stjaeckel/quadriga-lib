function test_combine_pattern

azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
elevation_grid  = [-0.9,0,0.9] * pi/2;
e_theta_re      = ones(3,4,2);
e_theta_im      = ones(3,4,2)*2;
e_phi_re        = ones(3,4,2)*3;
e_phi_im        = ones(3,4,2)*4;
element_pos     = zeros(3,2);
coupling_re     = [1;1];
coupling_im     = [0;0];
center_freq     = 2e9;

[A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq);

assertTrue( isa(A,'double') );
assertTrue( isa(B,'double') );
assertTrue( isa(C,'double') );
assertTrue( isa(D,'double') );

assertElementsAlmostEqual( A, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( B, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( C, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( D, ones(3,4)*8, 'absolute', 1e-14 );

e_theta_re      = ones(3,4);
e_theta_im      = zeros(3,4);
e_phi_re        = ones(3,4);
e_phi_im        = zeros(3,4);
element_pos     = ones(3,1);
coupling_re     = 1;
coupling_im     = 0;
center_freq     = 2e9;

[A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq);

assertTrue( A(1,1) ~= 0 );
assertTrue( B(1,1) ~= 0 );
assertTrue( C(1,1) ~= 0 );
assertTrue( D(1,1) ~= 0 );

[A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid);

assertElementsAlmostEqual( A, e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( B, e_theta_im, 'absolute', 1e-14 );
assertElementsAlmostEqual( C, e_phi_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( D, e_phi_im, 'absolute', 1e-14 );

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Need at least 6 inputs.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D,~] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Can have at most 4 outputs.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( 1, e_theta_im, e_phi_re, e_phi_im, azimuth_grid,elevation_grid);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sizes of ''e_theta_re'', ''e_theta_im'', ''e_phi_re'', ''e_phi_im'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, 1,elevation_grid);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''azimuth_grid'' does not match number of columns in pattern data.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid,elevation_grid, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Size of ''element_pos'' must be either empty or match [3, n_elements]';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid,elevation_grid, [], [1;1]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''Coupling'' must be a matrix with rows equal to number of elements';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid,elevation_grid, [], [], 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [A,B,C,D] = quadriga_lib.arrayant_combine_pattern( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid,elevation_grid, [], 1, [1;1]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''coupling_im'' must be empty or its size must match ''coupling_re''';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end


