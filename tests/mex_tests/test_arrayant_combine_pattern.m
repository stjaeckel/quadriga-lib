function test_arrayant_combine_pattern

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

ant_in = struct('e_theta_re', e_theta_re, 'e_theta_im', e_theta_im, 'e_phi_re', e_phi_re, 'e_phi_im', e_phi_im, ...
   'azimuth_grid', azimuth_grid, 'elevation_grid', elevation_grid, 'element_pos', element_pos, ...
   'coupling_re', coupling_re, 'coupling_im', coupling_im);

% Separate inputs, combined output
ant = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, 'bla');

assertElementsAlmostEqual( ant.e_theta_re, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_theta_im, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_re, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_im, ones(3,4)*8, 'absolute', 1e-14 );
assertEqual( ant.center_freq, center_freq );
assertEqual( ant.name, 'bla' );

% Combined inputs, combined output
ant = quadriga_lib.arrayant_combine_pattern( ant_in, 3e9 );
assertElementsAlmostEqual( ant.e_theta_re, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_theta_im, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_re, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_im, ones(3,4)*8, 'absolute', 1e-14 );
assertEqual( ant.center_freq, 3e9 );

% Combined inputs, separate output
[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_combine_pattern( ant_in, 3e9 );
assertElementsAlmostEqual( A, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( B, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( C, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( D, ones(3,4)*8, 'absolute', 1e-14 );
assertEqual( J, 3e9 );

e_theta_re      = ones(3,4);
e_theta_im      = zeros(3,4);
e_phi_re        = ones(3,4);
e_phi_im        = zeros(3,4);
element_pos     = ones(3,1);
coupling_re     = 1;
coupling_im     = 0;
center_freq     = 2e9;

[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq);

assertTrue( A(1,1) ~= 0 );
assertTrue( B(1,1) ~= 0 );
assertTrue( C(1,1) ~= 0 );
assertTrue( D(1,1) ~= 0 );

[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_combine_pattern( [], [], [], [],  e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, [0;0;0], 1, 0);

assertElementsAlmostEqual( A, e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( B, e_theta_im, 'absolute', 1e-14 );
assertElementsAlmostEqual( C, e_phi_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( D, e_phi_im, 'absolute', 1e-14 );

% Grid inerpolation
ant = quadriga_lib.arrayant_generate('3gpp', 10, [], [], [], [], 2, [], [], [], 0.0);
ant.e_theta_re(:,:,2) = 2 * ant.e_theta_re(:,:,2);
ant.coupling_re = [1;1];
ant.coupling_im = [0;0];

out = quadriga_lib.arrayant_combine_pattern(ant);
assertElementsAlmostEqual( out.e_theta_re, 3*ant.e_theta_re(:,:,1), 'absolute', 1e-14 );

az_grid = linspace(-pi, pi, 73);
out = quadriga_lib.arrayant_combine_pattern(ant, 3.0e9, az_grid);
assertElementsAlmostEqual( out.e_theta_re(:,1:2:end), 3*ant.e_theta_re(:,:,1), 'absolute', 1e-14 );

el_grid = linspace(-pi/2, pi/2, 37);
out = quadriga_lib.arrayant_combine_pattern(ant, 3.0e9, [], el_grid);
assertElementsAlmostEqual( out.e_theta_re(1:2:end,:), 3*ant.e_theta_re(:,:,1), 'absolute', 1e-14 );

directivity_ant = quadriga_lib.arrayant_calc_directivity(ant,1);
directivity_out = quadriga_lib.arrayant_calc_directivity(out);
assertElementsAlmostEqual( directivity_ant, directivity_out, 'absolute', 0.1 );

ant.element_pos(2,:) = [-0.25, 0.25];
out = quadriga_lib.arrayant_combine_pattern(ant);
directivity_out = quadriga_lib.arrayant_calc_directivity(out);
assertTrue(directivity_out > directivity_ant + 1.2);

% Errors
try
    out = quadriga_lib.arrayant_combine_pattern( ant, [],[],[],[]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~] = quadriga_lib.arrayant_combine_pattern( ant );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re(:,2:end), e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sizes of ''e_theta_re'' and ''e_theta_im'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid(1:2), elevation_grid, element_pos, coupling_re, coupling_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''azimuth_grid'' does not match number of columns in pattern data.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, 1, coupling_re, coupling_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Size of ''element_pos'' must be either empty or match [3, n_elements]';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, [1;1], [0;0]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''Coupling'' must be a matrix with rows equal to number of elements';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, [], 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, 1, [1,1]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''coupling_im'' must be empty or its size must match ''coupling_re''';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
