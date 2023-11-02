function test_calc_directivity

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid] = quadriga_lib.arrayant_generate('dipole');

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid);
assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, [1,1]);
assertElementsAlmostEqual( directivity, [ 1.760964; 1.760964 ], 'absolute', 1e-6 );

try
    quadriga_lib.arrayant_calc_directivity( e_theta_re, e_theta_im );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Need at least 6 inputs.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.arrayant_calc_directivity( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Element index out of bound.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~] = quadriga_lib.arrayant_calc_directivity( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, 1, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Can have at most 7 inputs.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~] = quadriga_lib.arrayant_calc_directivity( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

