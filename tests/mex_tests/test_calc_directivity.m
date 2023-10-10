function test_calc_directivity

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, ~, ~, ~, ~, ~] ...
    = quadriga_lib.arrayant_generate('dipole');

directivity = quadriga_lib.arrayant_calc_directivity(e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid);

assertElementsAlmostEqual( directivity, 1.760964, 'absolute', 1e-6 );

try
    O = quadriga_lib.arrayant_calc_directivity( e_theta_re, e_theta_im );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end

try
    O = quadriga_lib.arrayant_calc_directivity( e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if (strcmp(ME.identifier, 'moxunit:exceptionNotRaised'))
        error('moxunit:exceptionNotRaised', 'Expected an error!');
    end
end


end

