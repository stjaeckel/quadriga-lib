function test_arryant_generate_3GPP

% Generate a custom pattern with custom resolution
amp = quadriga_lib.arrayant_generate('custom',90,90,0,10);
assertTrue( size(amp,1) == 19 );
assertTrue( size(amp,2) == 37 );

% Generate a custom pattern
[ amp, z, ~, ~, az, el, ~, ~, ~, freq ] = quadriga_lib.arrayant_generate('custom',90,90,0);
assertTrue( size(amp,1) == 181 );
assertTrue( size(amp,2) == 361 );

% The single-element 3GPP default pattern
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('3GPP');

assertTrue( abs( e_theta_re(1,1) - 0.0794328 ) < 1e-7 );
assertTrue( abs( e_theta_re(91,181) - 2.51188 ) < 1e-5 );

assertElementsAlmostEqual( e_theta_im, z, 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_re, z, 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_im, z, 'absolute', 1e-14 );
assertElementsAlmostEqual( azimuth_grid, az, 'absolute', 1e-14 );
assertElementsAlmostEqual( elevation_grid, el, 'absolute', 1e-14 );
assertElementsAlmostEqual( element_pos, [0;0;0], 'absolute', 1e-14 );
assertTrue( abs( coupling_re - 1 ) < 1e-14 );
assertTrue( abs( coupling_im ) < 1e-14 );
assertTrue( abs( center_frequency - freq ) < 1e-14 );
assertTrue( strcmp(name,'3gpp') );

% The single-element 3GPP default pattern with +/- 90° polarization
[e_theta_re, e_theta_im, e_phi_re, e_phi_im] = quadriga_lib.arrayant_generate('3GPP',1,1,freq,2);

assertElementsAlmostEqual( e_theta_im, z(:,:,[1 1]), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_im, z(:,:,[1 1]), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_re(:,:,1), z, 'absolute', 1e-14 );

assertTrue( abs( e_theta_re(1,1,1) - 0.0794328 ) < 1e-7 );
assertTrue( abs( e_theta_re(91,181,1) - 2.51188 ) < 1e-5 );
assertTrue( abs( e_phi_re(1,1,2) + 0.0794328 ) < 1e-7 );
assertTrue( abs( e_phi_re(91,181,2) - 2.51188 ) < 1e-5 );

% The single-element 3GPP default pattern with +/- 45° polarization
[e_theta_re, ~, e_phi_re, ~] = quadriga_lib.arrayant_generate('3GPP',1,1,freq,3);

assertTrue( abs( e_theta_re(91,181,1) - 1.776172 ) < 1e-5 );
assertTrue( abs( e_theta_re(91,181,2) - 1.776172 ) < 1e-5 );
assertTrue( abs( e_phi_re(91,181,1) - 1.776172 ) < 1e-5 );
assertTrue( abs( e_phi_re(91,181,2) + 1.776172) < 1e-5 );

% Build a single 3GPP array with custom pattern
M = 2; N = 1; Mg = 1; Ng = 1;
spacing = 0.5; dgv = 0; dgh = 0;
pol = 1; tilt = 0;

[e_theta_re, ~, ~, ~, ~, ~, element_pos] = quadriga_lib.arrayant_generate('3GPP', ...
    M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, amp, z, z, z, az, el);

assertElementsAlmostEqual( e_theta_re, amp(:,:,[1 1]), 'absolute', 1e-14 );
assertElementsAlmostEqual( element_pos, [0,0,-0.25 ; 0,0,0.25]', 'absolute', 1e-14 );

pol = 4; % Combine patterns
[e_theta_re, e_theta_im, ~, ~, ~, ~, element_pos] = quadriga_lib.arrayant_generate('3GPP', ...
    M, N, freq, pol, tilt, spacing, Mg, Ng, dgv, dgh, amp, z, z, z, az, el);

amp_combined = sqrt( 2*amp(91,181).^2 );
assertTrue( abs( e_theta_re(91,181) - amp_combined ) < 1e-7 );
assertElementsAlmostEqual( element_pos, [0,0,0]', 'absolute', 1e-14 );
assertElementsAlmostEqual( e_theta_im, z, 'absolute', 1e-14 );

end

