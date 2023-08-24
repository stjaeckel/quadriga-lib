function test_get_channels_planar

% Generate test antenna
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid] = quadriga_lib.arrayant_generate('omni');
e_theta_re(:,:,2) = 2;
e_theta_im(:,:,2) = 0;
e_phi_re(:,:,2) = 0;
e_phi_im(:,:,2) = 0;
element_pos = [0,0; 1,-1; 0,0];
coupling_re = eye(2);
coupling_im = zeros(2);

aod = [0, 90] * pi/180;
eod = [0, 45] * pi/180;
aoa = [180, 180 - atand(10/20)] * pi/180;
eoa = [0, atand(10 / sqrt(10^2 + 20^2) )] * pi/180;

path_gain = [1,0.25];
path_length = [20,  sqrt(10^2 + 10^2) + sqrt(10^2 + 20^2 + 10^2)  ];
M = [1,1 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; 0,0 ; -1,-1 ; 0,0];

[coeff_re, coeff_im, delay, rx_Doppler] = quadriga_lib.get_channels_planar( ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, ...
    aod, eod, aoa, eoa, path_gain, path_length, M, ...
    [0;0;1], [0,0,0], [20;0;1], [0,0,0], 2997924580.0, 1);

amp = coeff_re.^2 + coeff_im.^2;
assertElementsAlmostEqual( amp(:,:,1), [1,4;4,16], 'absolute', 1e-13 );
assertElementsAlmostEqual( amp(:,:,2), [0.25,1;1,4], 'absolute', 1e-13 );

C = 299792458.0;
d0 = 20.0;
d1 = 20.0;
e0 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e1 = (sqrt(9.0 * 9.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));
e2 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(9.0 * 9.0 + 20.0 * 20.0 + 10.0 * 10.0));
e3 = (sqrt(11.0 * 11.0 + 10.0 * 10.0) + sqrt(11.0 * 11.0 + 20.0 * 20.0 + 10.0 * 10.0));

assertElementsAlmostEqual( delay(:,:,1), [d0,d1;d1,d0]/C, 'absolute', 1e-13 );
assertElementsAlmostEqual( delay(:,:,2), [e0,e2;e1,e3]/C, 'absolute', 1.2e-10 );

Doppler = cos(aoa(2)) * cos(eoa(2));
assertElementsAlmostEqual( rx_Doppler, [-1, Doppler], 'absolute', 1e-13 );

end
