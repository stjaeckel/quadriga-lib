function test_combine_pattern_example

% Generate dipole pattern
[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos] = ...
    quadriga_lib.arrayant_generate('dipole');

% Duplicate 4 times
e_theta_re  = repmat(e_theta_re, [1,1,4]);
e_theta_im  = repmat(e_theta_im, [1,1,4]);
e_phi_re    = repmat(e_phi_re, [1,1,4]);
e_phi_im    = repmat(e_phi_im, [1,1,4]);
element_pos = repmat(element_pos, [1,4]);

% Set element positions and coupling matrix
element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
coupling_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);

% Calculate effective pattern
[ e_theta_re_c, e_theta_im_c, e_phi_re_c, e_phi_im_c] = quadriga_lib.arrayant_combine_pattern( ...
    e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, coupling_re);

% Plot gain
% plot( azimuth_grid*180/pi, [ 10*log10( e_theta_re(91,:,1).^2 ); 10*log10( e_theta_re_c(91,:).^2 ) ]);
% axis([-180 180 -20 15]); ylabel('Gain (dBi)'); xlabel('Azimth angle (deg)'); legend('Dipole','Array')


end


