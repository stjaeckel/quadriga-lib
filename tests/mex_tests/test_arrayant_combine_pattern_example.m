function test_arrayant_combine_pattern_example

% Generate dipole pattern
ant = quadriga_lib.arrayant_generate('dipole');

% Duplicate 4 times
ant.e_theta_re  = repmat(ant.e_theta_re, [1,1,4]);
ant.e_theta_im  = repmat(ant.e_theta_im, [1,1,4]);
ant.e_phi_re    = repmat(ant.e_phi_re, [1,1,4]);
ant.e_phi_im    = repmat(ant.e_phi_im, [1,1,4]);
ant.element_pos = repmat(ant.element_pos, [1,4]);

% Set element positions and coupling matrix
ant.element_pos(2,:) = [ -0.75, -0.25, 0.25, 0.75];  % lambda, along y-axis
ant.coupling_re = [ 1 ; 1 ; 1 ; 1 ]/sqrt(4);
ant.coupling_im = [ 0 ; 0 ; 0 ; 0 ];

% Calculate effective pattern
ant_c = quadriga_lib.arrayant_combine_pattern( ant );

% Plot gain
%plot( ant.azimuth_grid*180/pi, [ 10*log10( ant.e_theta_re(91,:,1).^2 ); 10*log10( ant_c.e_theta_re(91,:).^2 ) ]);
%axis([-180 180 -20 15]); ylabel('Gain (dBi)'); xlabel('Azimth angle (deg)'); legend('Dipole','Array')


end


