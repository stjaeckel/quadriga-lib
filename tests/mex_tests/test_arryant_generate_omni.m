function test_arryant_generate_omni

[e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid, element_pos, ...
    coupling_re, coupling_im, center_frequency, name] = quadriga_lib.arrayant_generate('omni');

assertTrue( isa(e_theta_re,'double') );
assertTrue( isa(e_theta_im,'double') );
assertTrue( isa(e_phi_re,'double') );
assertTrue( isa(e_phi_im,'double') );
assertTrue( isa(azimuth_grid,'double') );
assertTrue( isa(elevation_grid,'double') );
assertTrue( isa(element_pos,'double') );
assertTrue( isa(coupling_re,'double') );
assertTrue( isa(coupling_im,'double') );
assertTrue( isa(center_frequency,'double') );

assertElementsAlmostEqual( e_theta_re, ones(181,361), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_theta_im, zeros(181,361), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_re, zeros(181,361), 'absolute', 1e-14 );
assertElementsAlmostEqual( e_phi_im, zeros(181,361), 'absolute', 1e-14 );
assertElementsAlmostEqual( azimuth_grid, linspace( -pi, pi, 361 ), 'absolute', 1e-13 );
assertElementsAlmostEqual( elevation_grid, linspace( -pi/2, pi/2, 181 ), 'absolute', 1e-13 );
assertElementsAlmostEqual( element_pos, [0;0;0], 'absolute', 1e-13 );
assertElementsAlmostEqual( coupling_re, 1, 'absolute', 1e-13 );
assertElementsAlmostEqual( coupling_im, 0, 'absolute', 1e-13 );
assertElementsAlmostEqual( center_frequency, 299792458, 'absolute', 1e-13 );
assertTrue( strcmp(name,'omni') );

e_theta_re = quadriga_lib.arrayant_generate('omni', 10);
assertElementsAlmostEqual( e_theta_re, ones(19,37), 'absolute', 1e-14 );

end

