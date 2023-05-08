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

end


