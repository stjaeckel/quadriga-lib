

if exist('test.dqant','file' )
    delete('test.qdant');
end

azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
elevation_grid  = [-0.9,0,0.9] * pi/2;
e_theta_re      = reshape(1:12,3,[]);
e_theta_im      = reshape(1:12,3,[])*0.001;
e_phi_re        = -reshape(1:12,3,[]);
e_phi_im        = -reshape(1:12,3,[])*0.001;
element_pos     = [1;2;4];
coupling_re     = 1;
coupling_im     = 0.1;
center_freq     = 2e9;
name            = 'name';

id_file = quadriga_lib.arrayant_qdant_write( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name,...
    'test.qdant',uint32(1))







