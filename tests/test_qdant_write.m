clear all

if 0
    a = qd_arrayant('dipole');
    a(1,2) = qd_arrayant('ula2');
    a.xml_write('dipole.qdant');
end


if exist('test.qdant','file' )
    delete('test.qdant');
    pause(1)
end

azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
elevation_grid  = [-0.9,0,0.9] * pi/2;
e_theta_re      = reshape(1:12,3,[])/2;
e_theta_im      = -reshape(0:11,3,[])*0.001;
e_phi_re        = -reshape(1:12,3,[]);
e_phi_im        = -reshape(1:12,3,[])*0.001;
element_pos     = [1 5;2 7;0.000004 7.6]; %[1;2;4];
coupling_re     = [1;1];
coupling_im     = [0.1;0.2];
center_freq     = 2e9;
name            = 'name';

id_file = quadriga_lib.arrayant_qdant_write( e_theta_re(:,:,[1 1]), e_theta_im(:,:,[1 1]), e_phi_re(:,:,[1 1]), e_phi_im(:,:,[1 1]), ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name,...
    'dipole.qdant',112,[1 112 ; 3 2])







