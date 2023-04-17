function test_qdant_write

%% Simple test
if exist( 'test.qdant','file' )
    delete('test.qdant');
end

azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
elevation_grid  = [-0.9,0,0.9] * pi/2;
e_theta_re      = reshape(1:12,3,[])/2;
e_theta_im      = -reshape(1:12,3,[])*0.002;
e_phi_re        = -reshape(1:12,3,[]);
e_phi_im        = -reshape(1:12,3,[])*0.001;
element_pos     = [1;2;4];
coupling_re     = 1;
coupling_im     = 0.1;
center_freq     = 2e9;
name            = 'name';

id_file = quadriga_lib.arrayant_qdant_write( e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name,...
    'test.qdant');

assert( id_file == 1 );

[e_theta_reI, e_theta_imI, e_phi_reI, e_phi_imI, azimuth_gridI, elevation_gridI, element_posI, ...
    coupling_reI, coupling_imI, center_frequencyI, nameI, layout] = quadriga_lib.arrayant_qdant_read('test.qdant');

assertElementsAlmostEqual( azimuth_grid, azimuth_grid, 'absolute', 1e-6 );
assertElementsAlmostEqual( elevation_grid, elevation_grid, 'absolute', 1e-6 );
assertElementsAlmostEqual( e_theta_reI, e_theta_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( e_theta_imI, e_theta_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( e_phi_reI, e_phi_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( e_phi_imI, e_phi_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( element_posI, element_pos, 'absolute', 1e-4 );
assertElementsAlmostEqual( coupling_reI, coupling_re, 'absolute', 1e-4 );
assertElementsAlmostEqual( coupling_imI, coupling_im, 'absolute', 1e-4 );
assertElementsAlmostEqual( center_frequencyI, center_freq, 'absolute', 1e-4 );
assertTrue( strcmp(nameI,name) );
assertEqual( layout, uint32(1) );

delete('test.qdant');



%%
% if 0
%     a = qd_arrayant('dipole');
%     a(1,2) = qd_arrayant('ula2');
%     a.xml_write('dipole.qdant');
% end
% 
% 
% if exist('test.qdant','file' )
%     delete('test.qdant');
%     pause(1)
% end
% 
% azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
% elevation_grid  = [-0.9,0,0.9] * pi/2;
% e_theta_re      = reshape(1:12,3,[])/2;
% e_theta_im      = -reshape(0:11,3,[])*0.001;
% e_phi_re        = -reshape(1:12,3,[]);
% e_phi_im        = -reshape(1:12,3,[])*0.001;
% element_pos     = [1 5;2 7;0.000004 7.6]; %[1;2;4];
% coupling_re     = [1;1];
% coupling_im     = [0.1;0.2];
% center_freq     = 2e9;
% name            = 'name';
% 
% id_file = quadriga_lib.arrayant_qdant_write( e_theta_re(:,:,[1 1]), e_theta_im(:,:,[1 1]), e_phi_re(:,:,[1 1]), e_phi_im(:,:,[1 1]), ...
%     azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, name,...
%     'dipole.qdant',112,[1 112 ; 3 2])
% 
% 
% 




