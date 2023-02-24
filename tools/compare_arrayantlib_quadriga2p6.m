clear all

if 0
    a = qd_arrayant('3gpp-3d',4,2,1e9,6,5);
    a.combine_pattern;
    a.xml_write('test.qdant');
end
a = qd_arrayant.xml_read('test.qdant');
a.element_position = rand( 3, a.no_elements );

use_single = 0;
i_element = [2,3,4];
n_ang = 10;

% Get data
e_phi_re                = real( a.Fa );
e_phi_im                = imag( a.Fa );
e_theta_re              = real( a.Fb );
e_theta_im              = imag( a.Fb );
azimuth_grid_rad        = a.azimuth_grid;
elevation_grid_rad      = a.elevation_grid;
orientation             = zeros(3,1);
element_position        = a.element_position(:,i_element);

azimuth_all             = 2*pi*(rand(numel(i_element),n_ang)-0.5);
elevation_all           = pi*(rand(numel(i_element),n_ang)-0.5);
azimuth_one             = 2*pi*(rand(1,n_ang)-0.5);
elevation_one           = pi*(rand(1,n_ang)-0.5);

% Reference
[Va,Ha,Da,azM,elM] = a.interpolate(azimuth_all,elevation_all,i_element,orientation);
[Vo,Ho,Do] = a.interpolate(azimuth_one,elevation_one,i_element,orientation);

if use_single
    e_phi_re = single( e_phi_re );
    e_phi_im = single( e_phi_im );
    e_theta_re = single( e_theta_re );
    e_theta_im = single( e_theta_im );
    azimuth_grid_rad = single( azimuth_grid_rad );
    elevation_grid_rad = single( elevation_grid_rad );
    orientation = single( orientation );
    element_position = single( element_position );
    azimuth_all = single( azimuth_all );
    elevation_all = single( elevation_all );
    azimuth_one = single( azimuth_one );
    elevation_one = single( elevation_one );
end


[Var,Vai,Har,Hai,Dai,az,el] = arrayant_lib.interpolate( e_phi_re, e_phi_im, e_theta_re, e_theta_im, ...
    azimuth_grid_rad, elevation_grid_rad, azimuth_all, elevation_all, i_element, orientation, element_position );

err1 = Va-Var-1j*Vai + Ha-Har-1j*Hai

[Vor,Voi,Hor,Hoi,Doi] = arrayant_lib.interpolate( e_phi_re, e_phi_im, e_theta_re, e_theta_im, ...
    azimuth_grid_rad, elevation_grid_rad, azimuth_one, elevation_one, i_element, orientation, element_position );

err2 = Vo-Vor-1j*Voi + Ho-Hor-1j*Hoi
