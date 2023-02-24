clear all

%% Pattern

a = qd_arrayant('3gpp-3d',4,16,2.6e9,6,5);
a.combine_pattern;
%%
e_phi_re                = real( a.Fa );
e_phi_im                = imag( a.Fa );
e_theta_re              = real( a.Fb );
e_theta_im              = imag( a.Fb );
azimuth_grid_rad        = a.azimuth_grid;
elevation_grid_rad      = a.elevation_grid;

% Interpolation
n_angles = 1e7;
n_el = a.no_elements;

orientation             = rand(3,1);
azimuth                 = 2*pi*(rand(n_el,n_angles)-0.5);
elevation               = pi*(rand(n_el,n_angles)-0.5);

if 1
    e_phi_re = single( e_phi_re );
    e_phi_im = single( e_phi_im );
    e_theta_re = single( e_theta_re );
    e_theta_im = single( e_theta_im );
    azimuth_grid_rad = single( azimuth_grid_rad );
    elevation_grid_rad = single( elevation_grid_rad );
    orientation = single( orientation );
    azimuth = single( azimuth );
    elevation = single( elevation );
end


% c++ Interpolation
tic
[Vr,Vi,Hr,Hi,Di,az,el] = arrayant_lib.interpolate( e_phi_re, e_phi_im, e_theta_re, e_theta_im, ...
    azimuth_grid_rad, elevation_grid_rad, azimuth, elevation, [], orientation );
toc

%% MATLAB Interpolation
%tic
%[V,H] = a.interpolate(azimuth,elevation,[],orientation);
%toc
