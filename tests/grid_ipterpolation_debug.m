clear all
clc

azimuth_grid =(-5:1:5)*pi/180;
elevation_grid =(-1:0.0498:1)*pi/180;
e_theta_re = zeros(numel(elevation_grid),numel(azimuth_grid),2);
e_theta_re(2,2:end-1,1) = 1:numel(azimuth_grid)-2;
e_theta_re(:,:,2) = -e_theta_re(:,:,1);
zr = zeros(numel(elevation_grid),numel(azimuth_grid),2);
element_pos = rand(3,2);
usage = 0;

x = 0;
y = 15;
z = 0;

tic
[e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r] ...
    = quadriga_lib.arrayant_rotate_pattern(e_theta_re, zr, zr, zr, azimuth_grid, elevation_grid, element_pos, x, y, z, usage);
[e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r] ...
    = quadriga_lib.arrayant_rotate_pattern(e_theta_re_r, e_theta_im_r, e_phi_re_r, e_phi_im_r, azimuth_grid_r, elevation_grid_r, element_pos_r, -x,- y, -z, usage);
toc

amp = sqrt(abs(e_theta_re_r.^2 + e_phi_re_r.^2));

figure(1)
imagesc(azimuth_grid_r*180/pi, elevation_grid_r*180/pi, amp(:,:,1))
colorbar
set(gca, 'YDir','normal')


a = qd_arrayant([]);
a.set_grid(azimuth_grid,elevation_grid);
a.no_elements=2;
a.Fa = e_theta_re;
a.element_position = element_pos;
a.rotate_pattern(y,'y',[],usage);
a.rotate_pattern(-y,'y',[],usage);

figure(2);
ampQ = sqrt(abs(a.Fa.^2 + a.Fb.^2));
imagesc(a.azimuth_grid*180/pi, a.elevation_grid*180/pi, ampQ(:,:,1))
colorbar
set(gca, 'YDir','normal')