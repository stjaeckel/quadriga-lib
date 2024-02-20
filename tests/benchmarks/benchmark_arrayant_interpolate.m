clear all
close all

center_freq = 3.7e9;

% TX array
[e_theta_re_tx, e_theta_im_tx, e_phi_re_tx, e_phi_im_tx, azimuth_grid_tx, elevation_grid_tx, element_pos_tx, ...
    coupling_re_tx, coupling_im_tx] = quadriga_lib.arrayant_generate('3GPP', 16, 16, center_freq, 2 );

% RX array
[e_theta_re_rx, e_theta_im_rx, e_phi_re_rx, e_phi_im_rx, azimuth_grid_rx, elevation_grid_rx, element_pos_rx, ...
    coupling_re_rx, coupling_im_rx] = quadriga_lib.arrayant_generate('xpol');

% Random FBS
n_path = 1000;

fbs_pos = 2*(rand(3,n_path)-0.5) * 100;
lbs_pos = 2*(rand(3,n_path)-0.5) * 100;
path_gain = ones(1,n_path);
path_length = zeros(1,n_path); % use shortest
M = zeros(8,n_path); M(1,:) = 1; M(7,:) = 1;

tx_pos = [0;0;10];
tx_orientation = [0;0;0];

rx_pos = [0;0;10];
rx_orientation = [0;0;0];
rx_orientation = [0;0;0];
use_absolute_delays = 1;
add_fake_los_path = 1;

rx_pos = [10;0;1.5];

tic

[ coeff_re, coeff_im, delays, aod, eod, aoa, eoa ] = quadriga_lib.get_channels_spherical( ...
    e_theta_re_tx, e_theta_im_tx, e_phi_re_tx, e_phi_im_tx, azimuth_grid_tx, elevation_grid_tx, element_pos_tx, coupling_re_tx, coupling_im_tx, ...
    e_theta_re_rx, e_theta_im_rx, e_phi_re_rx, e_phi_im_rx, azimuth_grid_rx, elevation_grid_rx, element_pos_rx, coupling_re_rx, coupling_im_rx, ...
    fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos, rx_orientation, center_freq, use_absolute_delays, add_fake_los_path );

% for yy = 0.1 : 0.1 : 1
% 
%     rx_pos_xx = rx_pos + [ 0; yy; 0];
%     [ coeff_re, coeff_im, delays, aodN, eodN, aoaN, eoaN ] = quadriga_lib.get_channels_spherical( ...
%         e_theta_re_tx, e_theta_im_tx, e_phi_re_tx, e_phi_im_tx, azimuth_grid_tx, elevation_grid_tx, element_pos_tx, coupling_re_tx, coupling_im_tx, ...
%         e_theta_re_rx, e_theta_im_rx, e_phi_re_rx, e_phi_im_rx, azimuth_grid_rx, elevation_grid_rx, element_pos_rx, coupling_re_rx, coupling_im_rx, ...
%         fbs_pos, lbs_pos, path_gain, path_length, M, tx_pos, tx_orientation, rx_pos_xx, rx_orientation, center_freq, use_absolute_delays, add_fake_los_path );
% 
% end

toc

%df = aod-aodN;












