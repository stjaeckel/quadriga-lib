% Fig. 1 and Fig. 2 of the paper: an example of the 2-D process, and the desired and the approximated 2-D ACF.
% Paper: combined ACF, 300 sinusoids, 250 m x 250 m with 0.5 m spacing, z = 0. Here: preset Comb256.
res_dir = sos_port_init();
[freq, acf, dist] = quadriga_lib.sos_preset('Comb256', 10);

% Fig. 1
phase = quadriga_lib.sos_init_phases(256, 1, 1);
x = 0:0.5:250;
map = quadriga_lib.sos_map(freq, phase, x, x);
f = figure('Visible', 'off', 'Position', [100 100 700 560]);
imagesc(x, x, map); axis xy; axis image; colorbar; caxis([-3 3]);
xlabel('x-position [m]'); ylabel('y-position [m]'); title('Spatially correlated 2-D process, Comb256, d_\lambda = 10 m');
print(f, fullfile(res_dir, 'fig1_sample_2d_process.png'), '-dpng', '-r100'); close(f);

% Mean of the map over many phase sets, against the mean of the target ACF over all pairs of the area
phase_mc = quadriga_lib.sos_init_phases(256, 200, 2);
xs = 0:5:250;
m = squeeze(mean(mean(quadriga_lib.sos_map(freq, phase_mc, xs, xs), 1), 2));
[gx, gy] = meshgrid(xs, xs); p = [gx(:), gy(:)];
dd = sqrt((p(:,1) - p(:,1).').^2 + (p(:,2) - p(:,2).').^2);
r_pair = interp1(dist, acf, dd(:), 'linear', 0);

% Fig. 2
[acf_2d, ax] = quadriga_lib.sos_acf_2d(acf, dist);
acf_ap = quadriga_lib.sos_acf_approx(freq, dist);
[cx, cy] = meshgrid(ax, ax); disc = sqrt(cx.^2 + cy.^2) <= 49.75;
err = acf_ap(:, :, 1) - acf_2d;
f = figure('Visible', 'off', 'Position', [100 100 700 1000]);
subplot(2, 1, 1); imagesc(ax, ax, acf_2d); axis xy; axis image; colorbar; caxis([-0.1 1]); xlim([-60 60]); ylim([-60 60]);
xlabel('x-distance [m]'); ylabel('y-distance [m]'); title('Desired 2-D ACF');
subplot(2, 1, 2); imagesc(ax, ax, acf_ap(:, :, 1)); axis xy; axis image; colorbar; caxis([-0.1 1]); xlim([-60 60]); ylim([-60 60]);
xlabel('x-distance [m]'); ylabel('y-distance [m]'); title('Approximated 2-D ACF on the x-y plane, 256 sinusoids');
print(f, fullfile(res_dir, 'fig2_acf_2d.png'), '-dpng', '-r100'); close(f);

fid = fopen(fullfile(res_dir, 'fig1_fig2.csv'), 'w');
fprintf(fid, 'quantity,value\n');
fprintf(fid, 'map_mean,%.4f\nmap_std,%.4f\nmap_min,%.3f\nmap_max,%.3f\n', mean(map(:)), std(map(:)), min(map(:)), max(map(:)));
fprintf(fid, 'std_of_map_mean_200_phase_sets,%.4f\nsqrt_mean_pair_acf_target,%.4f\n', std(m), sqrt(mean(r_pair)));
fprintf(fid, 'acf_xy_max_abs_err_in_disc,%.4f\nacf_xy_rms_err_in_disc,%.4f\nacf_xy_ase_dB_in_disc,%.2f\n', max(abs(err(disc))), ...
    sqrt(mean(err(disc).^2)), 10*log10(mean(err(disc).^2)));
fprintf(fid, 'acf_xy_max_abs_outside_disc_to_2Dmax,%.4f\n', max(abs(err(~disc))));
fclose(fid);
