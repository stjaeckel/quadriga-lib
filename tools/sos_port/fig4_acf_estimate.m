% Fig. 4 of the paper: ACF estimated from the generated values, for one device (3-D) and two devices (6-D).
% Paper: 300 sinusoids, 6000 positions in a cube of 56 m edge length, groups of 2 m, 10 initialisations.
% Here: preset Comb256, bin centers 1:2:69 m (cube edge 0.8 * 69 = 55.2 m), 3000 positions per device.
res_dir = sos_port_init();
[freq, acf, dist] = quadriga_lib.sos_preset('Comb256', 10);
bins = (1:2:69).'; n_bin = numel(bins); n_pos = 3000; n_rep = 10;
lo = [1e-5; 0.5 * (bins(1:end-1) + bins(2:end))]; hi = [lo(2:end); bins(end) + 1];

E3 = zeros(n_bin, n_rep); E6s = E3; E6t = E3; P3 = E3; P6t = E3; V = zeros(1, n_rep);
for r = 1:n_rep
    phase = quadriga_lib.sos_init_phases(256, 2, 100 + r);

    % One device, and two devices on the same map: library function
    [E3(:, r), ~, E6s(:, r)] = quadriga_lib.sos_acf_estimate(freq, phase(:, 1), bins, n_pos, acf, dist, r);

    % Two devices on two maps, equation (19) of the paper. The positions are those of sos_acf_estimate:
    % its documented stream is the one of sos_init_phases, with (2u - 1) = phase / pi.
    pos = reshape(quadriga_lib.sos_init_phases(6 * n_pos, 1, r) / pi, [], 3) * 0.4 * bins(end);
    pt = pos(1:n_pos, :); pr = pos(n_pos+1:end, :);
    v = quadriga_lib.sos_val(freq, phase(:, 1), pt, pr, phase(:, 2));
    dt = sqrt(max(sum(pt.^2, 2) + sum(pt.^2, 2).' - 2 * (pt * pt.'), 0));
    dr = sqrt(max(sum(pr.^2, 2) + sum(pr.^2, 2).' - 2 * (pr * pr.'), 0));
    up = triu(true(n_pos), 1);
    Rt = interp1(dist, acf, dt, 'linear', 0); Rr = interp1(dist, acf, dr, 'linear', 0);

    % Mean of the target ACF over all pairs of the cube: the variance of the mean of the values in the cube
    d3 = sqrt(max(sum(pos.^2, 2) + sum(pos.^2, 2).' - 2 * (pos * pos.'), 0));
    R3 = interp1(dist, acf, d3, 'linear', 0);
    V(r) = mean(R3(:));
    v6 = 0.5 * (mean(Rt(:)) + mean(Rr(:)));
    up3 = triu(true(2 * n_pos), 1);
    for b = 1:n_bin
        sel = up & dt > lo(b) & dt <= hi(b) & dr <= hi(b);
        [i, j] = find(sel);
        if numel(i) > 5, c = corrcoef(v(i), v(j)); E6t(b, r) = c(1, 2); else, E6t(b, r) = NaN; end
        % Expected Pearson coefficient when the mean of the cube is removed: (R - v) / (1 - v)
        P6t(b, r) = (mean(0.5 * (Rt(sel) + Rr(sel))) - v6) / (1 - v6);
        sel3 = up3 & d3 > lo(b) & d3 <= hi(b);
        P3(b, r) = (mean(R3(sel3)) - V(r)) / (1 - V(r));
    end
end

target = sos_port_acf('Comb', bins, 10);
fid = fopen(fullfile(res_dir, 'fig4_acf_estimate.csv'), 'w');
fprintf(fid, 'bin_center_m,target,est3d_mean,est3d_min,est3d_max,expected3d_finite_cube,est6d_two_maps_mean,est6d_two_maps_min,est6d_two_maps_max,expected6d_two_maps,est6d_same_map_mean,est6d_same_map_min,est6d_same_map_max\n');
for b = 1:n_bin
    fprintf(fid, '%g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', bins(b), target(b), mean(E3(b, :)), min(E3(b, :)), max(E3(b, :)), ...
        mean(P3(b, :)), mean(E6t(b, :), 'omitnan'), min(E6t(b, :)), max(E6t(b, :)), mean(P6t(b, :), 'omitnan'), mean(E6s(b, :), 'omitnan'), min(E6s(b, :)), max(E6s(b, :)));
end
fprintf(fid, 'variance_of_cube_mean,%.4f\n', mean(V));
fclose(fid);

dd = 0:0.25:49.75;
f = figure('Visible', 'off', 'Position', [100 100 800 900]);
subplot(2, 1, 1);
plot(dd, sos_port_acf('Comb', dd, 10), 'k-.', 'LineWidth', 3); hold on;
errorbar(bins, mean(E3, 2), mean(E3, 2) - min(E3, [], 2), max(E3, [], 2) - mean(E3, 2), 'bs-');
plot(bins, mean(P3, 2), 'g-', 'LineWidth', 1.5); hold off; grid on; xlim([0 70]); ylim([-0.3 1.05]);
xlabel('Distance d [m]'); ylabel('Pearson corr. coeff.');
legend('Desired ACF', '3-D approx., mean and range of 10', 'Desired ACF with the cube mean removed');
subplot(2, 1, 2);
plot(dd, sos_port_acf('Comb', dd, 10), 'k-.', 'LineWidth', 3); hold on;
m6 = mean(E6t, 2, 'omitnan');
errorbar(bins, m6, m6 - min(E6t, [], 2), max(E6t, [], 2) - m6, 'rs-');
plot(bins, mean(P6t, 2, 'omitnan'), 'g-', 'LineWidth', 1.5);
plot(bins, mean(E6s, 2, 'omitnan'), 'm.:', 'LineWidth', 1.2); hold off; grid on; xlim([0 70]); ylim([-0.3 1.05]);
xlabel('Distance d [m]'); ylabel('Pearson corr. coeff.');
legend('Desired ACF', '6-D approx., two maps (19), mean and range of 10', 'Expected for two maps, cube mean removed', '6-D, same map (library)');
print(f, fullfile(res_dir, 'fig4_acf_estimate.png'), '-dpng', '-r100'); close(f);
