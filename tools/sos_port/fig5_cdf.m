% Fig. 5 of the paper: CDF of the output values against the Gaussian CDF.
% Paper: 1e4 random positions in 1000 m x 1000 m x 50 m, equations (4) and (19). Here: preset Comb256.
res_dir = sos_port_init();
freq = quadriga_lib.sos_preset('Comb256', 10);
n = 1e4; n_rep = 20;
k = -3:0.1:3; gauss = 0.5 * erfc(-k / sqrt(2));
C3 = zeros(n_rep, numel(k)); C6 = C3; S = zeros(n_rep, 8);
for r = 1:n_rep
    phase = quadriga_lib.sos_init_phases(256, 2, r);
    u = reshape(quadriga_lib.sos_init_phases(6 * n, 1, 1000 + r) / (2*pi) + 0.5, [], 3);    % uniform in [0, 1)
    pos = u .* repmat([1000, 1000, 50], 2 * n, 1);
    v3 = quadriga_lib.sos_val(freq, phase(:, 1), pos(1:n, :));
    v6 = quadriga_lib.sos_val(freq, phase(:, 1), pos(1:n, :), pos(n+1:end, :), phase(:, 2));
    C3(r, :) = mean(v3 <= k, 1); C6(r, :) = mean(v6 <= k, 1);
    z3 = (v3 - mean(v3)) / std(v3); z6 = (v6 - mean(v6)) / std(v6);
    S(r, :) = [mean(v3), std(v3), mean(z3.^3), mean(z3.^4), mean(v6), std(v6), mean(z6.^3), mean(z6.^4)];
end
fid = fopen(fullfile(res_dir, 'fig5_cdf.csv'), 'w');
fprintf(fid, 'quantity,3d,6d_two_maps\n');
fprintf(fid, 'max_cdf_deviation_single_run_worst_of_%d,%.4f,%.4f\n', n_rep, max(max(abs(C3 - gauss))), max(max(abs(C6 - gauss))));
fprintf(fid, 'max_cdf_deviation_mean_over_runs,%.4f,%.4f\n', max(abs(mean(C3, 1) - gauss)), max(abs(mean(C6, 1) - gauss)));
fprintf(fid, 'rms_of_sample_mean,%.4f,%.4f\n', sqrt(mean(S(:, 1).^2)), sqrt(mean(S(:, 5).^2)));
fprintf(fid, 'mean_of_sample_std,%.4f,%.4f\n', mean(S(:, 2)), mean(S(:, 6)));
fprintf(fid, 'mean_skewness,%.4f,%.4f\n', mean(S(:, 3)), mean(S(:, 7)));
fprintf(fid, 'mean_kurtosis,%.4f,%.4f\n', mean(S(:, 4)), mean(S(:, 8)));
fclose(fid);
f = figure('Visible', 'off', 'Position', [100 100 800 450]);
plot(k, gauss, 'k-.', 'LineWidth', 3); hold on;
plot(k(1:2:end), C3(1, 1:2:end), 'ms-', k(2:2:end), C6(1, 2:2:end), 'ro--'); hold off; grid on;
xlabel('k'); ylabel('prob. ( k < abscissa )'); legend('Gaussian CDF', '3-D approximation', '6-D D2D approx. (two maps)', 'Location', 'southeast');
print(f, fullfile(res_dir, 'fig5_cdf.png'), '-dpng', '-r100'); close(f);
