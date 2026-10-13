% Fig. 7 and Fig. 8 of the paper: zero-crossing rate against the number of sinusoids, and the ACF close to
% the origin, both for a decorrelation distance of 20 m.
%
% For a Gaussian process the rate of zero crossings along a line is sqrt( -R''(0) ) / pi. For the sum of
% sinusoids, -R''(0) along the unit vector u is the mean of ( 2 pi f_n.u )^2, so the rate is
% 2 * sqrt( mean_n (f_n.u)^2 ). It is averaged over many directions u. For the presets the rate is also
% counted on generated values.
res_dir = sos_port_init(); p = sos_port_paper_values();
d_lambda = 20; types = {'Exp', 'Gauss', 'Comb'}; N = p.fig7.N; Np = [128 256 512 1024];

% Directions on the sphere (spiral lattice)
n_u = 2000; i = (0:n_u-1).'; el = asin(2 * (i + 0.5) / n_u - 1); az = i * pi * (3 - sqrt(5));
U = [cos(az) .* cos(el), sin(az) .* cos(el), sin(el)];
zcr = @(freq) mean(2 * sqrt(mean((freq * U.').^2, 1)));

Z = zeros(3, 4); Zp = zeros(3, 4); Zc = zeros(3, 4);
for it = 1:3
    for k = 1:4
        b = sos_port_best(types{it}, 3, 'lib', N(k));
        Z(it, k) = zcr(b.freq * 10 / d_lambda);
        freq = quadriga_lib.sos_preset(sprintf('%s%d', types{it}, Np(k)), d_lambda);
        Zp(it, k) = zcr(freq);
        % Count sign changes along lines of 5 km in x, y and a diagonal direction, 10 phase sets
        phase = quadriga_lib.sos_init_phases(Np(k), 10, 1); s = (0:0.1:5000).'; n_cross = 0;
        dirs = [1 0 0; 0 1 0; 1 1 1 / sqrt(3)] ; dirs(3, :) = [1 1 1] / sqrt(3);
        for id = 1:3
            v = quadriga_lib.sos_val(freq, phase, s * dirs(id, :));
            n_cross = n_cross + sum(sum(abs(diff(sign(v), 1, 1)) > 0));
        end
        Zc(it, k) = n_cross / (3 * 10 * 5000);
    end
end
fid = fopen(fullfile(res_dir, 'fig7_zcr.csv'), 'w');
fprintf(fid, 'acf,source,N,crossings_per_m_from_frequencies,crossings_per_m_counted,paper\n');
pv = [p.fig7.exp; p.fig7.gauss; p.fig7.comb];
for it = 1:3
    for k = 1:4, fprintf(fid, '%s,generated_library_recipe,%d,%.4f,,%.4f\n', types{it}, N(k), Z(it, k), pv(it, k)); end
    for k = 1:4, fprintf(fid, '%s,preset,%d,%.4f,%.4f,\n', types{it}, Np(k), Zp(it, k), Zc(it, k)); end
end
fprintf(fid, 'Gauss,target_sqrt2_over_pi_d,,%.4f,,\n', sqrt(2) / (pi * d_lambda));
fclose(fid);
f = figure('Visible', 'off', 'Position', [100 100 800 450]);
plot(N, p.fig7.exp, 'k^-.', N, p.fig7.gauss, 'rs--', N, p.fig7.comb, 'bo--', 'LineWidth', 1.5); hold on;
plot(N, Z(1, :), 'k^-', N, Z(2, :), 'rs-', N, Z(3, :), 'bo-', Np, Zp(1, :), 'kd:', Np, Zp(2, :), 'rd:', Np, Zp(3, :), 'bd:', 'LineWidth', 1.5);
hold off; grid on; ylim([0 0.07]); xlabel('Number of sinusoids'); ylabel('Crossings per meter');
legend('Paper: Exponential', 'Paper: Gaussian', 'Paper: Combined', 'Generated: Exponential', 'Generated: Gaussian', 'Generated: Combined', ...
    'Preset: Exponential', 'Preset: Gaussian', 'Preset: Combined', 'Location', 'eastoutside');
print(f, fullfile(res_dir, 'fig7_zcr.png'), '-dpng', '-r100'); close(f);

% Fig. 8: ACF of the sinusoids close to the origin, averaged over the directions
d = 0:0.2:4;
racf = @(freq) arrayfun(@(x) mean(mean(cos(2 * pi * (freq * U.') * x))), d);
b150 = sos_port_best('Exp', 3, 'lib', 150); b1000 = sos_port_best('Exp', 3, 'lib', 1000);
A = [racf(b150.freq / 2); racf(b1000.freq / 2); racf(quadriga_lib.sos_preset('Exp128', d_lambda)); racf(quadriga_lib.sos_preset('Exp1024', d_lambda))];
fid = fopen(fullfile(res_dir, 'fig8_acf_origin.csv'), 'w');
fprintf(fid, 'd_m,exp_target,gauss_target,approx_exp_N150,approx_exp_N1000,preset_Exp128,preset_Exp1024\n');
for k = 1:numel(d), fprintf(fid, '%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', d(k), exp(-d(k) / 20), exp(-(d(k) / 20)^2), A(:, k)); end
fclose(fid);
f = figure('Visible', 'off', 'Position', [100 100 800 600]);
plot(d, exp(-d / 20), 'k^-.', d, exp(-(d / 20).^2), 'rs--', d, A(1, :), 'b>-', d, A(2, :), 'b<-', d, A(3, :), 'm:', d, A(4, :), 'm--', 'LineWidth', 1.3); hold on;
plot(p.fig8.d, p.fig8.N150, 'bp', p.fig8.d, p.fig8.N1000, 'bh', 'MarkerSize', 12); hold off; grid on; ylim([0.8 1]);
xlabel('Distance d [m]'); ylabel('Corr. coeff.');
legend('Exponential ACF', 'Gaussian ACF', 'Approx. Exp., N = 150', 'Approx. Exp., N = 1000', 'Preset Exp128', 'Preset Exp1024', 'Paper, N = 150', 'Paper, N = 1000', 'Location', 'southwest');
print(f, fullfile(res_dir, 'fig8_acf_origin.png'), '-dpng', '-r100'); close(f);
