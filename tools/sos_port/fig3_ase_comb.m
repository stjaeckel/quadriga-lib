% Fig. 3 of the paper: achievable ASE against the number of sinusoids for the combined ACF (3-D tables).
res_dir = sos_port_init(); p = sos_port_paper_values();
dist = (0:199) * 0.25; acf = sos_port_acf('Comb', dist, 10);
fid = fopen(fullfile(res_dir, 'fig3_ase_comb.csv'), 'w');
fprintf(fid, 'source,N,ase_3d_T104_dB,ase_2d_planes_dB,ase_planes_to_2Dmax_dB,max_abs_tail,n_dc,ase_2d_worst_seed_dB,paper_ase_3d_dB,paper_ase_2d_dB\n');
res = struct('lib', zeros(4, 2), 'paper', zeros(4, 2));
rec = {'lib', 'paper'};
for ir = 1:2
    for k = 1:4
        [b, a] = sos_port_best('Comb', 3, rec{ir}, p.fig3.N(k));
        res.(rec{ir})(k, :) = [-b.mse_dir, -b.mse_core];
        fprintf(fid, 'generated_%s_recipe,%d,%.2f,%.2f,%.2f,%.4f,%d,%.2f,%.1f,%.1f\n', rec{ir}, b.N, -b.mse_dir, -b.mse_core, -b.mse_all, ...
            max(abs(b.tail)), b.n_dc, -min([a.mse_core]), p.fig3.ase_3d(k), p.fig3.ase_2d(k));
    end
end
Np = [128 256 512 1024]; pre = zeros(4, 2);
for k = 1:4
    freq = quadriga_lib.sos_preset(sprintf('Comb%d', Np(k)));
    [m, c, a, tail] = quadriga_lib.sos_calc_mse(freq, acf, dist, 104);
    pre(k, :) = [-m, -c];
    fprintf(fid, 'preset,%d,%.2f,%.2f,%.2f,%.4f,%d,,,\n', Np(k), -m, -c, -a, max(abs(tail)), sum(sqrt(sum(freq.^2, 2)) * 10 < 0.01));
end
fclose(fid);

f = figure('Visible', 'off', 'Position', [100 100 800 480]);
semilogx(p.fig3.N, p.fig3.ase_3d, 'rs--', p.fig3.N, p.fig3.ase_2d, 'bo-', 'LineWidth', 1.5); hold on;
semilogx(p.fig3.N, res.lib(:, 1), 'r^:', p.fig3.N, res.lib(:, 2), 'b^-', 'LineWidth', 1.5);
semilogx(p.fig3.N, res.paper(:, 2), 'kx-.', Np, pre(:, 2), 'md-', 'LineWidth', 1.5); hold off; grid on;
set(gca, 'XTick', [128 150 256 300 500 512 1000 1024]); xlim([110 1100]);
xlabel('Number of sinusoids'); ylabel('10 log10( ASE )'); title('Combined ACF, 3-D approximation');
legend('Paper: 3-D ASE (T = 104)', 'Paper: 2-D ASE', 'Library recipe: 3-D ASE (T = 104)', 'Library recipe: 2-D ASE', ...
    'Paper-like recipe: 2-D ASE', 'Presets: 2-D ASE', 'Location', 'northeast');
print(f, fullfile(res_dir, 'fig3_ase_comb.png'), '-dpng', '-r100'); close(f);
