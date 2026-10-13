% Fig. 6 of the paper: achievable ASE against the number of sinusoids for the exponential ACF, 2-D and 3-D tables.
% The ASE is the error on the coordinate plane(s) inside the disc covered by the ACF (mse_core of sos_calc_mse).
res_dir = sos_port_init(); p = sos_port_paper_values();
N = p.fig6.N; rec = {'lib', 'paper', 'lib02'}; res = zeros(numel(N), 2, 3);
fid = fopen(fullfile(res_dir, 'fig6_ase_exp.csv'), 'w');
fprintf(fid, 'recipe,n_dim,N,ase_planes_dB,ase_dir_T104_dB,ase_planes_to_2Dmax_dB,max_abs_tail,n_dc,ase_planes_worst_seed_dB,paper_dB\n');
for ir = 1:3
    for n_dim = 2:3
        for k = 1:numel(N)
            [b, a] = sos_port_best('Exp', n_dim, rec{ir}, N(k));
            res(k, n_dim - 1, ir) = -b.mse_core;
            if n_dim == 2, pv = p.fig6.ase_2d(k); else, pv = p.fig6.ase_3d(k); end
            fprintf(fid, '%s,%d,%d,%.2f,%.2f,%.2f,%.4f,%d,%.2f,%.1f\n', rec{ir}, n_dim, N(k), -b.mse_core, -b.mse_dir, -b.mse_all, ...
                max(abs(b.tail)), b.n_dc, -min([a.mse_core]), pv);
        end
    end
end
fclose(fid);

f = figure('Visible', 'off', 'Position', [100 100 800 480]);
plot(p.fig6.wang_N, p.fig6.wang, 'k^-.', N, p.fig6.ase_3d, 'rs--', N, p.fig6.ase_2d, 'bo--', 'LineWidth', 1.5); hold on;
plot(N, res(:, 2, 1), 'r^-', N, res(:, 1, 1), 'b^-', N, res(:, 2, 2), 'rx:', N, res(:, 1, 2), 'bx:', 'LineWidth', 1.5);
plot(N, res(:, 2, 3), 'rd-', N, res(:, 1, 3), 'bd-', 'LineWidth', 1.0); hold off; grid on;
xlabel('Number of sinusoids'); ylabel('10 log10( ASE )'); title('Exponential ACF');
legend('Wang et al.', 'Paper: 3-D', 'Paper: 2-D', 'Library recipe: 3-D', 'Library recipe: 2-D', 'Paper-like recipe: 3-D', ...
    'Paper-like recipe: 2-D', 'Library recipe, limit 0.02 / d: 3-D', 'Library recipe, limit 0.02 / d: 2-D', 'Location', 'northeast');
print(f, fullfile(res_dir, 'fig6_ase_exp.png'), '-dpng', '-r100'); close(f);
