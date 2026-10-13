% Cross-check for Fig. 4: is the scatter of the estimated ACF between initialisations a property of the sum of
% sinusoids or of any field with this ACF? An exact Gaussian field with the exponential ACF is drawn from its
% covariance matrix (Cholesky) at 1500 positions in the cube of Fig. 4 and estimated with the same bins, and
% compared with the preset Exp256 at the same positions. 40 realisations each.
res_dir = sos_port_init();
[freq, acf, dist] = quadriga_lib.sos_preset('Exp256', 10);
bins = (1:2:69).'; n_bin = numel(bins); n = 1500; n_rep = 40;
lo = [1e-5; 0.5 * (bins(1:end-1) + bins(2:end))]; hi = [lo(2:end); bins(end) + 1];
pos = reshape(quadriga_lib.sos_init_phases(3 * n, 1, 7) / pi, [], 3) * 0.4 * bins(end);
dd = sqrt(max(sum(pos.^2, 2) + sum(pos.^2, 2).' - 2 * (pos * pos.'), 0));
L = chol(exp(-dd / 10) + 1e-10 * eye(n), 'lower');
up = triu(true(n), 1); I = cell(n_bin, 1); J = I;
for b = 1:n_bin, [I{b}, J{b}] = find(up & dd > lo(b) & dd <= hi(b)); end
rng(1); phase = quadriga_lib.sos_init_phases(256, n_rep, 11);
vs = quadriga_lib.sos_val(freq, phase, pos); ve = L * randn(n, n_rep);
Es = zeros(n_bin, n_rep); Ee = Es;
for r = 1:n_rep
    for b = 1:n_bin
        c = corrcoef(vs(I{b}, r), vs(J{b}, r)); Es(b, r) = c(1, 2);
        c = corrcoef(ve(I{b}, r), ve(J{b}, r)); Ee(b, r) = c(1, 2);
    end
end
fid = fopen(fullfile(res_dir, 'fig4b_exact_field_check.csv'), 'w');
fprintf(fid, 'bin_center_m,target,exact_mean,exact_std,exact_min,exact_max,sos_mean,sos_std,sos_min,sos_max\n');
for b = 1:n_bin
    fprintf(fid, '%g,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n', bins(b), exp(-bins(b) / 10), mean(Ee(b, :)), std(Ee(b, :)), min(Ee(b, :)), ...
        max(Ee(b, :)), mean(Es(b, :)), std(Es(b, :)), min(Es(b, :)), max(Es(b, :)));
end
fclose(fid);
