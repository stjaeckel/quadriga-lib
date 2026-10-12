function gen_tables(type, n_dim, recipe, N, seeds)
% Generates and caches the tables of one configuration, and appends their error measures to
% results/tables.csv. Called by run_all and usable as a job for parallel runs.
if ischar(n_dim), n_dim = str2double(n_dim); end
if ischar(N), N = str2double(N); end
if nargin < 5, seeds = 1:4; end
res_dir = sos_port_init();
for seed = seeds
    t = sos_port_table(type, n_dim, recipe, N, seed);
    fid = fopen(fullfile(res_dir, 'tables.csv'), 'a');
    fprintf(fid, '%s,%d,%s,%d,%d,%.3f,%.3f,%.3f,%.4f,%.4f,%.4f,%.4f,%d,%.1f,%d\n', type, n_dim, recipe, N, seed, ...
        t.mse_dir, t.mse_core, t.mse_all, t.tail, t.n_dc, t.sec, t.n_pass);
    fclose(fid);
end
end
