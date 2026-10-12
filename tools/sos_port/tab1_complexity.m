% Table I of the paper: cost of the sum-of-sinusoids method, measured with the library.
res_dir = sos_port_init();
fid = fopen(fullfile(res_dir, 'tab1_complexity.csv'), 'w');
fprintf(fid, 'case,N,n_pos,n_gen,ms_per_call,ns_per_output_value,ns_per_value_and_sinusoid\n');
cases = {256, 1e5, 1; 1024, 1e5, 1; 256, 1e4, 20; 256, 1000, 2288};
for k = 1:size(cases, 1)
    N = cases{k, 1}; n_pos = cases{k, 2}; n_gen = cases{k, 3};
    freq = quadriga_lib.sos_preset(sprintf('Comb%d', N));
    phase = quadriga_lib.sos_init_phases(N, n_gen, 1);
    pos = [rand(n_pos, 2) * 750, zeros(n_pos, 1)];                              % D = 2: positions on a plane
    v = quadriga_lib.sos_val(freq, phase, pos); reps = 0; t = tic;
    while toc(t) < 2, v = quadriga_lib.sos_val(freq, phase, pos); reps = reps + 1; end
    sec = toc(t) / reps;
    fprintf(fid, 'sos_val,%d,%d,%d,%.3f,%.2f,%.4f\n', N, n_pos, n_gen, 1e3 * sec, 1e9 * sec / (n_pos * n_gen), 1e9 * sec / (n_pos * n_gen * N));
end
% Initialisation: new phases for the 2288 variables of the paper's example
t = tic; for r = 1:20, phase = quadriga_lib.sos_init_phases(256, 2288, r); end
fprintf(fid, 'sos_init_phases,256,,2288,%.3f,,\n', 1e3 * toc(t) / 20);
fprintf(fid, 'memory_bytes_frequencies_double,256,,,%d,,\n', 256 * 3 * 8);
fprintf(fid, 'memory_bytes_phases_double_2288_generators,256,,2288,%d,,\n', 256 * 2288 * 8);
fprintf(fid, 'max_threads,%d,,,,,\n', maxNumCompThreads);
fclose(fid);
