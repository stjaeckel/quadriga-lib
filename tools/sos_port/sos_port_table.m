function t = sos_port_table(type, n_dim, recipe, N, seed)
% Frequency table for the ACF of the paper (200 samples, 0 to 49.75 m, decorrelation distance 10 m).
% The table is generated with quadriga_lib.sos_generate and cached in results/tables.
%
% recipe 'lib'   : settings of the library presets: regular directions, 300 test directions,
%                  lower frequency limit 0.005 / m (wavelength at most 20 decorrelation distances)
% recipe 'paper' : settings close to the paper: random directions, 28 test directions, no lower limit
% recipe 'lib02' : as 'lib' with a lower frequency limit of 0.002 / m (wavelength at most 50 decorrelation
%                  distances)
%
% Output: struct with the frequencies and the error measures of quadriga_lib.sos_calc_mse:
%   mse_dir  error along 104 test directions, the "3-D ASE (T = 104)" of Fig. 3, in dB (positive)
%   mse_core error on the coordinate planes inside the disc covered by the ACF, the "2-D ASE"
%   mse_all  error on the planes up to twice the range
%   tail     approximated ACF at 10, 20, 50 and 100 decorrelation distances
%   n_dc     number of sinusoids with abs(f) * d_lambda < 0.01
res_dir = sos_port_init();
dist = (0:199) * 0.25;
acf = sos_port_acf(type, dist, 10);
fn = fullfile(res_dir, 'tables', sprintf('%s_%dD_%s_N%d_seed%d.txt', type, n_dim, recipe, N, seed));
if exist(fn, 'file')
    freq = load(fn);
    sec = NaN; n_pass = NaN;
else
    tic;
    switch recipe
        case 'lib'
            [freq, ase] = quadriga_lib.sos_generate(acf, dist, N, n_dim, true, 300, true, 0.005, 15/49.75, seed);
        case 'paper'
            [freq, ase] = quadriga_lib.sos_generate(acf, dist, N, n_dim, false, 28, true, 0, 15/49.75, seed);
        case 'lib02'
            [freq, ase] = quadriga_lib.sos_generate(acf, dist, N, n_dim, true, 300, true, 0.002, 15/49.75, seed);
        otherwise, error('Unknown recipe.');
    end
    sec = toc; n_pass = numel(ase);
    fid = fopen(fn, 'w');
    fprintf(fid, [repmat('%.17g ', 1, n_dim), '\n'], freq.');
    fclose(fid);
end
[mse_dir, mse_core, mse_all, tail] = quadriga_lib.sos_calc_mse(freq, acf, dist, 104);
t = struct('type', type, 'n_dim', n_dim, 'recipe', recipe, 'N', N, 'seed', seed, 'freq', freq, 'mse_dir', mse_dir, ...
    'mse_core', mse_core, 'mse_all', mse_all, 'tail', tail, 'n_dc', sum(sqrt(sum(freq.^2, 2)) * 10 < 0.01), ...
    'sec', sec, 'n_pass', n_pass);
end
