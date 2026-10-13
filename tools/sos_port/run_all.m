% Re-creates the results of the 2018 paper "Efficient Sum-of-Sinusoids-Based Spatial Consistency for the
% 3GPP New-Radio Channel Model" with the sum-of-sinusoids functions of quadriga-lib.
%
% Requires the MEX files (make matlab=ON, or octave=ON) in +quadriga_lib of the repository root.
% Results (CSV tables and PNG figures) are written to tools/sos_port/results.
% The frequency tables are cached in results/tables; delete that folder to regenerate them (about 40 minutes
% on one core; gen_tables can be started as separate jobs, one line of results/jobs.txt each).
sos_port_init();
p = sos_port_paper_values();
for N = p.fig3.N
    gen_tables('Comb', 3, 'lib', N); gen_tables('Comb', 3, 'paper', N); gen_tables('Gauss', 3, 'lib', N);
end
for N = p.fig6.N
    for n_dim = 2:3, gen_tables('Exp', n_dim, 'lib', N); gen_tables('Exp', n_dim, 'paper', N); gen_tables('Exp', n_dim, 'lib02', N); end
end
fig1_fig2_maps;
fig3_ase_comb;
fig4_acf_estimate;
fig4b_exact_field_check;
fig5_cdf;
fig6_ase_exp;
fig7_fig8_zcr;
tab1_complexity;
