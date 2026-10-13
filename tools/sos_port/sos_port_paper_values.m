function p = sos_port_paper_values()
% Values of the 2018 paper, read from its text and figures (figure values to about 0.1 dB).
% ASE values are given as in the paper, 10*log10(ASE), i.e. negative numbers.
p.fig3.N       = [150 300 500 1000];          % Combined ACF
p.fig3.ase_3d  = [-29.7 -32.7 -35.0 -38.0];   % "3-D ASE (T = 104)"
p.fig3.ase_2d  = [-30.5 -33.8 -35.8 -38.7];   % "2-D ASE", coordinate planes inside the disc
p.fig6.N       = [100 150 300 500 1000 2000]; % Exponential ACF
p.fig6.ase_3d  = [-27.0 -28.7 -31.7 -34.0 -36.9 -39.8];
p.fig6.ase_2d  = [-29.0 -31.1 -34.4 -36.8 -40.0 -42.7];   % text: -29, -36.8, -42.7 for 100, 500, 2000
p.fig6.wang_N  = [100 500 2000];
p.fig6.wang    = [-23 -30 -36];
p.fig7.N       = [150 300 500 1000];          % Zero crossings per meter, decorrelation distance 20 m
p.fig7.exp     = [0.0432 0.0463 0.0489 0.0550];
p.fig7.gauss   = [0.0217 0.0225 0.0225 0.0225];
p.fig7.comb    = [0.0226 0.0228 0.0232 0.0235];
p.fig8.d       = [2 4];                        % Approximated exponential ACF close to the origin, 20 m
p.fig8.N150    = [0.960 0.858];
p.fig8.N1000   = [0.944 0.823];
end
