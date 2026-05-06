clear all

n_tx       = 256;
n_rx       = 8;
n_path     = 30;
n_carrier  = 2048;
n_snap     = 128;

coeff_re = randn( n_tx, n_rx, n_path, n_snap);
coeff_im = randn( n_tx, n_rx, n_path, n_snap);

fc = 299792458.0;

delay = rand( 1,1,n_path,n_snap )*100 + 1;

pilots = 0 : 1/(n_carrier-1) : 1;

tic
[ hmat_re, hmat_im ] = quadriga_lib.baseband_freq_response( coeff_re, coeff_im, delay, pilots, fc );
toc

