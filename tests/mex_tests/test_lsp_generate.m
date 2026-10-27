function test_lsp_generate
% MEX interface test for quadriga_lib.lsp_generate.
% Scope: argument and output handling, optional outputs, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_lsp_generate.cpp).

%% Setup: UMa files, 3 links, 2 frequencies
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMa_LOS.conf' ) );
sp_nlos = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMa_NLOS.conf' ) );
sp_inf = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_InF_NLOS_SL.conf' ) );
tx_pos = [ 0, 0, 25 ];
rx_pos = [ 100, 0, 1.5; 0, 200, 1.5; -50, 50, 22.5 ];
fc = [ 2e9, 28e9 ];
rnd = zeros( 3, 11 );
rnd(:, 10) = 0.5;                                                              % medium ground

%% Basic call, all outputs
[ lsp, abs_toa, gr_eps_re, gr_eps_im, eod_offset_deg, lsp_mu, lsp_sigma ] = ...
    quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd );
assertEqual( size(lsp), [3, 2, 8] );
assertEqual( size(abs_toa), [3, 1] );
assertEqual( size(gr_eps_re), [3, 2] );
assertEqual( size(gr_eps_im), [3, 2] );
assertEqual( size(eod_offset_deg), [3, 2] );
assertEqual( size(lsp_mu), [3, 2, 8] );
assertEqual( size(lsp_sigma), [3, 2, 8] );
assertTrue( isa(lsp, 'double') && isa(abs_toa, 'double') && isa(lsp_sigma, 'double') );

% rnd = 0 gives the median LSPs: 10.^mu for DS, 10.^(0.1·mu) for KF
assertElementsAlmostEqual( lsp(:, :, 1), 10.^lsp_mu(:, :, 1), 'relative', 1e-12 );
assertElementsAlmostEqual( lsp(:, :, 2), 10.^(0.1 * lsp_mu(:, :, 2)), 'relative', 1e-12 );
assertEqual( lsp(:, :, 3), ones(3, 2) );                                      % SF: mean 0 dB
assertTrue( all( lsp_sigma(:) >= 0 ) );
assertTrue( abs( lsp_sigma(1, 1, 3) - 4 ) < 1e-12 );                           % SF_sigma of the file

% Column 10 of rnd (0.5) is the medium ground: 30.4·f^-0.47 + j·0.18·f^1.05, f in GHz
f = repmat( fc / 1e9, 3, 1 );
assertElementsAlmostEqual( gr_eps_re, 30.4 * f.^-0.47, 'relative', 1e-12 );
assertElementsAlmostEqual( gr_eps_im, 0.18 * f.^1.05, 'relative', 1e-12 );

% Column 9 of rnd is the absolute TOA (UMa NLOS: absTOA_mu = -7.4, no bound); LOS: disabled
assertEqual( abs_toa, zeros(3, 1) );
rnd_t = rnd;
rnd_t(:, 9) = [ 0; 1; -1 ];
[ ~, toa ] = quadriga_lib.lsp_generate( sp_nlos, fc, tx_pos, rx_pos, rnd_t );
absTOA_sigma = sp_nlos.absTOA_sigma;
assertElementsAlmostEqual( toa, 10.^( -7.4 + absTOA_sigma * [ 0; 1; -1 ] ), 'relative', 1e-12 );

%% Optional inputs: sf_sigma
[ lsp2, ~, ~, ~, ~, ~, sig2 ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd, [] );
assertEqual( lsp2, lsp );
assertEqual( sig2, lsp_sigma );
sf_in = [ 1, 2; 3, 4; 5, 6 ];
rnd_s = rnd;
rnd_s(:, 3) = 1;                                                               % SF value passed through
[ lsp3, ~, ~, ~, ~, ~, sig3 ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_s, sf_in );
assertEqual( sig3(:, :, 3), sf_in );                                           % replaces slice 3
assertEqual( sig3(:, :, [1, 2, 4:8]), lsp_sigma(:, :, [1, 2, 4:8]) );
assertElementsAlmostEqual( lsp3(:, :, 3), 10.^(0.1 * sf_in), 'relative', 1e-12 );

%% Optional inputs: cholesky (TR order: the SF value is not mixed with KF; symmetric root: it is)
rnd_k = rnd;
rnd_k(:, 2) = 1;                                                               % KF value only
lsp_c = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_k );
assertEqual( lsp_c(:, :, 3), ones(3, 2) );
assertEqual( quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_k, [], true ), lsp_c );
assertEqual( quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_k, [], [] ), lsp_c );
lsp_q = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_k, [], false );
assertTrue( all( abs( lsp_q(:, 1, 3) - 1 ) > 1e-3 ) );

%% Optional inputs: clip_spreads
rnd_a = rnd;
rnd_a(:, 4) = 6;                                                               % large ASD
lsp_a = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_a );
assertEqual( max(max(lsp_a(:, :, 4))), 104 );
assertEqual( quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_a, [], [], true ), lsp_a );
assertEqual( quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_a, [], [], [] ), lsp_a );
lsp_u = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_a, [], [], false );
assertTrue( max(max(lsp_u(:, :, 4))) > 104 );

%% scenpar: DS_VS changes the InF delay spread by lg(a·V/S + b); [] = defaults
[ ~, ~, ~, ~, ~, mu_4 ] = quadriga_lib.lsp_generate( sp_inf, fc, tx_pos, rx_pos, rnd );
sp_inf.DS_VS = 10;
[ lsp_10, ~, ~, ~, ~, mu_10 ] = quadriga_lib.lsp_generate( sp_inf, fc, tx_pos, rx_pos, rnd );
assertElementsAlmostEqual( mu_10(:, :, 1) - mu_4(:, :, 1), log10( (30 * 10 + 32) / (30 * 4 + 32) ) * ones(3, 2), ...
    'absolute', 1e-12 );
assertElementsAlmostEqual( lsp_10(:, :, 1), 10.^mu_10(:, :, 1), 'relative', 1e-12 );
assertEqual( size( quadriga_lib.lsp_generate( [], fc, tx_pos, rx_pos, rnd ) ), [3, 2, 8] );

%% Outputs: none and fewer; column 10 of rnd is checked only when the permittivity is requested
quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd );
assertEqual( quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd ), lsp );
[ ~, ~, ~, ~, eod5 ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd );
assertEqual( eod5, eod_offset_deg );
rnd_b = rnd;
rnd_b(:, 10) = 1.5;
assertEqual( quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_b ), lsp );
[ lsp_b, toa_b ] = quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd_b );
assertEqual( lsp_b, lsp );
assertEqual( toa_b, abs_toa );
assert_throws( @() out3( sp, fc, tx_pos, rx_pos, rnd_b ), 'Column 9 of ''rnd'' must be in the range from 0 to 1.' );

%% Type handling and orientation
lsp_t = quadriga_lib.lsp_generate( sp, single(fc'), single(tx_pos), single(rx_pos), single(rnd), [], int32(1), uint32(1) );
assertTrue( isa(lsp_t, 'double') );
assertElementsAlmostEqual( lsp_t, lsp, 'relative', 1e-12 );                   % inputs exact in single
assertEqual( quadriga_lib.lsp_generate( sp, fc, repmat(tx_pos, 3, 1), rx_pos, rnd ), lsp );

%% Errors
assert_throws( @() quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd, [], true, true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out8( sp, fc, tx_pos, rx_pos, rnd ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd(:, 1:10) ), ...
    'Input ''rnd'' must have size [n_link, 11].' );                                                 % C++
assert_throws( @() quadriga_lib.lsp_generate( sp, fc, tx_pos, rx_pos, rnd, -sf_in ), ...
    'Input ''sf_sigma'' must be finite and >= 0.' );                                                % C++
assert_throws( @() quadriga_lib.lsp_generate( struct( 'ds_kf', 0.9, 'ds_sf', 0.9, 'sf_kf', -0.9 ), fc, tx_pos, ...
    rx_pos, rnd ), 'LSP cross-correlation matrix in ''scenpar'' is not positive definite.' );       % C++
assert_throws( @() quadriga_lib.lsp_generate( sp, [], tx_pos, rx_pos, rnd ), 'Input ''fc'' cannot be empty.' );

end

% Request 3 outputs: the ground permittivity is computed and column 10 of rnd is checked
function out3(varargin)
[~, ~, ~] = quadriga_lib.lsp_generate( varargin{:} ); %#ok<ASGLU>
end

% Request 8 outputs so that the output-count guard fires
function out8(varargin)
[~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.lsp_generate( varargin{:} ); %#ok<ASGLU>
end

% Assert that fn() raises an error whose message contains expected (substring).
function assert_throws(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end
