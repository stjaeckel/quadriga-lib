function test_path_loss
% MEX interface test for quadriga_lib.path_loss.
% Scope: argument and output handling, struct inputs, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_path_loss.cpp).

%% Setup: a hand-made logdist model, 3 links with d3D = 100, 1000 and 50 m, 2 frequencies
plpar = struct( 'model', 'logdist', 'A', 20, 'B', 30, 'C', 20 );
scenpar = struct( 'SF_sigma', 3 );
tx_pos = [ 0, 0, 10 ];
rx_pos = [ 100, 0, 10; 0, 1000, 10; 30, 40, 10 ];
fc = [ 1e9, 10e9 ];
ref = 20 * log10( [100; 1000; 50] ) + 30 + 20 * log10( fc / 1e9 );           % [3, 2]

conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
[ sp_los, pl_los ] = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMa_LOS.conf' ) );
[ sp_nlos, pl_nlos ] = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMa_NLOS.conf' ) );

%% Basic call and values
[ pl, sf_sigma ] = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos );
assertEqual( size(pl), [3, 2] );
assertEqual( size(sf_sigma), [3, 2] );
assertTrue( isa(pl, 'double') && isa(sf_sigma, 'double') );
assertElementsAlmostEqual( pl(1, 1), 70, 'absolute', 1e-12 );                % 100 m, 1 GHz
assertElementsAlmostEqual( pl, ref, 'absolute', 1e-12 );
assertEqual( sf_sigma, 3 * ones(3, 2) );                                       % SF_sigma of scenpar

%% Struct inputs
plpar_sf = plpar;
plpar_sf.SF = 5;                                                               % sf from the model
[ pl2, sf2 ] = quadriga_lib.path_loss( plpar_sf, scenpar, fc, tx_pos, rx_pos );
assertEqual( pl2, pl );
assertEqual( sf2, 5 * ones(3, 2) );
[ ~, sf3 ] = quadriga_lib.path_loss( plpar, [], fc, tx_pos, rx_pos );          % scenpar []: default SF_sigma
assertEqual( sf3, 8 * ones(3, 2) );
assertEqual( quadriga_lib.path_loss( [], scenpar, fc, tx_pos, rx_pos ), zeros(3, 2) );      % no model
assertEqual( quadriga_lib.path_loss( struct( 'A', 20 ), scenpar, fc, tx_pos, rx_pos ), zeros(3, 2) );

% A v19 file read with conf_file_read: dual_slope (LOS) and nlos (NLOS) with the random hE
tx_u = [ 0, 0, 25 ];
rx_u = [ 300, 0, 20 ];
[ pa, sfa ] = quadriga_lib.path_loss( pl_los, sp_los, 1e9, tx_u, rx_u );
assertEqual( size(pa), [1, 1] );
assertTrue( pa > 80 && pa < 120 );
assertEqual( sfa, 4 );                                                         % SF_sigma of the file
pn = quadriga_lib.path_loss( pl_nlos, sp_nlos, 1e9, tx_u, rx_u );
assertTrue( pn >= pa );                                                        % NLOS >= LOS term
assertTrue( abs( quadriga_lib.path_loss( pl_nlos, sp_nlos, 1e9, rx_u, tx_u ) - pn ) > 0.1 );  % Tx and Rx differ

%% Optional inputs: d2d_in (omitted or [] = geometric distance, 0 = outdoor link)
assertEqual( quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, [] ), pl );
assertEqual( quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, zeros(3, 1) ), pl );
pl_in = quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, [ 0, 20, 0 ] );         % row vector
assertEqual( pl_in([1, 3], :), pl([1, 3], :) );
assertElementsAlmostEqual( pl_in(2, :), 20 * log10(980) + 30 + 20 * log10( fc / 1e9 ), 'absolute', 1e-12 );

%% Optional inputs: rnd_he (UMa: hE = 1 m for u < p1, hE = 18 m for u near 1 at hUT = 20 m)
assertEqual( quadriga_lib.path_loss( pl_los, sp_los, 1e9, tx_u, rx_u, [], [] ), pa );
assertElementsAlmostEqual( quadriga_lib.path_loss( pl_los, sp_los, 1e9, tx_u, rx_u, [], 0 ), pa, 'absolute', 1e-12 );
pr = quadriga_lib.path_loss( pl_los, sp_los, 1e9, tx_u, rx_u, [], 0.999 );
assertTrue( pr - pa > 1 );                                                     % beyond the breakpoint
assertEqual( quadriga_lib.path_loss( pl_los, sp_los, 1e9, tx_u, rx_u, 0, 0.999 ), pr );

%% Outputs: none and fewer
quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos );
assertEqual( quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos ), pl );

%% Type handling and orientation
plpar_t = struct( 'model', 'logdist', 'A', int32(20), 'B', single(30), 'C', uint32(20) );
[ pl_t, sf_t ] = quadriga_lib.path_loss( plpar_t, struct( 'SF_sigma', int32(3) ), single(fc'), ...
    single(tx_pos), single(rx_pos), single([ 0; 20; 0 ]) );                   % values exact in single
assertTrue( isa(pl_t, 'double') && isa(sf_t, 'double') );
assertElementsAlmostEqual( pl_t, pl_in, 'absolute', 1e-12 );
assertEqual( sf_t, sf_sigma );
assertEqual( quadriga_lib.path_loss( plpar, scenpar, fc, repmat(tx_pos, 3, 1), rx_pos ), pl );  % [n_link, 3]

%% Errors
assert_throws( @() quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, [], [], 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out3( plpar, scenpar, fc, tx_pos, rx_pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.path_loss( struct( 'model', 'foo' ), scenpar, fc, tx_pos, rx_pos ), ...
    'Path-loss model ''foo'' is not supported.' );                                                  % C++
assert_throws( @() quadriga_lib.path_loss( struct( 'model', 'logdist', 'A', 20 ), scenpar, fc, tx_pos, rx_pos ), ...
    'Path-loss model ''logdist'' requires the parameter ''B''.' );                                  % C++
assert_throws( @() quadriga_lib.path_loss( pl_los, sp_los, 1e9, tx_u, rx_u, [], 1.5 ), ...
    'Input ''rnd_he'' must be in the range from 0 to 1.' );                                         % C++
assert_throws( @() quadriga_lib.path_loss( plpar, scenpar, fc, tx_pos, rx_pos, [ 0, 1 ] ), ...
    'Number of elements in ''d2d_in'' does not match number of rows in ''rx_pos''.' );              % C++
assert_throws( @() quadriga_lib.path_loss( plpar, struct( 'Foo', 1 ), fc, tx_pos, rx_pos ), ...
    'Unknown key ''Foo'' in ''scenpar''.' );                                                        % C++
assert_throws( @() quadriga_lib.path_loss( 5, scenpar, fc, tx_pos, rx_pos ), 'Input ''plpar'' must be a 1x1 struct.' );

end

% Request 3 outputs so that the output-count guard fires
function out3(varargin)
[~, ~, ~] = quadriga_lib.path_loss( varargin{:} ); %#ok<ASGLU>
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
