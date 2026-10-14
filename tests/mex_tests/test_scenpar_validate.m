function test_scenpar_validate
% MEX interface test for quadriga_lib.scenpar_validate.
% Scope: argument and output handling, struct conversion in both directions, defaults, error paths.
% The value checks are validated in the Catch2 suite (test_scenpar_validate.cpp).

%% Basic call: no input returns the default set
[ scenpar, lsp_xcorr, lsp_xcorr_chk ] = quadriga_lib.scenpar_validate();
names = fieldnames( scenpar );
assertEqual( size(scenpar), [1, 1] );
assertEqual( numel(names), 166 );
assertEqual( names{1}, 'NumClusters' );
assertEqual( names{107}, 'SubpathMethod' );
assertEqual( names{108}, 'LOS_scatter_radius' );
assertEqual( names{139}, 'xpr_esa' );
assertEqual( names{140}, 'LSP_fc_min' );
assertEqual( names{166}, 'PolVar_sigma' );
assertEqual( scenpar.NumClusters, 6 );
assertEqual( scenpar.DS_mu, -6 );
assertEqual( scenpar.ES_D_mu_min, -Inf );
assertEqual( scenpar.XPR_sigma, 10 );
assertEqual( scenpar.SubpathMethod, 'legacy' );
assertEqual( scenpar.PolVar_sigma, 0 );
assertEqual( lsp_xcorr, eye(8) );
assertTrue( islogical(lsp_xcorr_chk) && isequal(size(lsp_xcorr_chk), [1, 1]) );
assertTrue( lsp_xcorr_chk );

%% Empty inputs give the default set as well
assertEqual( quadriga_lib.scenpar_validate( [] ), scenpar );
assertEqual( quadriga_lib.scenpar_validate( struct() ), scenpar );
assertEqual( quadriga_lib.scenpar_validate( struct([]) ), scenpar );
assertEqual( quadriga_lib.scenpar_validate( [], [] ), scenpar );
assertEqual( quadriga_lib.scenpar_validate( [], true ), scenpar );

%% A partial struct: given values are kept, the rest is completed, the field order is canonical
in = struct( 'xpr_esa', 0.25, 'SubpathMethod', 'Laplacian', 'DS_mu', -7.5, 'ds_kf', -0.4, 'NumClusters', 12 );
[ out, X, chk ] = quadriga_lib.scenpar_validate( in );
assertEqual( fieldnames(out), names );
assertEqual( out.xpr_esa, 0.25 );
assertEqual( out.SubpathMethod, 'Laplacian' );
assertEqual( out.DS_mu, -7.5 );
assertEqual( out.ds_kf, -0.4 );
assertEqual( out.NumClusters, 12 );
assertEqual( out.NumSubPaths, 20 );
ref = eye(8);
ref(1,2) = -0.4; ref(2,1) = -0.4; ref(7,8) = 0.25; ref(8,7) = 0.25;         % DS-KF and ESA-XPR
assertEqual( X, ref );
assertTrue( chk );

% Round trip: a complete struct is returned unchanged
[ out2, X2 ] = quadriga_lib.scenpar_validate( out );
assertEqual( out2, out );
assertEqual( X2, X );

% A struct with the fields of QuaDRiGa (no new keys) is completed with the 27 new fields
legacy = rmfield( out, names(140:166) );
assertEqual( numel(fieldnames(legacy)), 139 );
assertEqual( quadriga_lib.scenpar_validate( legacy ), out );

%% Optional input 'validate'
bad = struct( 'r_DS', -1, 'SubpathMethod', 'bla', 'ds_kf', 0.9, 'ds_sf', 0.9, 'sf_kf', -0.9 );
assert_throws( @() quadriga_lib.scenpar_validate( bad ), 'r_DS' );          % thrown by C++
assert_throws( @() quadriga_lib.scenpar_validate( bad, [] ), 'r_DS' );
[ out3, ~, chk3 ] = quadriga_lib.scenpar_validate( bad, false );
assertEqual( out3.r_DS, -1 );
assertEqual( out3.SubpathMethod, 'bla' );
assertEqual( chk3, false );                                                 % not positive definite, no error
assert_throws( @() quadriga_lib.scenpar_validate( struct('SubpathMethod', 'bla') ), 'SubpathMethod' );

%% Outputs: none, and fewer than the maximum
quadriga_lib.scenpar_validate( in );
out4 = quadriga_lib.scenpar_validate( in );
assertEqual( out4, out );
[ ~, X5 ] = quadriga_lib.scenpar_validate( in );
assertEqual( X5, X );

%% Type handling: any real numeric class is accepted, the output is double
in6 = struct( 'NumClusters', uint32(12), 'DS_mu', single(-7.5), 'GR_enabled', true, 'NumSubPaths', int64(10) );
out6 = quadriga_lib.scenpar_validate( in6, 1 );
assertTrue( isa(out6.NumClusters, 'double') && isa(out6.DS_mu, 'double') && isa(out6.GR_enabled, 'double') );
assertEqual( out6.NumClusters, 12 );
assertEqual( out6.DS_mu, -7.5 );
assertEqual( out6.GR_enabled, 1 );
assertEqual( out6.NumSubPaths, 10 );

%% Errors
assert_throws( @() quadriga_lib.scenpar_validate( in, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4x( in ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.scenpar_validate( 5 ), 'must be a 1x1 struct' );
assert_throws( @() quadriga_lib.scenpar_validate( 'legacy' ), 'must be a 1x1 struct' );
assert_throws( @() quadriga_lib.scenpar_validate( [in, in] ), 'must be a 1x1 struct' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu', [1, 2]) ), 'Field ''DS_mu'' must be a real numeric scalar.' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu', []) ), 'Field ''DS_mu'' must be a real numeric scalar.' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu', 'abc') ), 'Field ''DS_mu'' must be a real numeric scalar.' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu', 1+2j) ), 'Field ''DS_mu'' must be a real numeric scalar.' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu', {{1}}) ), 'Field ''DS_mu'' must be a real numeric scalar.' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('SubpathMethod', 5) ), 'Field ''SubpathMethod'' must be a string.' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu_typo', 1) ), 'Unknown key ''DS_mu_typo''' );      % C++
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu_typo', 1), false ), 'Unknown key ''DS_mu_typo''' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('GR_epsilon', 10) ), 'Unknown key ''GR_epsilon''' );     % not supported
assert_throws( @() quadriga_lib.scenpar_validate( struct('DS_mu', NaN) ), 'DS_mu' );
assert_throws( @() quadriga_lib.scenpar_validate( struct('NumClusters', 1001) ), 'from 1 to 1000' );

% Errors of the shared helpers carry the identifier of the library
assert_identifier( @() quadriga_lib.scenpar_validate( struct('NumClusters', int8(7)) ), 'quadriga_lib:CPPerror' );

end

% Request 4 outputs so that the nlhs > 3 guard fires
function out4x(varargin)
[~, ~, ~, ~] = quadriga_lib.scenpar_validate( varargin{:} ); %#ok<ASGLU>
end

% Assert that fn() raises an error with the identifier 'id'.
function assert_identifier(fn, id)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, id)
        error('moxunit:exceptionNotRaised', ['EXPECTED ID: "', id, '", GOT: "', ME.identifier, '"']);
    end
end
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
