function test_conf_file_read
% MEX interface test for quadriga_lib.conf_file_read.
% Scope: argument and output handling, struct layout, defaults, error paths.
% Grammar and value checks are validated in the Catch2 suite (test_conf_file_read.cpp).

%% Setup: a conf file with distinct values for every kind of key
fn = 'test_conf_file_read_mex.conf';
fn_bad = 'test_conf_file_read_mex_bad.conf';
if exist( fn, 'file' ); delete( fn ); end
if exist( fn_bad, 'file' ); delete( fn_bad ); end

write_file( fn, { '% Comment', 'DS_mu = -7.5   % delay spread', 'NumClusters = 12', ...
    'SubpathMethod = legacy', 'ds_kf = -0.4', 'xpr_esa = 0.25', 'LSP_fc_min = 6', ...
    'PL_sig = 4', 'PL_model = dual_slope', 'PL_A = 21.5', 'PL_B = 1 2.5 -30', ...
    'ShortName = Ab', 'ShortName = Cd', 'Foo = 1', 'GR_epsilon = 1+1e7j' } );

%% Basic call, all outputs
[ scenpar, plpar, lsp_xcorr, lsp_xcorr_chk, short_names, ignored_keys ] = quadriga_lib.conf_file_read( fn );

% scenpar: 171 fields in the canonical order, new keys at the end; the line SubpathMethod = legacy is ignored
names = fieldnames( scenpar );
assertEqual( size(scenpar), [1, 1] );
assertEqual( numel(names), 171 );
assertEqual( names{1}, 'NumClusters' );
assertEqual( names{2}, 'NumSubPaths' );
assertEqual( names{106}, 'absTOA_lambda' );
assertEqual( names{107}, 'LOS_scatter_radius' );
assertEqual( names{110}, 'GR_enabled' );
assertEqual( names{111}, 'asD_ds' );
assertEqual( names{138}, 'xpr_esa' );
assertEqual( names{139}, 'LSP_fc_min' );
assertEqual( names{165}, 'PolVar_sigma' );
assertEqual( names{166}, 'LOS_state_lambda' );
assertEqual( names{167}, 'IO_state_lambda' );
assertEqual( names{168}, 'O2I_d2D_in_max' );
assertEqual( names{169}, 'DS_VS' );
assertEqual( names{170}, 'DS_VS_a' );
assertEqual( names{171}, 'DS_VS_b' );
assertTrue( ~isfield( scenpar, 'GR_epsilon' ) );
assertTrue( ~isfield( scenpar, 'SubpathMethod' ) );

% Values from the file, defaults for the rest; numeric fields are double scalars
assertEqual( scenpar.DS_mu, -7.5 );
assertEqual( scenpar.NumClusters, 12 );
assertEqual( scenpar.ds_kf, -0.4 );
assertEqual( scenpar.xpr_esa, 0.25 );
assertEqual( scenpar.LSP_fc_min, 6 );
assertEqual( scenpar.NumSubPaths, 20 );
assertEqual( scenpar.DS_sigma, 0.5 );
assertEqual( scenpar.ES_D_mu_min, -Inf );
assertEqual( scenpar.absTOA_mu, -100 );
for n = 1:171
    v = scenpar.(names{n});
    assertTrue( isa(v, 'double') && isequal(size(v), [1, 1]) );
end

% plpar: model first, then the keys in file order as row vectors
assertEqual( fieldnames(plpar), {'model'; 'sig'; 'A'; 'B'} );
assertEqual( plpar.model, 'dual_slope' );
assertEqual( plpar.sig, 4 );
assertEqual( plpar.A, 21.5 );
assertEqual( plpar.B, [1, 2.5, -30] );

% lsp_xcorr: DS-KF is element (1,2), ESA-XPR is element (7,8)
ref = eye(8);
ref(1,2) = -0.4; ref(2,1) = -0.4; ref(7,8) = 0.25; ref(8,7) = 0.25;
assertEqual( lsp_xcorr, ref );
assertTrue( islogical(lsp_xcorr_chk) && isequal(size(lsp_xcorr_chk), [1, 1]) );
assertTrue( lsp_xcorr_chk );

assertEqual( short_names, {'Ab'; 'Cd'} );
assertEqual( ignored_keys, {'Foo'; 'GR_epsilon'} );

%% Optional input: omitted, empty, explicit
scenpar2 = quadriga_lib.conf_file_read( fn, [] );
assertEqual( scenpar2, scenpar );
scenpar3 = quadriga_lib.conf_file_read( fn, true );
assertEqual( scenpar3, scenpar );

write_file( fn_bad, { 'r_DS = -1', 'ds_kf = 0.9', 'ds_sf = 0.9', 'sf_kf = -0.9' } );
assert_throws( @() quadriga_lib.conf_file_read( fn_bad ), 'r_DS' );         % value check, thrown by C++
[ scenpar4, ~, ~, chk4 ] = quadriga_lib.conf_file_read( fn_bad, false );    % no value checks
assertEqual( scenpar4.r_DS, -1 );
assertEqual( chk4, false );                                                 % not positive definite, no error

%% Outputs: none, and fewer than the maximum
quadriga_lib.conf_file_read( fn );
scenpar5 = quadriga_lib.conf_file_read( fn );
assertEqual( scenpar5, scenpar );
[ ~, plpar6 ] = quadriga_lib.conf_file_read( fn );
assertEqual( plpar6, plpar );
[ ~, ~, lsp_xcorr7 ] = quadriga_lib.conf_file_read( fn );
assertEqual( lsp_xcorr7, lsp_xcorr );
[ ~, ~, ~, ~, short_names8 ] = quadriga_lib.conf_file_read( fn );
assertEqual( short_names8, short_names );

%% Files without path-loss keys, short names and unknown keys
write_file( fn_bad, { 'DS_mu = -7' } );
[ scenpar9, plpar9, ~, ~, short_names9, ignored_keys9 ] = quadriga_lib.conf_file_read( fn_bad );
assertTrue( isempty(plpar9) && isa(plpar9, 'double') );                     % [] as in QuaDRiGa
assertTrue( iscell(short_names9) && isempty(short_names9) );
assertTrue( iscell(ignored_keys9) && isempty(ignored_keys9) );

write_file( fn_bad, { 'PL_A = 3' } );                                       % PL keys without a model
[ ~, plpar10 ] = quadriga_lib.conf_file_read( fn_bad );
assertEqual( fieldnames(plpar10), {'A'} );

%% Type handling
scenpar11 = quadriga_lib.conf_file_read( fn, 1 );
assertEqual( scenpar11, scenpar );
scenpar12 = quadriga_lib.conf_file_read( fn_bad, uint32(0) );
assertEqual( scenpar12.DS_mu, -6 );

%% Errors
assert_throws( @() quadriga_lib.conf_file_read(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.conf_file_read( fn, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out7( fn ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.conf_file_read( 'test_conf_file_read_mex_missing.conf' ), 'Cannot open file' );
write_file( fn_bad, { 'DS_sigma = 0.3', 'DS_mu = abc' } );                  % grammar error, thrown by C++
assert_throws( @() quadriga_lib.conf_file_read( fn_bad ), 'line 2' );
assert_throws( @() quadriga_lib.conf_file_read( fn_bad, false ), 'DS_mu' );
write_file( fn_bad, { 'DS_sigma = 0.3', 'DS_mu = 1e400' } );                % beyond the range of a double
assert_throws( @() quadriga_lib.conf_file_read( fn_bad ), 'line 2: Value of ''DS_mu'' is out of range.' );
write_file( fn_bad, { 'DS_sigma = 0.3', 'SubpathMethod = mmMAGIC' } );       % only 'legacy' is accepted
msg_sub = 'line 2: Value ''mmMAGIC'' of ''SubpathMethod'' is not supported; only ''legacy'' is accepted.';
assert_throws( @() quadriga_lib.conf_file_read( fn_bad ), msg_sub );
assert_throws( @() quadriga_lib.conf_file_read( fn_bad, false ), msg_sub );

% Errors of the shared helpers and of C++ carry the identifier of the library
assert_identifier( @() quadriga_lib.conf_file_read( 5 ), 'quadriga_lib:CPPerror' );
assert_identifier( @() quadriga_lib.conf_file_read( fn_bad ), 'quadriga_lib:CPPerror' );

%% A UTF-8 byte order mark at the start of the file is skipped
fid = fopen( fn_bad, 'w' );
fwrite( fid, uint8( [ 239, 187, 191, double('DS_mu = -7.25'), 10, double('ShortName = Bm'), 10 ] ) );
fclose( fid );
[ scenpar13, ~, ~, ~, short_names13, ignored_keys13 ] = quadriga_lib.conf_file_read( fn_bad );
assertEqual( scenpar13.DS_mu, -7.25 );
assertEqual( short_names13, {'Bm'} );
assertTrue( isempty(ignored_keys13) );

delete( fn );
delete( fn_bad );

end

% Write the lines of a cell array to a text file
function write_file(fn, lines)
fid = fopen( fn, 'w' );
for n = 1:numel(lines)
    fprintf( fid, '%s\n', lines{n} );
end
fclose( fid );
end

% Request 7 outputs so that the nlhs > 6 guard fires
function out7(varargin)
[~, ~, ~, ~, ~, ~, ~] = quadriga_lib.conf_file_read( varargin{:} ); %#ok<ASGLU>
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
