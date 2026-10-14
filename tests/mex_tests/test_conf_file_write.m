function test_conf_file_write
% MEX interface test for quadriga_lib.conf_file_write.
% Scope: argument handling, struct conversion, defaults, error paths. The file is read back with
% quadriga_lib.conf_file_read.
% Layout, number format and validation rules are validated in the Catch2 suite (test_conf_file_write.cpp).

%% Setup: distinct values of several classes, a path-loss struct with a column vector
fn = 'test_conf_file_write_mex.conf';
fn2 = 'test_conf_file_write_mex2.conf';
fn_nl = [ 'test_conf_file_write_mex_x', char(10), 'KF_mu = 5 %.conf' ];   % line break in the file name
if exist( fn, 'file' ); delete( fn ); end
if exist( fn2, 'file' ); delete( fn2 ); end
if exist( fn_nl, 'file' ); delete( fn_nl ); end

s = struct( 'DS_mu', -7.25, 'NumClusters', int32(12), 'SubpathMethod', 'mmMAGIC', 'ds_kf', single(-0.5), ...
    'LNS_ksi', 0, 'PolVar_sigma', 3, 'DS_omega', 0.1 + 0.2, 'GR_enabled', true );
pl = struct( 'model', 'nlos', 'A1', 22, 'B', [28; 1.5] );
vr = [ 0.5, 100; 10, 10; NaN, 25; 1.5, 22.5 ];

%% Basic call, all inputs
quadriga_lib.conf_file_write( fn, s, pl, {'Ab', 'Cd'}, 'Test ref', {'T1', '', 'T3'}, vr, false );
[ sp, plb, ~, ~, sn, ig ] = quadriga_lib.conf_file_read( fn );

% scenpar: every given value lands in its field, exactly; the rest has the default
assertEqual( sp.DS_mu, -7.25 );
assertEqual( sp.NumClusters, 12 );
assertEqual( sp.SubpathMethod, 'mmMAGIC' );
assertEqual( sp.ds_kf, -0.5 );
assertEqual( sp.LNS_ksi, 0 );                 % zero with a non-zero default is kept
assertEqual( sp.PolVar_sigma, 3 );
assertEqual( sp.DS_omega, 0.1 + 0.2 );        % bit-exact, needs 17 digits
assertEqual( sp.GR_enabled, 1 );
assertEqual( sp.DS_sigma, 0.5 );
assertEqual( sp.ES_D_mu_min, -Inf );

% plpar: struct order kept, the column vector comes back as a row
assertEqual( fieldnames(plb), {'model'; 'A1'; 'B'} );
assertEqual( plb.model, 'nlos' );
assertEqual( plb.A1, 22 );
assertEqual( plb.B, [28, 1.5] );

% short names, no ignored keys
assertEqual( sn, {'Ab'; 'Cd'} );
assertTrue( isempty(ig) );

% Header, validity range and section references land in the text
t = fileread( fn );
assertTrue( ~isempty( strfind( t, '% Config file for scenario "test_conf_file_write_mex"' ) ) );
assertTrue( ~isempty( strfind( t, '% See: Test ref' ) ) );
assertTrue( ~isempty( strfind( t, '% Valid distance range:         10 m' ) ) );
assertTrue( isempty( strfind( t, 'BS antenna height' ) ) );            % row with NaN
assertTrue( ~isempty( strfind( t, '% See: Test ref; T1' ) ) );         % entry 1: LSP section
assertTrue( ~isempty( strfind( t, '% See: Test ref; T3' ) ) );         % entry 3: correlations
assertTrue( isempty( strfind( t, 'NumSubPaths' ) ) );                  % default, not written

%% Optional inputs: omitted, empty, explicit
quadriga_lib.conf_file_write( fn, s );
t1 = fileread( fn );
quadriga_lib.conf_file_write( fn, s, [], [], [], [], [], [] );
assertEqual( fileread( fn ), t1 );
assertTrue( isempty( strfind( t1, 'PL_model' ) ) );
assertTrue( isempty( strfind( t1, '% See:' ) ) );
assertTrue( isempty( strfind( t1, 'ShortName' ) ) );

quadriga_lib.conf_file_write( fn, s, [], [], [], [], [], true );      % write_defaults
t2 = fileread( fn );
assertEqual( numel( strfind( t2, 'NumSubPaths =' ) ), 1 );
assertEqual( numel( strfind( t2, 'PolVar_sigma =' ) ), 1 );

quadriga_lib.conf_file_write( fn, [] );                                  % empty scenpar: default set
assertEqual( fileread( fn ), sprintf( '%% Config file for scenario "test_conf_file_write_mex"\n' ) );

quadriga_lib.conf_file_write( fn, s, [], 'Xy' );                       % one short name as string
[ ~, ~, ~, ~, sn ] = quadriga_lib.conf_file_read( fn );
assertEqual( sn, {'Xy'} );

%% Round trip of the complete output of conf_file_read, with and without write_defaults
quadriga_lib.conf_file_write( fn, s, pl );
[ sp_full, pl_full ] = quadriga_lib.conf_file_read( fn );
for wd = [ false, true ]
    quadriga_lib.conf_file_write( fn2, sp_full, pl_full, {}, '', {}, [], wd );
    [ sp2, pl2 ] = quadriga_lib.conf_file_read( fn2 );
    assertEqual( sp2, sp_full );
    assertEqual( pl2, pl_full );
end

% The output of scenpar_validate is accepted as well; the 8 section references of QuaDRiGa pass
sv = quadriga_lib.scenpar_validate( struct( 'DS_mu', -6.5 ) );
quadriga_lib.conf_file_write( fn2, sv, pl_full, {}, 'ref', {'a','b','c','d','e','f','g','h'}, vr );
sp3 = quadriga_lib.conf_file_read( fn2 );
assertEqual( sp3, sv );

%% Type handling: integer and logical classes in the structs, numeric write_defaults
quadriga_lib.conf_file_write( fn, struct( 'NumClusters', uint32(7), 'SC_lambda', single(50) ), ...
    struct( 'model', 'logdist', 'A', int64([20, 30]) ), [], [], [], [], 1 );
[ sp4, pl4 ] = quadriga_lib.conf_file_read( fn );
assertEqual( sp4.NumClusters, 7 );
assertEqual( sp4.SC_lambda, 50 );
assertEqual( pl4.A, [20, 30] );
assertEqual( numel( strfind( fileread( fn ), 'NumSubPaths =' ) ), 1 );   % write_defaults = 1

%% Path-loss fields: row and column vectors are accepted and written in their element order
quadriga_lib.conf_file_write( fn, s, struct( 'A', [1, 2, 3], 'B', [4; 5; 6] ) );
[ ~, pl7 ] = quadriga_lib.conf_file_read( fn );
assertEqual( pl7.A, [1, 2, 3] );
assertEqual( pl7.B, [4, 5, 6] );

%% Errors from the wrapper
assert_throws( @() quadriga_lib.conf_file_write( fn ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, pl, {}, '', {}, [], false, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out1( fn, s ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, 5 ), 'Input ''scenpar'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, struct( 'DS_mu', [1, 2] ) ), ...
    'Field ''DS_mu'' must be a real numeric scalar.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, struct( 'SubpathMethod', 1 ) ), ...
    'Field ''SubpathMethod'' must be a string.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, 'nlos' ), 'Input ''plpar'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, struct( 'model', 1 ) ), 'Field ''model'' must be a string.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, struct( 'A', 1 + 2i ) ), ...
    'Field ''A'' must be a real numeric vector.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, struct( 'A', true ) ), ...
    'Field ''A'' must be a real numeric vector.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, struct( 'A', [1, 2; 3, 4] ) ), ...
    'Field ''A'' must be a real numeric vector.' );                    % a matrix is not a vector
assert_throws( @() quadriga_lib.conf_file_write( fn, s, struct( 'A', ones(1, 1, 2) ) ), ...
    'Field ''A'' must be a real numeric vector.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, struct( 'NumClusters', uint8(7) ) ), ...
    'unsupported data type' );                                         % 8- and 16-bit classes
assert_throws( @() quadriga_lib.conf_file_write( fn, s, [], 5 ), ...
    'Input ''short_names'' must be a cell array of strings.' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, [], {}, '', 'T1' ), ...
    'Input ''ref_sec'' must be a cell array of strings.' );

%% Errors from C++; a failed call leaves no file
delete( fn );
assert_throws( @() quadriga_lib.conf_file_write( fn, struct( 'GR_epsilon', 1 ) ), 'Unknown key ''GR_epsilon''' );
assert_throws( @() quadriga_lib.conf_file_write( fn, struct( 'SubpathMethod', 'bla' ) ), 'SubpathMethod' );
assert_throws( @() quadriga_lib.conf_file_write( fn, struct( 'NumClusters', 0 ) ), 'NumClusters' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, struct( 'A', [] ) ), 'has no value' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, [], {}, '', repmat( {'x'}, 1, 11 ) ), 'at most 10' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, [], {}, '', {}, ones(3, 2) ), 'size [4, 2]' );
assert_throws( @() quadriga_lib.conf_file_write( fn, s, [], {'Ab', 'Cd', 'Ab'} ), 'must be distinct' );
assertTrue( ~exist( fn, 'file' ) );
assert_throws( @() quadriga_lib.conf_file_write( fn_nl, s ), 'line break in the file name' );
assertTrue( ~exist( fn_nl, 'file' ) );

if exist( fn2, 'file' ); delete( fn2 ); end

end

% Request one output so that the nlhs > 0 guard fires
function out1(varargin)
x = quadriga_lib.conf_file_write( varargin{:} ); %#ok<NASGU>
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
