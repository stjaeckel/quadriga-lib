function test_ssf_init_phases
% MEX interface test for quadriga_lib.ssf_init_phases.
% Scope: argument and output handling, set index, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_ssf_phases.cpp).

%% Setup: UMi LOS file, N = 12 NLOS clusters
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
N = sp.NumClusters - 1;
blk = [ 0, 2, 1, 4, 3, 6, 5, 8, 7 ];                                         % mirror block of block 0 ... 8
mirror = reshape( (0:N-1)' + blk * N + 1, 1, [] );                           % 1-based mirror columns

%% Basic call and values
phase = quadriga_lib.ssf_init_phases( sp, 42 );
assertEqual( size(phase), [256, 9 * N, 2] );
assertTrue( isa(phase, 'double') );
assertTrue( all( phase(:) >= -pi & phase(:) < pi ) );
assertEqual( phase(:, :, 2), phase(:, mirror, 1) );                          % slice 2 mirrors slice 1
assertEqual( quadriga_lib.ssf_init_phases( sp, 42 ), phase );                % same seed, same phases

%% Optional inputs: seed
p1 = quadriga_lib.ssf_init_phases( sp, 43 );
assertTrue( max(abs(p1(:) - phase(:))) > 1 );
pr = quadriga_lib.ssf_init_phases( sp );                                      % seed -1: system random device
assertEqual( size(pr), [256, 9 * N, 2] );
assertTrue( max(abs(pr(:) - phase(:))) > 1 );
assertEqual( size( quadriga_lib.ssf_init_phases( sp, [] ) ), [256, 9 * N, 2] );

% The large-scale phases of the same seed are not repeated
pl = quadriga_lib.lsf_init_phases( sp, 42, [], [], [], false );
assertTrue( max(abs(pl(:, 1, 1) - phase(:, 1, 1))) > 1 );

%% Optional inputs: set_id (0 = shared set, k >= 1 = frequency k)
assertEqual( quadriga_lib.ssf_init_phases( sp, 42, 0 ), phase );
assertEqual( quadriga_lib.ssf_init_phases( sp, 42, [] ), phase );
f1 = quadriga_lib.ssf_init_phases( sp, 42, 1 );
f2 = quadriga_lib.ssf_init_phases( sp, 42, 2 );
assertEqual( size(f1), [256, N, 2] );
assertEqual( size(f2), [256, N, 2] );
assertEqual( f1(:, :, 2), f1(:, :, 1) );                                      % shadowing mirrors itself
assertTrue( max(abs(f2(:) - f1(:))) > 1 );                                    % independent streams
assertTrue( max(abs(f1(:, 1, 1) - phase(:, 1, 1))) > 1 );
assertEqual( quadriga_lib.ssf_init_phases( sp, 42, 1 ), f1 );

%% Optional inputs: preset
pc = quadriga_lib.ssf_init_phases( sp, 42, 0, 'Comb1024' );
assertEqual( size(pc), [1024, 9 * N, 2] );
assertEqual( quadriga_lib.ssf_init_phases( sp, 42, 0, 'Comb256' ), phase );
assertEqual( quadriga_lib.ssf_init_phases( sp, 42, 0, [] ), phase );

%% scenpar: a subset of fields; NumClusters = 1 gives empty sets
pn = quadriga_lib.ssf_init_phases( struct( 'NumClusters', 5, 'NumClusters_min', 1 ), 42 );
assertEqual( size(pn), [256, 36, 2] );
p0 = quadriga_lib.ssf_init_phases( struct( 'NumClusters', 1, 'NumClusters_min', 1 ), 42, 1 );
assertTrue( isempty(p0) );

%% Outputs: none
quadriga_lib.ssf_init_phases( sp, 42 );

%% Type handling
assertEqual( quadriga_lib.ssf_init_phases( sp, int64(42), uint32(0) ), phase );
assertEqual( quadriga_lib.ssf_init_phases( sp, single(42), single(1) ), f1 );

%% Errors
assert_throws( @() quadriga_lib.ssf_init_phases(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.ssf_init_phases( sp, 42, 0, 'Comb256', 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( sp, 42 ), 'Wrong number of output arguments.' );
msg_set = 'Input ''set_id'' must be 0 (shared set) or a frequency index >= 1.';
assert_throws( @() quadriga_lib.ssf_init_phases( sp, 42, -1 ), msg_set );
assert_throws( @() quadriga_lib.ssf_init_phases( sp, 42, 1.5 ), msg_set );
assert_throws( @() quadriga_lib.ssf_init_phases( sp, 42, NaN ), msg_set );
assert_throws( @() quadriga_lib.ssf_init_phases( sp, -2 ), 'seed' );
assert_throws( @() quadriga_lib.ssf_init_phases( sp, 42, 0, 'Unknown' ), 'preset' );                   % C++
assert_throws( @() quadriga_lib.ssf_init_phases( struct( 'NumClusters', 0 ), 42 ), 'NumClusters' );     % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.ssf_init_phases( varargin{:} ); %#ok<ASGLU>
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
