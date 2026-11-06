function test_gbsm_init_phases
% MEX interface test for quadriga_lib.gbsm_init_phases.
% Scope: argument and output handling, set types and set index, scenario struct rule, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_gbsm_phases_lsf.cpp, _cluster.cpp, _ray.cpp).

% ---- Set type 'lsf' ----
%% Setup: distinct decorrelation distances, 4 Rx positions, one Tx
scenpar = struct( 'DS_lambda', 10, 'AS_D_lambda', 18, 'AS_A_lambda', 15 );
rx_pos = [ 0, 0, 1.5; 20, 5, 1.5; 45, -10, 3; 80, 30, 1.5 ];
tx_pos = [ 10, 10, 25 ];
mirror = [ 1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11 ];                        % ASD <-> ASA, ESD <-> ESA

%% Basic call and values
phase = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7 );
assertEqual( size(phase), [256, 11, 2] );
assertTrue( isa(phase, 'double') );
assertEqual( phase(:, :, 1), quadriga_lib.sos_init_phases( 256, 11, 7 ) );    % raw phases without positions
assertEqual( phase(:, :, 2), phase(:, mirror, 1) );                           % slice 2 mirrors slice 1
assertTrue( all( phase(:) >= -pi & phase(:) < pi ) );

%% Optional inputs: seed
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7 ), phase );             % same seed, same phases
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', int64(7) ), phase );
p1 = quadriga_lib.gbsm_init_phases( scenpar, 'lsf' );                                 % seed -1: system random device
p2 = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', [] );
assertEqual( size(p1), [256, 11, 2] );
assertEqual( size(p2), [256, 11, 2] );
assertTrue( max(abs(p1(:) - p2(:))) > 0.1 );
assertEqual( p2(:, :, 2), p2(:, mirror, 1) );

%% Optional inputs: positions and bias removal
pb = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos );                  % bias removal over 4 positions
assertEqual( pb(:, :, 2), pb(:, mirror, 1) );
assertTrue( all( max(abs(pb(:, 1:9, 1) - phase(:, 1:9, 1))) > 1e-6 ) );       % Normal generators corrected
assertEqual( pb(:, 10:11, :), phase(:, 10:11, :) );                           % Uniform generators unchanged

% The corrected fields have a zero mean over the positions (gbsm_draw may use single-precision sin/cos)
rnd = quadriga_lib.gbsm_draw( scenpar, 'lsf', pb, tx_pos, rx_pos );
assertElementsAlmostEqual( mean(rnd(:, 1:9)), zeros(1, 9), 'absolute', 1e-4 );

assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos(1:2, :) ), phase );     % 2 positions: none
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos, [], false ), phase );  % remove_bias = false
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos, [], [] ), pb );        % [] = true
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], tx_pos, rx_pos ), pb );            % tx_pos unused
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], 'Comb256', [], rx_pos, false, true, true ), pb );
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos, [], [], [] ), pb );

% Dual mobility: bias removal over the Rx and Tx positions; one Tx row equals n_link equal rows
pd = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], tx_pos, rx_pos, true );
assertTrue( max(abs(pd(:) - pb(:))) > 1e-6 );
assertEqual( pd(:, :, 2), pd(:, mirror, 1) );
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], repmat(tx_pos, 4, 1), rx_pos, true ), pd );

% horizontal: z is not used in 2D, used in 3D
rx_z = rx_pos;
rx_z(:, 3) = [ 0; 7; 2; 11 ];
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_z ), pb );
ph = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos, [], [], false );
assertTrue( max(abs(ph(:) - pb(:))) > 1e-6 );
assertEqual( quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos, [], [], true ), pb );

% preset: sets n_coeff
pc = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], 'Comb1024', [], [], [], [] );
assertEqual( size(pc), [1024, 11, 2] );
assertEqual( pc(:, :, 1), quadriga_lib.sos_init_phases( 1024, 11, 7 ) );

% scenpar: [] = defaults; the decorrelation distances reach the bias removal
assertEqual( quadriga_lib.gbsm_init_phases( [], 'lsf', 7 ), phase );
pbd = quadriga_lib.gbsm_init_phases( [], 'lsf', 7, [], [], [], rx_pos );
assertTrue( max(abs(pbd(:, 1, 1) - pb(:, 1, 1))) > 1e-6 );                    % DS_lambda differs
assertEqual( pbd(:, 2, 1), pb(:, 2, 1) );                                      % KF_lambda is the default in both
sv = quadriga_lib.scenpar_validate( scenpar );                                 % complete struct
assertEqual( quadriga_lib.gbsm_init_phases( sv, 'lsf', 7, [], [], [], rx_pos ), pb );

%% Outputs: none
quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7 );

%% Type handling
ps = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', int64(7), [], [], [], single(rx_pos), int32(0), uint32(1) );
assertTrue( isa(ps, 'double') );
assertEqual( ps, pb );                                                         % positions are exact in single
pt = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', uint64(7), [], [], single(tx_pos), single(rx_pos), true );
assertEqual( pt, pd );

%% Errors
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], 'Comb256', [], [], false, true, true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( scenpar, 'lsf', 7 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], 'Comb999', [], [], [], [] ), ...
    'Input ''preset'' is not a known preset.' );                                                    % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos, true ), ...
    'Input ''tx_pos'' is required for the bias removal with dual mobility.' );                      % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], rx_pos(:, 1:2) ), ...
    'Input ''rx_pos'' must have 3 columns.' );                                                      % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( struct( 'Foo', 1 ), 'lsf', 7 ), ...
    'Unknown key ''Foo'' in ''scenpar''.' );                                                        % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( struct( 'DS_lambda', -1 ), 'lsf', 7 ), 'DS_lambda' );     % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( 5, 'lsf', 7 ), 'Input ''scenpar'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', -2 ), 'Input ''seed''' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 5 ), ...
    'Input ''set_type'' must be a string (character array).' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], 1 ), ...
    'Input ''preset'' must be a string (character array).' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], {1} ), ...
    'Input ''tx_pos'' has an unsupported data type.' );
assert_throws( @() quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], [], [], {1} ), ...
    'Input ''rx_pos'' has an unsupported data type.' );

% ---- Set type 'cluster' ----
%% Setup: UMi LOS file, N = 12 NLOS clusters
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
N = sp.NumClusters - 1;
blk = [ 0, 2, 1, 4, 3, 6, 5, 8, 7 ];                                         % mirror block of block 0 ... 8
mirror = reshape( (0:N-1)' + blk * N + 1, 1, [] );                           % 1-based mirror columns

%% Basic call and values
phase = quadriga_lib.gbsm_init_phases( sp, 'cluster', 42 );
assertEqual( size(phase), [256, 9 * N, 2] );
assertTrue( isa(phase, 'double') );
assertTrue( all( phase(:) >= -pi & phase(:) < pi ) );
assertEqual( phase(:, :, 2), phase(:, mirror, 1) );                          % slice 2 mirrors slice 1
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', 42 ), phase );                % same seed, same phases

%% Optional inputs: seed
p1 = quadriga_lib.gbsm_init_phases( sp, 'cluster', 43 );
assertTrue( max(abs(p1(:) - phase(:))) > 1 );
pr = quadriga_lib.gbsm_init_phases( sp, 'cluster' );                                      % seed -1: system random device
assertEqual( size(pr), [256, 9 * N, 2] );
assertTrue( max(abs(pr(:) - phase(:))) > 1 );
assertEqual( size( quadriga_lib.gbsm_init_phases( sp, 'cluster', [] ) ), [256, 9 * N, 2] );

% The large-scale phases of the same seed are not repeated
pl = quadriga_lib.gbsm_init_phases( sp, 'lsf', 42, [], [], [], [], [], false );
assertTrue( max(abs(pl(:, 1, 1) - phase(:, 1, 1))) > 1 );

%% Optional inputs: set_id (0 = shared set, k >= 1 = frequency k)
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 0 ), phase );
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, [] ), phase );
f1 = quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 1 );
f2 = quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 2 );
assertEqual( size(f1), [256, N, 2] );
assertEqual( size(f2), [256, N, 2] );
assertEqual( f1(:, :, 2), f1(:, :, 1) );                                      % shadowing mirrors itself
assertTrue( max(abs(f2(:) - f1(:))) > 1 );                                    % independent streams
assertTrue( max(abs(f1(:, 1, 1) - phase(:, 1, 1))) > 1 );
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 1 ), f1 );

%% Optional inputs: preset
pc = quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 0, 'Comb1024' );
assertEqual( size(pc), [1024, 9 * N, 2] );
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 0, 'Comb256' ), phase );
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 0, [] ), phase );

%% scenpar: a subset of fields; NumClusters = 1 gives empty sets
pn = quadriga_lib.gbsm_init_phases( struct( 'NumClusters', 5, 'NumClusters_min', 1 ), 'cluster', 42 );
assertEqual( size(pn), [256, 36, 2] );
p0 = quadriga_lib.gbsm_init_phases( struct( 'NumClusters', 1, 'NumClusters_min', 1 ), 'cluster', 42, 1 );
assertTrue( isempty(p0) );

%% Outputs: none
quadriga_lib.gbsm_init_phases( sp, 'cluster', 42 );

%% Type handling
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', int64(42), uint32(0) ), phase );
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'cluster', single(42), single(1) ), f1 );

%% Errors
msg_set = 'Input ''set_id'' must be 0 (shared set) or a frequency index >= 1.';
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, -1 ), msg_set );
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 1.5 ), msg_set );
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, NaN ), msg_set );
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'cluster', -2 ), 'seed' );
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'cluster', 42, 0, 'Unknown' ), 'preset' );                   % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( struct( 'NumClusters', 0 ), 'cluster', 42 ), 'NumClusters' );     % C++

%% Set types and set index
sp = struct( 'DS_lambda', 10 );
p_lsf = quadriga_lib.gbsm_init_phases( sp, 'lsf', 7 );
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'lsf', 7, 5 ), p_lsf );                % set_id is ignored for 'lsf'
assertEqual( quadriga_lib.gbsm_init_phases( sp, 'lsf', 7, -1 ), p_lsf );
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'LSF', 7 ), ...
    'Input ''set_type'' must be ''lsf'', ''cluster'' or ''ray''.' );                               % C++
assert_throws( @() quadriga_lib.gbsm_init_phases( sp, 'cluster', 7, 0, [], [], [ 0, 0, 1.5 ] ), ...
    'Inputs ''tx_pos'' and ''rx_pos'' are used for set type ''lsf'' only.' );                        % C++

%% Set type 'ray': [n_coeff, 14·N·M + N, 2] for every set index
sr = struct( 'NumClusters', 4, 'NumClusters_min', 1, 'NumSubPaths', 5 );           % N = 3, M = 5
pr = quadriga_lib.gbsm_init_phases( sr, 'ray', 7 );
assertEqual( size(pr), [256, 213, 2] );
assertTrue( isa(pr, 'double') && all( abs(pr(:)) <= pi ) );
assertEqual( quadriga_lib.gbsm_init_phases( sr, 'ray', 7, 0 ), pr );                  % 0 = shared set (default)
pr2 = quadriga_lib.gbsm_init_phases( sr, 'ray', 7, 2, 'Exp512' );
assertEqual( size(pr2), [512, 213, 2] );
assertEqual( size( quadriga_lib.gbsm_init_phases( setfield( sr, 'NumSubPaths', 1 ), 'ray', 7, 1 ) ), [256, 45, 2] );
assertFalse( isequal( quadriga_lib.gbsm_init_phases( sr, 'ray', 7, 1 ), pr ) );      % set 1 differs from set 0
assertTrue( isempty( quadriga_lib.gbsm_init_phases( setfield( sr, 'NumClusters', 1 ), 'ray', 7 ) ) );
assert_throws( @() quadriga_lib.gbsm_init_phases( sr, 'ray', 7, -1 ), msg_set );
assert_throws( @() quadriga_lib.gbsm_init_phases( sr, 'ray', 7, 0, [], [ 0, 0, 1.5 ] ), ...
    'Inputs ''tx_pos'' and ''rx_pos'' are used for set type ''lsf'' only.' );                        % C++

%% Scenario struct: SubpathMethod = 'legacy' is accepted, other values and types are errors
sl = sp;
sl.SubpathMethod = 'legacy';
assertEqual( quadriga_lib.gbsm_init_phases( sl, 'lsf', 7 ), p_lsf );
sl.SubpathMethod = 'mmMAGIC';
assert_throws( @() quadriga_lib.gbsm_init_phases( sl, 'lsf', 7 ), ...
    'Value ''mmMAGIC'' of ''SubpathMethod'' is not supported; only ''legacy'' is accepted.' );
sl.SubpathMethod = 5;
assert_throws( @() quadriga_lib.gbsm_init_phases( sl, 'lsf', 7 ), 'Field ''SubpathMethod'' must be a string.' );
assertFalse( isfield( quadriga_lib.scenpar_validate( sp ), 'SubpathMethod' ) );          % outputs have no such field

end
% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.gbsm_init_phases( varargin{:} ); %#ok<ASGLU>
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
