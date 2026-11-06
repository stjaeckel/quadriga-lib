function test_gbsm_draw
% MEX interface test for quadriga_lib.gbsm_draw.
% Scope: argument and output handling, set types and set index, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_gbsm_phases_lsf.cpp, _cluster.cpp, _ray.cpp).

% ---- Set type 'lsf' ----
%% Setup: phases of a scenario with distinct decorrelation distances, 5 links
scenpar = struct( 'DS_lambda', 10, 'AS_D_lambda', 18, 'AS_A_lambda', 15, 'ES_D_lambda', 12, 'ES_A_lambda', 8 );
phase = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7 );
rx_pos = [ 0, 0, 1.5; 20, 5, 1.5; 45, -10, 3; 80, 30, 1.5; 3, 4, 10 ];
tx_pos = [ 10, 10, 25 ];
swap = [ 1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11 ];                          % ASD <-> ASA, ESD <-> ESA

%% Basic call and values
rnd = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos );
assertEqual( size(rnd), [5, 11] );
assertTrue( isa(rnd, 'double') );
assertTrue( all( all( rnd(:, 10:11) > 0 & rnd(:, 10:11) < 1 ) ) );           % Uniform columns
assertTrue( max(max(abs(rnd(:, 1:9)))) < 6 && std(rnd(:)) > 0.1 );

% Column 1 is the DS field (10 m) of slice 1 at the Rx positions projected on the x-y plane; gbsm_draw may
% evaluate sin/cos in single precision (AVX2 kernel)
freq = quadriga_lib.sos_preset( 'Comb256', 10 );
ref = quadriga_lib.sos_val( freq, phase(:, 1, 1), [ rx_pos(:, 1:2), zeros(5, 1) ] );
assertElementsAlmostEqual( rnd(:, 1), ref, 'absolute', 1e-5 );

% Single mobility: the values do not depend on tx_pos; one link alone gives its row
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, [ 0, 0, 0 ], rx_pos ), rnd );
assertElementsAlmostEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos(3, :) ), rnd(3, :), 'absolute', 1e-12 );

%% Optional inputs: floor_id (labels, no index shift)
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [] ), rnd );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], zeros(5, 1) ), rnd );
fl = [ 0; 1; 0; 2; 1 ];
rf = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], fl );
assertEqual( rf([1, 3], :), rnd([1, 3], :) );                                  % floor 0 unchanged
assertTrue( all( max(abs(rf([2, 4, 5], :) - rnd([2, 4, 5], :)), [], 2) > 1e-3 ) );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], fl' ), rf );           % row vector

% Floor 1 adds the offsets 2·pi·n/256 to the phases (n = 0 ... 255)
phase_1 = phase + repmat( 2 * pi * (0:255)' / 256, [1, 11, 2] );
r1 = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase_1, tx_pos, rx_pos([2, 5], :) );
assertElementsAlmostEqual( r1, rf([2, 5], :), 'absolute', 1e-10 );

%% Optional inputs: dual mobility, swap reciprocity
tx_d = [ 5, 5, 1.5; 30, 0, 1.5; 40, -20, 1.5; 60, 10, 1.5; 0, 9, 1.5 ];
rd = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_d, rx_pos, [], [], [], [], true );
assertTrue( max(abs(rd(:) - rnd(:))) > 1e-3 );
rs = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, rx_pos, tx_d, [], [], [], [], true );
assertEqual( rs, rd(:, swap) );                                                % columns 4/5 and 6/7 swapped
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_d, rx_pos, [], [], [], [], false ), rnd );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_d, rx_pos, [], [], [], [], [] ), rnd );

%% Optional inputs: horizontal
rx_z = rx_pos;
rx_z(:, 3) = rx_z(:, 3) + [ 0; 5; 10; 2; 7 ];
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_z ), rnd );     % 2D: z is not used
r3 = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [], [], [], false );
r3z = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_z, [], [], [], [], [], false );
assertTrue( max(abs(r3z(:) - r3(:))) > 1e-3 );                                 % 3D: z is used
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [], [], [], true ), rnd );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [], [], [], [] ), rnd );

%% Optional inputs: preset
phase_c = quadriga_lib.gbsm_init_phases( scenpar, 'lsf', 7, [], 'Comb1024', [], [], [], [] );
rc = quadriga_lib.gbsm_draw( scenpar, 'lsf', phase_c, tx_pos, rx_pos, [], 'Comb1024', [], [], [] );
assertEqual( size(rc), [5, 11] );
assertTrue( max(abs(rc(:) - rnd(:))) > 1e-3 );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], 'Comb256', [], [], [] ), rnd );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [], [], [] ), rnd );

%% scenpar: another DS decorrelation distance changes column 1 only; the full struct is accepted
sp2 = scenpar;
sp2.DS_lambda = 50;
r2 = quadriga_lib.gbsm_draw( sp2, 'lsf', phase, tx_pos, rx_pos );
assertTrue( max(abs(r2(:, 1) - rnd(:, 1))) > 1e-3 );
assertEqual( r2(:, 2:end), rnd(:, 2:end) );
sv = quadriga_lib.scenpar_validate( scenpar );                                 % complete struct
assertEqual( quadriga_lib.gbsm_draw( sv, 'lsf', phase, tx_pos, rx_pos ), rnd );

%% Outputs: none
quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos );

%% Type handling
rt = quadriga_lib.gbsm_draw( scenpar, 'lsf', single(phase), single(tx_pos), single(rx_pos), [], [], single(fl), [], int32(0) );
assertTrue( isa(rt, 'double') );
assertElementsAlmostEqual( rt, rf, 'absolute', 1e-4 );                         % phases rounded to single
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], uint32(fl) ), rf );
assertEqual( quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], 'Comb256', int64(fl), [], false, true ), rf );

%% Errors
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], 'Comb256', [], [], false, true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( scenpar, 'lsf', phase, tx_pos, rx_pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [ 0; -1; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^32.' );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [ 0; 1.5; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^32.' );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [ 0; NaN; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^32.' );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [ 0; 2^32; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^32.' );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase(:, 1:10, :), tx_pos, rx_pos ), ...
    'Input ''phase'' must have size [256, 11, 2] for preset ''Comb256''.' );                        % C++
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], 'Comb1024', [], [], [] ), ...
    'Input ''phase'' must have size [1024, 11, 2] for preset ''Comb1024''.' );                      % C++
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], [ 0; 1 ] ), 'floor_id' );    % C++
msg3d = 'Input ''floor_id'' cannot be used with ''horizontal = false'': in 3D mode the height separates the floors.';
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], [], fl, [], [], false ), msg3d );   % C++
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos, rx_pos, [], 'Comb256', zeros(5, 1), [], true, false ), msg3d );
assert_throws( @() quadriga_lib.gbsm_draw( scenpar, 'lsf', phase, tx_pos(1:2), rx_pos ), ...
    'Input ''tx_pos'' must have 3 columns.' );                                                      % C++

% ---- Set type 'cluster' ----
%% Setup: UMi LOS file (N = 12, SC_lambda = 12 m), shared and per-frequency phases, 5 links
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
N = sp.NumClusters - 1;
phase = quadriga_lib.gbsm_init_phases( sp, 'cluster', 7 );
phase_f = quadriga_lib.gbsm_init_phases( sp, 'cluster', 7, 1 );
rx_pos = [ 0, 0, 1.5; 20, 5, 1.5; 45, -10, 3; 80, 30, 1.5; 3, 4, 10 ];
tx_pos = [ 10, 10, 25 ];
blk = [ 0, 2, 1, 4, 3, 6, 5, 8, 7 ];
swap = reshape( (0:N-1)' + blk * N + 1, 1, [] );                             % departure <-> arrival blocks

%% Basic call and values
rnd = quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos );
assertEqual( size(rnd), [5, 9 * N] );
assertTrue( isa(rnd, 'double') );
assertTrue( all( all( rnd(:, 1:N) > 0 & rnd(:, 1:N) < 1 ) ) );               % Uniform delay block
assertTrue( max(max(abs(rnd(:, N+1:end)))) < 6 && std(rnd(:)) > 0.1 );

% Column N + 1 (sign of AOD, cluster 0) is the field of slice 1 at the Rx positions projected on the
% x-y plane, at SC_lambda; column 1 is its uniform map. gbsm_draw may use single-precision sin/cos (AVX2)
freq = quadriga_lib.sos_preset( 'Comb256', sp.SC_lambda );
pos_xy = [ rx_pos(:, 1:2), zeros(5, 1) ];
ref = quadriga_lib.sos_val( freq, phase(:, N + 1, 1), pos_xy );
assertElementsAlmostEqual( rnd(:, N + 1), ref, 'absolute', 1e-5 );
ref = quadriga_lib.sos_val( freq, phase(:, 1, 1), pos_xy );
assertElementsAlmostEqual( rnd(:, 1), 0.5 * erfc( -ref / sqrt(2) ), 'absolute', 1e-5 );

% Single mobility: the values do not depend on tx_pos; one link alone gives its row
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, [ 0, 0, 0 ], rx_pos ), rnd );
assertElementsAlmostEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos(3, :) ), rnd(3, :), 'absolute', 1e-12 );

%% Optional inputs: set_id (0 = shared set, k >= 1 = per-frequency set)
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0 ), rnd );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, [] ), rnd );
rf = quadriga_lib.gbsm_draw( sp, 'cluster', phase_f, tx_pos, rx_pos, 1 );
assertEqual( size(rf), [5, N] );
assertTrue( any( rf(:) < 0 ) );                                               % Normal shadowing values
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase_f, tx_pos, rx_pos, 2 ), rf );   % same set type
ref = quadriga_lib.sos_val( freq, phase_f(:, 1, 1), pos_xy );
assertElementsAlmostEqual( rf(:, 1), ref, 'absolute', 1e-5 );

%% Optional inputs: floor_id (labels, no index shift)
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [] ), rnd );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], zeros(5, 1) ), rnd );
fl = [ 0; 1; 0; 2; 1 ];
rl = quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], fl );
assertEqual( rl([1, 3], :), rnd([1, 3], :) );                                % floor 0 unchanged
assertTrue( all( max(abs(rl([2, 4, 5], :) - rnd([2, 4, 5], :)), [], 2) > 1e-3 ) );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], fl' ), rl );           % row vector

%% Optional inputs: dual mobility, swap reciprocity
tx_d = [ 5, 5, 1.5; 30, 0, 1.5; 40, -20, 1.5; 60, 10, 1.5; 0, 9, 1.5 ];
rd = quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_d, rx_pos, 0, [], [], [], true );
assertTrue( max(abs(rd(:) - rnd(:))) > 1e-3 );
rs = quadriga_lib.gbsm_draw( sp, 'cluster', phase, rx_pos, tx_d, 0, [], [], [], true );
assertEqual( rs, rd(:, swap) );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_d, rx_pos, 0, [], [], [], false ), rnd );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_d, rx_pos, 0, [], [], [], [] ), rnd );

%% Optional inputs: preset and horizontal
phase_c = quadriga_lib.gbsm_init_phases( sp, 'cluster', 7, 0, 'Comb1024' );
rc = quadriga_lib.gbsm_draw( sp, 'cluster', phase_c, tx_pos, rx_pos, 0, 'Comb1024', [], [], [] );
assertEqual( size(rc), [5, 9 * N] );
assertTrue( max(abs(rc(:) - rnd(:))) > 1e-3 );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [], [], [] ), rnd );
rx_z = rx_pos;
rx_z(:, 3) = rx_z(:, 3) + [ 0; 5; 10; 2; 7 ];
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_z ), rnd );       % 2D: z is not used
r3 = quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [], [], [], false );
r3z = quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_z, 0, [], [], [], [], false );
assertTrue( max(abs(r3z(:) - r3(:))) > 1e-3 );                               % 3D: z is used
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [], [], [], true ), rnd );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [], [], [], [] ), rnd );

%% scenpar: NumClusters = 1 gives an empty set and accepts an empty phase
sp0 = struct( 'NumClusters', 1, 'NumClusters_min', 1 );
r0 = quadriga_lib.gbsm_draw( sp0, 'cluster', quadriga_lib.gbsm_init_phases( sp0, 'cluster', 7 ), tx_pos, rx_pos );
assertTrue( isempty(r0) );

%% Outputs: none
quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos );

%% Type handling
rt = quadriga_lib.gbsm_draw( sp, 'cluster', phase, single(tx_pos), single(rx_pos), uint32(0), [], int32(fl), [], int32(0) );
assertTrue( isa(rt, 'double') );
assertEqual( rt, rl );                                                         % positions exact in single
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', phase_f, tx_pos, rx_pos, int64(1) ), rf );

%% Errors
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, 'Comb256', [], [], false, true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( sp, 'cluster', phase, tx_pos, rx_pos ), 'Wrong number of output arguments.' );
msg_set = 'Input ''set_id'' must be 0 (shared set) or a frequency index >= 1.';
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, -1 ), msg_set );
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0.5 ), msg_set );
msg_fl = 'Entries in ''floor_id'' must be non-negative integers below 2^32.';
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [ 0; -1; 0; 0; 0 ] ), msg_fl );
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], [ 0; 1.5; 0; 0; 0 ] ), msg_fl );
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase_f, tx_pos, rx_pos ), ...
    'Input ''phase'' must have size [256, 108, 2] for preset ''Comb256'' and this set.' );      % C++
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 1 ), '[256, 12, 2]' );     % C++
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, zeros(0, 3) ), 'rx_pos' );        % zero links
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'cluster', phase, tx_pos, rx_pos, 0, [], fl, [], [], false ), 'floor_id' );

%% Set types, set index and center_freq
sp = struct( 'DS_lambda', 10 );
ph = quadriga_lib.gbsm_init_phases( sp, 'lsf', 7 );
r = quadriga_lib.gbsm_draw( sp, 'lsf', ph, tx_pos, rx_pos );
assertEqual( quadriga_lib.gbsm_draw( sp, 'lsf', ph, tx_pos, rx_pos, 5 ), r );           % set_id is ignored for 'lsf'
assertEqual( quadriga_lib.gbsm_draw( sp, 'lsf', ph, tx_pos, rx_pos, [], [], [], [ -1, NaN ] ), r );  % center_freq ignored
pc = quadriga_lib.gbsm_init_phases( sp, 'cluster', 7 );
rc = quadriga_lib.gbsm_draw( sp, 'cluster', pc, tx_pos, rx_pos );
assertEqual( quadriga_lib.gbsm_draw( sp, 'cluster', pc, tx_pos, rx_pos, 0, [], [], 3.5e9 ), rc );
assert_throws( @() quadriga_lib.gbsm_draw( sp, 'rays', ph, tx_pos, rx_pos ), ...
    'Input ''set_type'' must be ''lsf'', ''cluster'' or ''ray''.' );                               % C++

%% Set type 'ray': [n_link, 14·N·M + N]; center_freq is required by the shared set (set_id = 0) only
sr = struct( 'NumClusters', 3, 'NumClusters_min', 1, 'NumSubPaths', 4, 'SC_lambda', 15, ...
    'PerClusterDS', 11, 'PerClusterDS_gamma', -2, 'PerClusterAS_A', 22 );            % N = 2, M = 4
n_link = size( rx_pos, 1 );
pr = quadriga_lib.gbsm_init_phases( sr, 'ray', 7 );
rr = quadriga_lib.gbsm_draw( sr, 'ray', pr, tx_pos, rx_pos, 0, [], [], 3.5e9 );
assertEqual( size(rr), [n_link, 114] );
assertTrue( all( all( rr(:, 107:114) > 0 & rr(:, 107:114) < 1 ) ) );               % block 14, Uniform
assertTrue( max(max(abs(rr(:, 1:106)))) < 6 && std(rr(:)) > 0.1 );
assertFalse( isequal( quadriga_lib.gbsm_draw( sr, 'ray', pr, tx_pos, rx_pos, 0, [], [], 28e9 ), rr ) );
assert_throws( @() quadriga_lib.gbsm_draw( sr, 'ray', pr, tx_pos, rx_pos ), ...
    'Input ''center_frequency'' is required for the shared ''ray'' set.' );                       % C++
assert_throws( @() quadriga_lib.gbsm_draw( sr, 'ray', pr, tx_pos, rx_pos, 0, [], [], -1 ), ...
    'Input ''center_frequency'' must be finite and > 0.' );                                      % C++
pk = quadriga_lib.gbsm_init_phases( sr, 'ray', 7, 2 );
rk = quadriga_lib.gbsm_draw( sr, 'ray', pk, tx_pos, rx_pos, 2 );                         % k >= 1
assertEqual( size(rk), [n_link, 114] );
assertEqual( quadriga_lib.gbsm_draw( sr, 'ray', pk, tx_pos, rx_pos, 2, [], [], [ -1, NaN ] ), rk );
assert_throws( @() quadriga_lib.gbsm_draw( sr, 'ray', pk, tx_pos, rx_pos, -1 ), msg_set );

end
% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.gbsm_draw( varargin{:} ); %#ok<ASGLU>
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
