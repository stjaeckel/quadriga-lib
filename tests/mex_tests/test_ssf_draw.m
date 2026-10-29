function test_ssf_draw
% MEX interface test for quadriga_lib.ssf_draw.
% Scope: argument and output handling, set index, floor IDs, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_ssf_phases.cpp).

%% Setup: UMi LOS file (N = 12, SC_lambda = 12 m), shared and per-frequency phases, 5 links
conf_dir = fullfile( fileparts( mfilename('fullpath') ), '..', '..', 'data', 'config' );
sp = quadriga_lib.conf_file_read( fullfile( conf_dir, '3GPP_38.901_v19_UMi_LOS.conf' ) );
N = sp.NumClusters - 1;
phase = quadriga_lib.ssf_init_phases( sp, 7 );
phase_f = quadriga_lib.ssf_init_phases( sp, 7, 1 );
rx_pos = [ 0, 0, 1.5; 20, 5, 1.5; 45, -10, 3; 80, 30, 1.5; 3, 4, 10 ];
tx_pos = [ 10, 10, 25 ];
blk = [ 0, 2, 1, 4, 3, 6, 5, 8, 7 ];
swap = reshape( (0:N-1)' + blk * N + 1, 1, [] );                             % departure <-> arrival blocks

%% Basic call and values
rnd = quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos );
assertEqual( size(rnd), [5, 9 * N] );
assertTrue( isa(rnd, 'double') );
assertTrue( all( all( rnd(:, 1:N) > 0 & rnd(:, 1:N) < 1 ) ) );               % Uniform delay block
assertTrue( max(max(abs(rnd(:, N+1:end)))) < 6 && std(rnd(:)) > 0.1 );

% Column N + 1 (sign of AOD, cluster 0) is the field of slice 1 at the Rx positions projected on the
% x-y plane, at SC_lambda; column 1 is its uniform map. ssf_draw may use single-precision sin/cos (AVX2)
freq = quadriga_lib.sos_preset( 'Comb256', sp.SC_lambda );
pos_xy = [ rx_pos(:, 1:2), zeros(5, 1) ];
ref = quadriga_lib.sos_val( freq, phase(:, N + 1, 1), pos_xy );
assertElementsAlmostEqual( rnd(:, N + 1), ref, 'absolute', 1e-5 );
ref = quadriga_lib.sos_val( freq, phase(:, 1, 1), pos_xy );
assertElementsAlmostEqual( rnd(:, 1), 0.5 * erfc( -ref / sqrt(2) ), 'absolute', 1e-5 );

% Single mobility: the values do not depend on tx_pos; one link alone gives its row
assertEqual( quadriga_lib.ssf_draw( sp, phase, [ 0, 0, 0 ], rx_pos ), rnd );
assertElementsAlmostEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos(3, :) ), rnd(3, :), 'absolute', 1e-12 );

%% Optional inputs: set_id (0 = shared set, k >= 1 = per-frequency set)
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0 ), rnd );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, [] ), rnd );
rf = quadriga_lib.ssf_draw( sp, phase_f, tx_pos, rx_pos, 1 );
assertEqual( size(rf), [5, N] );
assertTrue( any( rf(:) < 0 ) );                                               % Normal shadowing values
assertEqual( quadriga_lib.ssf_draw( sp, phase_f, tx_pos, rx_pos, 2 ), rf );   % same set type
ref = quadriga_lib.sos_val( freq, phase_f(:, 1, 1), pos_xy );
assertElementsAlmostEqual( rf(:, 1), ref, 'absolute', 1e-5 );

%% Optional inputs: floor_id (labels, no index shift)
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [] ), rnd );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, zeros(5, 1) ), rnd );
fl = [ 0; 1; 0; 2; 1 ];
rl = quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, fl );
assertEqual( rl([1, 3], :), rnd([1, 3], :) );                                % floor 0 unchanged
assertTrue( all( max(abs(rl([2, 4, 5], :) - rnd([2, 4, 5], :)), [], 2) > 1e-3 ) );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, fl' ), rl );           % row vector

%% Optional inputs: dual mobility, swap reciprocity
tx_d = [ 5, 5, 1.5; 30, 0, 1.5; 40, -20, 1.5; 60, 10, 1.5; 0, 9, 1.5 ];
rd = quadriga_lib.ssf_draw( sp, phase, tx_d, rx_pos, 0, [], true );
assertTrue( max(abs(rd(:) - rnd(:))) > 1e-3 );
rs = quadriga_lib.ssf_draw( sp, phase, rx_pos, tx_d, 0, [], true );
assertEqual( rs, rd(:, swap) );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_d, rx_pos, 0, [], false ), rnd );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_d, rx_pos, 0, [], [] ), rnd );

%% Optional inputs: preset and horizontal
phase_c = quadriga_lib.ssf_init_phases( sp, 7, 0, 'Comb1024' );
rc = quadriga_lib.ssf_draw( sp, phase_c, tx_pos, rx_pos, 0, [], [], 'Comb1024' );
assertEqual( size(rc), [5, 9 * N] );
assertTrue( max(abs(rc(:) - rnd(:))) > 1e-3 );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [], [], [] ), rnd );
rx_z = rx_pos;
rx_z(:, 3) = rx_z(:, 3) + [ 0; 5; 10; 2; 7 ];
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_z ), rnd );       % 2D: z is not used
r3 = quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [], [], [], false );
r3z = quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_z, 0, [], [], [], false );
assertTrue( max(abs(r3z(:) - r3(:))) > 1e-3 );                               % 3D: z is used
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [], [], [], true ), rnd );
assertEqual( quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [], [], [], [] ), rnd );

%% scenpar: NumClusters = 1 gives an empty set and accepts an empty phase
sp0 = struct( 'NumClusters', 1, 'NumClusters_min', 1 );
r0 = quadriga_lib.ssf_draw( sp0, quadriga_lib.ssf_init_phases( sp0, 7 ), tx_pos, rx_pos );
assertTrue( isempty(r0) );

%% Outputs: none
quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos );

%% Type handling
rt = quadriga_lib.ssf_draw( sp, phase, single(tx_pos), single(rx_pos), uint32(0), int32(fl), int32(0) );
assertTrue( isa(rt, 'double') );
assertEqual( rt, rl );                                                         % positions exact in single
assertEqual( quadriga_lib.ssf_draw( sp, phase_f, tx_pos, rx_pos, int64(1) ), rf );

%% Errors
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [], false, 'Comb256', true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( sp, phase, tx_pos, rx_pos ), 'Wrong number of output arguments.' );
msg_set = 'Input ''set_id'' must be 0 (shared set) or a frequency index >= 1.';
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, -1 ), msg_set );
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0.5 ), msg_set );
msg_fl = 'Entries in ''floor_id'' must be non-negative integers below 2^53.';
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [ 0; -1; 0; 0; 0 ] ), msg_fl );
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, [ 0; 1.5; 0; 0; 0 ] ), msg_fl );
assert_throws( @() quadriga_lib.ssf_draw( sp, phase_f, tx_pos, rx_pos ), ...
    'Input ''phase'' must have size [256, 108, 2] for preset ''Comb256'' and this set.' );      % C++
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 1 ), '[256, 12, 2]' );     % C++
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, zeros(0, 3) ), 'rx_pos' );        % zero links
assert_throws( @() quadriga_lib.ssf_draw( sp, phase, tx_pos, rx_pos, 0, fl, [], [], false ), 'floor_id' );

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.ssf_draw( varargin{:} ); %#ok<ASGLU>
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
