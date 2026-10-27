function test_lsf_draw
% MEX interface test for quadriga_lib.lsf_draw.
% Scope: argument and output handling, floor IDs, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_gbsm_phases.cpp).

%% Setup: phases of a scenario with distinct decorrelation distances, 5 links
scenpar = struct( 'DS_lambda', 10, 'AS_D_lambda', 18, 'AS_A_lambda', 15, 'ES_D_lambda', 12, 'ES_A_lambda', 8 );
phase = quadriga_lib.lsf_init_phases( scenpar, 7 );
rx_pos = [ 0, 0, 1.5; 20, 5, 1.5; 45, -10, 3; 80, 30, 1.5; 3, 4, 10 ];
tx_pos = [ 10, 10, 25 ];
swap = [ 1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11 ];                          % ASD <-> ASA, ESD <-> ESA

%% Basic call and values
rnd = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos );
assertEqual( size(rnd), [5, 11] );
assertTrue( isa(rnd, 'double') );
assertTrue( all( all( rnd(:, 10:11) > 0 & rnd(:, 10:11) < 1 ) ) );           % Uniform columns
assertTrue( max(max(abs(rnd(:, 1:9)))) < 6 && std(rnd(:)) > 0.1 );

% Column 1 is the DS field (10 m) of slice 1 at the Rx positions projected on the x-y plane; lsf_draw may
% evaluate sin/cos in single precision (AVX2 kernel)
freq = quadriga_lib.sos_preset( 'Comb256', 10 );
ref = quadriga_lib.sos_val( freq, phase(:, 1, 1), [ rx_pos(:, 1:2), zeros(5, 1) ] );
assertElementsAlmostEqual( rnd(:, 1), ref, 'absolute', 1e-5 );

% Single mobility: the values do not depend on tx_pos; one link alone gives its row
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, [ 0, 0, 0 ], rx_pos ), rnd );
assertElementsAlmostEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos(3, :) ), rnd(3, :), 'absolute', 1e-12 );

%% Optional inputs: floor_id (labels, no index shift)
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [] ), rnd );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, zeros(5, 1) ), rnd );
fl = [ 0; 1; 0; 2; 1 ];
rf = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, fl );
assertEqual( rf([1, 3], :), rnd([1, 3], :) );                                  % floor 0 unchanged
assertTrue( all( max(abs(rf([2, 4, 5], :) - rnd([2, 4, 5], :)), [], 2) > 1e-3 ) );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, fl' ), rf );           % row vector

% Floor 1 adds the offsets 2·pi·n/256 to the phases (n = 0 ... 255)
phase_1 = phase + repmat( 2 * pi * (0:255)' / 256, [1, 11, 2] );
r1 = quadriga_lib.lsf_draw( scenpar, phase_1, tx_pos, rx_pos([2, 5], :) );
assertElementsAlmostEqual( r1, rf([2, 5], :), 'absolute', 1e-10 );

%% Optional inputs: dual mobility, swap reciprocity
tx_d = [ 5, 5, 1.5; 30, 0, 1.5; 40, -20, 1.5; 60, 10, 1.5; 0, 9, 1.5 ];
rd = quadriga_lib.lsf_draw( scenpar, phase, tx_d, rx_pos, [], true );
assertTrue( max(abs(rd(:) - rnd(:))) > 1e-3 );
rs = quadriga_lib.lsf_draw( scenpar, phase, rx_pos, tx_d, [], true );
assertEqual( rs, rd(:, swap) );                                                % columns 4/5 and 6/7 swapped
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_d, rx_pos, [], false ), rnd );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_d, rx_pos, [], [] ), rnd );

%% Optional inputs: horizontal
rx_z = rx_pos;
rx_z(:, 3) = rx_z(:, 3) + [ 0; 5; 10; 2; 7 ];
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_z ), rnd );     % 2D: z is not used
r3 = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], [], [], false );
r3z = quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_z, [], [], [], false );
assertTrue( max(abs(r3z(:) - r3(:))) > 1e-3 );                                 % 3D: z is used
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], [], [], true ), rnd );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], [], [], [] ), rnd );

%% Optional inputs: preset
phase_c = quadriga_lib.lsf_init_phases( scenpar, 7, [], [], [], [], 'Comb1024' );
rc = quadriga_lib.lsf_draw( scenpar, phase_c, tx_pos, rx_pos, [], [], 'Comb1024' );
assertEqual( size(rc), [5, 11] );
assertTrue( max(abs(rc(:) - rnd(:))) > 1e-3 );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], [], 'Comb256' ), rnd );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], [], [] ), rnd );

%% scenpar: another DS decorrelation distance changes column 1 only; the full struct is accepted
sp2 = scenpar;
sp2.DS_lambda = 50;
r2 = quadriga_lib.lsf_draw( sp2, phase, tx_pos, rx_pos );
assertTrue( max(abs(r2(:, 1) - rnd(:, 1))) > 1e-3 );
assertEqual( r2(:, 2:end), rnd(:, 2:end) );
sv = quadriga_lib.scenpar_validate( scenpar );                                 % with 'SubpathMethod'
assertEqual( quadriga_lib.lsf_draw( sv, phase, tx_pos, rx_pos ), rnd );

%% Outputs: none
quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos );

%% Type handling
rt = quadriga_lib.lsf_draw( scenpar, single(phase), single(tx_pos), single(rx_pos), single(fl), int32(0) );
assertTrue( isa(rt, 'double') );
assertElementsAlmostEqual( rt, rf, 'absolute', 1e-4 );                         % phases rounded to single
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, uint32(fl) ), rf );
assertEqual( quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, int64(fl), false, 'Comb256', true ), rf );

%% Errors
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], false, 'Comb256', true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( scenpar, phase, tx_pos, rx_pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [ 0; -1; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^53.' );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [ 0; 1.5; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^53.' );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [ 0; NaN; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^53.' );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [ 0; 2^53; 0; 0; 0 ] ), ...
    'Entries in ''floor_id'' must be non-negative integers below 2^53.' );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase(:, 1:10, :), tx_pos, rx_pos ), ...
    'Input ''phase'' must have size [256, 11, 2] for preset ''Comb256''.' );                        % C++
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [], [], 'Comb1024' ), ...
    'Input ''phase'' must have size [1024, 11, 2] for preset ''Comb1024''.' );                      % C++
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, [ 0; 1 ] ), 'floor_id' );    % C++
msg3d = 'Input ''floor_id'' cannot be used with ''horizontal = false'': in 3D mode the height separates the floors.';
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, fl, [], [], false ), msg3d );   % C++
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos, rx_pos, zeros(5, 1), true, 'Comb256', false ), msg3d );
assert_throws( @() quadriga_lib.lsf_draw( scenpar, phase, tx_pos(1:2), rx_pos ), ...
    'Input ''tx_pos'' must have 3 columns.' );                                                      % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.lsf_draw( varargin{:} ); %#ok<ASGLU>
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
