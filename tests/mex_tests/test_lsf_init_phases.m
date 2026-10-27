function test_lsf_init_phases
% MEX interface test for quadriga_lib.lsf_init_phases.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_lsf_init_phases.cpp).

%% Setup: distinct decorrelation distances, 4 Rx positions, one Tx
scenpar = struct( 'DS_lambda', 10, 'AS_D_lambda', 18, 'AS_A_lambda', 15 );
rx_pos = [ 0, 0, 1.5; 20, 5, 1.5; 45, -10, 3; 80, 30, 1.5 ];
tx_pos = [ 10, 10, 25 ];
mirror = [ 1, 2, 3, 5, 4, 7, 6, 8, 9, 10, 11 ];                        % ASD <-> ASA, ESD <-> ESA

%% Basic call and values
phase = quadriga_lib.lsf_init_phases( scenpar, 7 );
assertEqual( size(phase), [256, 11, 2] );
assertTrue( isa(phase, 'double') );
assertEqual( phase(:, :, 1), quadriga_lib.sos_init_phases( 256, 11, 7 ) );    % raw phases without positions
assertEqual( phase(:, :, 2), phase(:, mirror, 1) );                           % slice 2 mirrors slice 1
assertTrue( all( phase(:) >= -pi & phase(:) < pi ) );

%% Optional inputs: seed
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7 ), phase );             % same seed, same phases
assertEqual( quadriga_lib.lsf_init_phases( scenpar, int64(7) ), phase );
p1 = quadriga_lib.lsf_init_phases( scenpar );                                 % seed -1: system random device
p2 = quadriga_lib.lsf_init_phases( scenpar, [] );
assertEqual( size(p1), [256, 11, 2] );
assertEqual( size(p2), [256, 11, 2] );
assertTrue( max(abs(p1(:) - p2(:))) > 0.1 );
assertEqual( p2(:, :, 2), p2(:, mirror, 1) );

%% Optional inputs: positions and bias removal
pb = quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos );                  % bias removal over 4 positions
assertEqual( pb(:, :, 2), pb(:, mirror, 1) );
assertTrue( all( max(abs(pb(:, 1:9, 1) - phase(:, 1:9, 1))) > 1e-6 ) );       % Normal generators corrected
assertEqual( pb(:, 10:11, :), phase(:, 10:11, :) );                           % Uniform generators unchanged

% The corrected fields have a zero mean over the positions (lsf_draw may use single-precision sin/cos)
rnd = quadriga_lib.lsf_draw( scenpar, pb, tx_pos, rx_pos );
assertElementsAlmostEqual( mean(rnd(:, 1:9)), zeros(1, 9), 'absolute', 1e-4 );

assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos(1:2, :) ), phase );     % 2 positions: none
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, [], false ), phase );  % remove_bias = false
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, [], [] ), pb );        % [] = true
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, tx_pos, rx_pos ), pb );            % tx_pos unused
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, false, true, 'Comb256', true ), pb );
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, [], [], [], [] ), pb );

% Dual mobility: bias removal over the Rx and Tx positions; one Tx row equals n_link equal rows
pd = quadriga_lib.lsf_init_phases( scenpar, 7, tx_pos, rx_pos, true );
assertTrue( max(abs(pd(:) - pb(:))) > 1e-6 );
assertEqual( pd(:, :, 2), pd(:, mirror, 1) );
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, repmat(tx_pos, 4, 1), rx_pos, true ), pd );

% horizontal: z is not used in 2D, used in 3D
rx_z = rx_pos;
rx_z(:, 3) = [ 0; 7; 2; 11 ];
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_z ), pb );
ph = quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, [], [], [], false );
assertTrue( max(abs(ph(:) - pb(:))) > 1e-6 );
assertEqual( quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, [], [], [], true ), pb );

% preset: sets n_coeff
pc = quadriga_lib.lsf_init_phases( scenpar, 7, [], [], [], [], 'Comb1024' );
assertEqual( size(pc), [1024, 11, 2] );
assertEqual( pc(:, :, 1), quadriga_lib.sos_init_phases( 1024, 11, 7 ) );

% scenpar: [] = defaults; the decorrelation distances reach the bias removal
assertEqual( quadriga_lib.lsf_init_phases( [], 7 ), phase );
pbd = quadriga_lib.lsf_init_phases( [], 7, [], rx_pos );
assertTrue( max(abs(pbd(:, 1, 1) - pb(:, 1, 1))) > 1e-6 );                    % DS_lambda differs
assertEqual( pbd(:, 2, 1), pb(:, 2, 1) );                                      % KF_lambda is the default in both
sv = quadriga_lib.scenpar_validate( scenpar );                                 % complete struct incl. 'SubpathMethod'
assertEqual( quadriga_lib.lsf_init_phases( sv, 7, [], rx_pos ), pb );

%% Outputs: none
quadriga_lib.lsf_init_phases( scenpar, 7 );

%% Type handling
ps = quadriga_lib.lsf_init_phases( scenpar, int64(7), [], single(rx_pos), int32(0), uint32(1) );
assertTrue( isa(ps, 'double') );
assertEqual( ps, pb );                                                         % positions are exact in single
pt = quadriga_lib.lsf_init_phases( scenpar, uint64(7), single(tx_pos), single(rx_pos), true );
assertEqual( pt, pd );

%% Errors
assert_throws( @() quadriga_lib.lsf_init_phases(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.lsf_init_phases( scenpar, 7, [], [], false, true, 'Comb256', true, 1 ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( scenpar, 7 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.lsf_init_phases( scenpar, 7, [], [], [], [], 'Comb999' ), ...
    'Input ''preset'' is not a known preset.' );                                                    % C++
assert_throws( @() quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos, true ), ...
    'Input ''tx_pos'' is required for the bias removal with dual mobility.' );                      % C++
assert_throws( @() quadriga_lib.lsf_init_phases( scenpar, 7, [], rx_pos(:, 1:2) ), ...
    'Input ''rx_pos'' must have 3 columns.' );                                                      % C++
assert_throws( @() quadriga_lib.lsf_init_phases( struct( 'Foo', 1 ), 7 ), ...
    'Unknown key ''Foo'' in ''scenpar''.' );                                                        % C++
assert_throws( @() quadriga_lib.lsf_init_phases( struct( 'DS_lambda', -1 ), 7 ), 'DS_lambda' );     % C++
assert_throws( @() quadriga_lib.lsf_init_phases( 5, 7 ), 'Input ''scenpar'' must be a 1x1 struct.' );
assert_throws( @() quadriga_lib.lsf_init_phases( scenpar, -2 ), 'Input ''seed''' );

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.lsf_init_phases( varargin{:} ); %#ok<ASGLU>
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
