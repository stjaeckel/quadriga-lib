function test_sos_remove_bias
% MEX interface test for quadriga_lib.sos_remove_bias.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_remove_bias.cpp).

%% Setup
freq  = quadriga_lib.sos_preset( 'Comb128' );
phase = quadriga_lib.sos_init_phases( 128, 3, 5 );
phase_b = quadriga_lib.sos_init_phases( 128, 3, 6 );
[x, y] = meshgrid( 0:5:20, 0:5:15 );
pos   = [x(:), y(:), 1.5 * ones(numel(x), 1)];                                 % 20 positions
pos_b = [y(:) + 3, x(:) - 2, 1.5 * ones(numel(x), 1)];
[ ~, acf, dist ] = quadriga_lib.sos_preset( 'Comb128' );

%% Basic call, one device, all outputs
[ phase_out, phase_b_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( freq, phase, pos );
assertEqual( size(phase_out), [128, 3] );
assertTrue( isempty(phase_b_out) );
assertEqual( size(offset), [3, 1] );
assertEqual( size(mean_before), [3, 1] );
assertTrue( isa(phase_out, 'double') && isa(offset, 'double') && isa(mean_before, 'double') );
m0 = mean( quadriga_lib.sos_val( freq, phase, pos, [], [], [], [], [], [], 1 ), 1 );
m1 = mean( quadriga_lib.sos_val( freq, phase_out, pos, [], [], [], [], [], [], 1 ), 1 );
assertTrue( all(abs(m0) > 1e-3) );
assertElementsAlmostEqual( m1, [0, 0, 0], 'absolute', 1e-13 );
assertElementsAlmostEqual( mean_before, m0', 'absolute', 1e-13 );              % one row per generator
d = angle( exp( 1j * ( phase_out - phase - repmat( offset', 128, 1 ) ) ) );    % one common offset per column
assertElementsAlmostEqual( d, zeros(128, 3), 'absolute', 1e-13 );
assertTrue( all(phase_out(:) >= -pi) && all(phase_out(:) < pi) );
assertTrue( all(abs(offset) <= pi/2 + 1e-12) );
assertTrue( max(abs(offset)) > 1e-3 );

%% The input is not modified
phase_copy = quadriga_lib.sos_init_phases( 128, 3, 5 );
assertEqual( phase, phase_copy );

%% Optional inputs: omitted, empty, explicit
p2 = quadriga_lib.sos_remove_bias( freq, phase, pos, [], [], [], [], [] );
assertEqual( p2, phase_out );

% Two devices on the same map
[ p3, pb3, o3, mb3 ] = quadriga_lib.sos_remove_bias( freq, phase, pos, pos_b, [], acf, dist );
assertTrue( isempty(pb3) );
assertEqual( size(o3), [3, 1] );
m3 = mean( quadriga_lib.sos_val( freq, p3, pos, pos_b, [], acf, dist, [], [], 1 ), 1 );
assertElementsAlmostEqual( m3, [0, 0, 0], 'absolute', 1e-13 );
m3b = mean( quadriga_lib.sos_val( freq, phase, pos, pos_b, [], acf, dist, [], [], 1 ), 1 );
assertElementsAlmostEqual( mb3, m3b', 'absolute', 1e-13 );

% Two devices on different maps, with a frequency scale for the second device
[ p4, pb4, o4, mb4 ] = quadriga_lib.sos_remove_bias( freq, phase, pos, pos_b, phase_b, [], [], 2.5 );
assertEqual( size(pb4), [128, 3] );
assertEqual( size(o4), [3, 2] );
assertEqual( size(mb4), [3, 2] );
assertEqual( p4, phase_out );                                                  % the first device is treated on its own
m4b = mean( quadriga_lib.sos_val( freq * 2.5, pb4, pos_b, [], [], [], [], [], [], 1 ), 1 );
assertElementsAlmostEqual( m4b, [0, 0, 0], 'absolute', 1e-13 );
m4b0 = mean( quadriga_lib.sos_val( freq * 2.5, phase_b, pos_b, [], [], [], [], [], [], 1 ), 1 );
assertElementsAlmostEqual( mb4, [m0', m4b0'], 'absolute', 1e-13 );             % column 1: first device, column 2: second
assertElementsAlmostEqual( o4(:, 1), offset, 'absolute', 1e-14 );
d4 = angle( exp( 1j * ( pb4 - phase_b - repmat( o4(:, 2)', 128, 1 ) ) ) );
assertElementsAlmostEqual( d4, zeros(128, 3), 'absolute', 1e-13 );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_remove_bias( freq, phase, pos );
p5 = quadriga_lib.sos_remove_bias( freq, phase, pos );
assertEqual( p5, phase_out );
[ ~, ~, o6 ] = quadriga_lib.sos_remove_bias( freq, phase, pos );
assertEqual( o6, offset );

%% Type handling
p7 = quadriga_lib.sos_remove_bias( single(freq), single(phase), single(pos) );
assertTrue( isa(p7, 'double') );
assertElementsAlmostEqual( angle( exp( 1j * ( p7 - phase_out ) ) ), zeros(128, 3), 'absolute', 1e-4 );

%% Errors
assert_throws( @() quadriga_lib.sos_remove_bias( freq, phase ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_remove_bias( freq, phase, pos, [], [], [], [], 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out5( freq, phase, pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_remove_bias( freq, phase(1:5, :), pos ), 'phase' );                 % thrown by C++
assert_throws( @() quadriga_lib.sos_remove_bias( freq, phase, pos, pos_b ), 'acf' );                    % thrown by C++
assert_throws( @() quadriga_lib.sos_remove_bias( freq, phase, pos, [], phase_b ), 'phase_b' );          % thrown by C++

end

% Request 5 outputs so that the output-count guard fires
function out5(varargin)
[~, ~, ~, ~, ~] = quadriga_lib.sos_remove_bias( varargin{:} ); %#ok<ASGLU>
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
