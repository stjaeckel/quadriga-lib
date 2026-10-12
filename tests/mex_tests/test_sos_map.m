function test_sos_map
% MEX interface test for quadriga_lib.sos_map.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_map.cpp).

%% Setup: one sinusoid per axis with different wavelengths (4 m, 8 m, 16 m), amplitude sqrt(2/3)
freq  = diag( [1/4, 1/8, 1/16] );
phase = [0, pi/2; 0, 0; 0, 0];
x = [0, 1, 2]; y = [0, 4]; z = [0, 4, 8, 16];
a = sqrt(2/3);

%% Basic call: meshgrid order [ny, nx, nz, n_gen]
map = quadriga_lib.sos_map( freq, phase, x, y, z, [], 1 );
assertEqual( size(map), [2, 3, 4, 2] );
assertTrue( isa(map, 'double') );
for ix = 1:3
    for iy = 1:2
        for iz = 1:4
            cy = cos( pi/4 * y(iy) ); cz = cos( pi/8 * z(iz) );
            assertElementsAlmostEqual( map(iy, ix, iz, 1), a * ( cos( pi/2 * x(ix) ) + cy + cz ), 'absolute', 1e-13 );
            assertElementsAlmostEqual( map(iy, ix, iz, 2), a * ( -sin( pi/2 * x(ix) ) + cy + cz ), 'absolute', 1e-13 );
        end
    end
end

%% Optional inputs: omitted, empty, explicit
m1 = quadriga_lib.sos_map( freq, phase, x );                                   % y and z omitted: one sample at 0
assertEqual( size(m1), [1, 3, 1, 2] );
assertElementsAlmostEqual( m1(1, :, 1, 1), a * ( cos( pi/2 * x ) + 2 ), 'absolute', 1e-6 );
m2 = quadriga_lib.sos_map( freq, phase, x, [], [], [], [] );
assertEqual( m2, m1 );
m3 = quadriga_lib.sos_map( freq, phase, x, y );                                % [ny, nx, 1, n_gen]
assertEqual( size(m3), [2, 3, 1, 2] );
assertElementsAlmostEqual( m3, map(:, :, 1, :), 'absolute', 1e-6 );
m4 = quadriga_lib.sos_map( freq, phase, x, [], z, [], 1 );                     % z without y
assertEqual( size(m4), [1, 3, 4, 2] );
assertElementsAlmostEqual( m4, map(1, :, :, :), 'absolute', 1e-13 );
m5 = quadriga_lib.sos_map( freq, phase, x, y, z, true, 1 );                    % uniform
assertElementsAlmostEqual( m5, 0.5 * erfc( -map / sqrt(2) ), 'absolute', 1e-14 );
m6 = quadriga_lib.sos_map( freq, phase, x, y, z, false, 0 );                   % automatic kernel
assertElementsAlmostEqual( m6, map, 'absolute', 1e-6 );

%% Outputs: none
quadriga_lib.sos_map( freq, phase, x );

%% Type handling and orientation: column vectors, single precision
m7 = quadriga_lib.sos_map( single(freq), single(phase), single(x'), single(y'), single(z'), false, int32(1) );
assertTrue( isa(m7, 'double') );
assertElementsAlmostEqual( m7, map, 'absolute', 1e-6 );

%% Errors
assert_throws( @() quadriga_lib.sos_map( freq, phase ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_map( freq, phase, x, y, z, 0, 0, 0 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( freq, phase, x ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_map( freq, phase, [] ), 'x' );                       % thrown by C++
assert_throws( @() quadriga_lib.sos_map( freq(:, 1:2), phase, x, y, z ), 'z' );          % thrown by C++
assert_throws( @() quadriga_lib.sos_map( freq, phase(1:2, :), x ), 'phase' );            % thrown by C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.sos_map( varargin{:} ); %#ok<ASGLU>
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
