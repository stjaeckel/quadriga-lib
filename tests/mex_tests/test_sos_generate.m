function test_sos_generate
% MEX interface test for quadriga_lib.sos_generate.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_generate.cpp).

%% Setup: 1-D target made of two sinusoids, which the optimiser must recover
dist = 0:0.5:50;
acf = 0.5 * ( cos( 2*pi * 0.04 * dist ) + cos( 2*pi * 0.13 * dist ) );
expo = exp( -(0:1:40) / 10 );

%% Basic call, all inputs and outputs, 1-D
[ freq, ase, test_dir ] = quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10, true, 0, 0, 3 );
assertEqual( size(freq), [2, 1] );                                             % [n_coeff, n_dim]
assertTrue( isa(freq, 'double') && isa(ase, 'double') );
assertElementsAlmostEqual( sort(abs(freq)), [0.04; 0.13], 'absolute', 1e-5 );
assertTrue( size(ase, 2) == 1 && size(ase, 1) >= 1 );
assertTrue( ase(end) < 1e-6 );
assertTrue( isempty(test_dir) );

%% Defaults: n_dim = 3, n_dir = 10, random seed
[ f2, ~, d2 ] = quadriga_lib.sos_generate( expo, 0:40, 12 );
assertEqual( size(f2), [12, 3] );
assertEqual( size(d2), [10, 2] );
[ f3, ~, d3 ] = quadriga_lib.sos_generate( expo, 0:40, 12, [], [], [], [], [], [], [] );
assertEqual( size(f3), [12, 3] );
assertEqual( d3, d2 );

%% Each optional input reaches the library
[ f4, a4, d4 ] = quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, true, 0, 0, 5 );    % n_dim, n_dir
assertEqual( size(f4), [12, 2] );
assertEqual( size(d4), [7, 1] );
assertElementsAlmostEqual( d4, pi * (0:6)' / 7, 'absolute', 1e-14 );
az = pi * (0:11)' / 12;                                                        % uniform_smp = true: regular directions
assertElementsAlmostEqual( f4(:, 2) .* cos(az) - f4(:, 1) .* sin(az), zeros(12, 1), 'absolute', 1e-14 );
f5 = quadriga_lib.sos_generate( expo, 0:40, 12, 2, false, 7, true, 0, 0, 5 );                % uniform_smp = false
assertTrue( max(abs( f5(:, 2) .* cos(az) - f5(:, 1) .* sin(az) )) > 1e-3 );
f6 = quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, false, 0, 0, 5 );                % random_init = false
assertTrue( max(abs(f6(:) - f4(:))) > 1e-6 );
f7 = quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, true, 0.03, 0.2, 5 );            % freq_min, freq_max
mag = sqrt( sum( f7.^2, 2 ) );
assertTrue( all(mag >= 0.03 - 1e-12) && all(mag <= 0.2 + 1e-12) );
assertTrue( min( sqrt( sum( f4.^2, 2 ) ) ) < 0.03 );
f8 = quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, true, 0, 0, 5 );                 % seed: repeatable
assertEqual( f8, f4 );
f9 = quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, true, 0, 0, 6 );
assertTrue( max(abs(f9(:) - f4(:))) > 1e-6 );
assertTrue( all(diff(a4) <= 0) );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, true, 0, 0, 5 );
[ ~, a10 ] = quadriga_lib.sos_generate( expo, 0:40, 12, 2, true, 7, true, 0, 0, 5 );
assertEqual( a10, a4 );

%% Type handling and orientation
f11 = quadriga_lib.sos_generate( acf', dist', uint32(2), int32(1), true, single(10), true, single(0), single(0), int32(3) );
assertEqual( f11, freq );

%% Errors
assert_throws( @() quadriga_lib.sos_generate( acf, dist ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10, true, 0, 0, 3, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( acf, dist, 2, 1 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_generate( acf, dist, 0 ), 'n_coeff' );                   % thrown by C++
assert_throws( @() quadriga_lib.sos_generate( acf, dist, 2, 4 ), 'n_dim' );                  % thrown by C++
assert_throws( @() quadriga_lib.sos_generate( acf, dist(1:5), 2 ), 'dist' );                 % thrown by C++

% Counts and the seed are validated by the wrapper before the conversion to an integer type
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, -1, 1 ), 'Input ''n_coeff'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, NaN, 1 ), 'Input ''n_coeff'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, -1 ), 'Input ''n_dim'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, 0 ), 'Input ''n_dim'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, -1 ), 'Input ''n_dir'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, 0 ), 'Input ''n_dir'' must be an integer >= 1.' );
assert_throws( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10, true, 0, 0, NaN ), 'Input ''seed''' );
assert_throws( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10, true, 0, 1e6 ), 'freq_max' );   % thrown by C++

%% Non-integer counts and seeds are rejected, integer-valued doubles are accepted
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2.9, 1 ), 'Input ''n_coeff'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, 1.9 ), 'Input ''n_dim'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10.9 ), 'Input ''n_dir'' must be an integer >= 1.' );
assert_cpp_error( @() quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10, true, 0, 0, 3.9 ), ...
    'Input ''seed'' must be an integer from -1 to 9223372036854775807.' );
f1 = quadriga_lib.sos_generate( acf, dist, 2.0, 1.0, true, 10.0, true, 0, 0, 3.0 );
f2 = quadriga_lib.sos_generate( acf, dist, 2, 1, true, 10, true, 0, 0, 3 );
assertEqual( f1, f2 );

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.sos_generate( varargin{:} ); %#ok<ASGLU>
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

% Assert that fn() raises quadriga_lib:CPPerror with a message that contains expected (substring).
function assert_cpp_error(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, 'quadriga_lib:CPPerror') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: quadriga_lib:CPPerror "', expected, '", GOT: ', ...
            ME.identifier, ' "', ME.message, '"']);
    end
end
end
