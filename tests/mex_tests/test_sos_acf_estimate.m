function test_sos_acf_estimate
% MEX interface test for quadriga_lib.sos_acf_estimate.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_acf_estimate.cpp).

%% Setup
[ freq, acf, dist ] = quadriga_lib.sos_preset( 'Gauss256' );
phase = quadriga_lib.sos_init_phases( 256, 1, 3 );
bins = [2, 5, 10, 15, 22, 30];

%% Basic call, all outputs
[ acf_est, dist_est, acf_est_dual ] = quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, acf, dist, 1 );
assertEqual( size(acf_est), [6, 1] );
assertEqual( size(dist_est), [6, 1] );
assertEqual( size(acf_est_dual), [6, 1] );
assertTrue( isa(acf_est, 'double') );
assertTrue( all(isfinite(acf_est)) && all(abs(acf_est) <= 1) );
assertTrue( acf_est(1) > 0.7 );                                                % 2 m with a Gaussian ACF of 10 m
assertTrue( abs(acf_est(6)) < 0.4 );                                           % 30 m
assertTrue( all(dist_est > [0; 3.5; 7.5; 12.5; 18.5; 26]) && all(dist_est <= [3.5; 7.5; 12.5; 18.5; 26; 32.8]) );

%% Optional inputs: omitted, empty, explicit
a2 = quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, [], [], 1 );       % acf, dist only needed for the dual output
assertEqual( a2, acf_est );
a3 = quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, [], [], 2 );       % another seed, other positions
assertTrue( max(abs(a3 - acf_est)) > 1e-4 );
a4 = quadriga_lib.sos_acf_estimate( freq, phase, bins, 150 );                  % seed -1
assertEqual( size(a4), [6, 1] );
a5 = quadriga_lib.sos_acf_estimate( freq, phase, bins, 20, [], [], 1 );        % n_pos reaches the library
assertTrue( max(abs(a5 - acf_est)) > 1e-4 || any(isnan(a5)) );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, [], [], 1 );
[ ~, d6 ] = quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, [], [], 1 );
assertEqual( d6, dist_est );

%% Type handling and orientation
a7 = quadriga_lib.sos_acf_estimate( single(freq), single(phase'), single(bins'), uint32(150), [], [], int32(1) );
assertTrue( isa(a7, 'double') );
assertElementsAlmostEqual( a7, acf_est, 'absolute', 1e-3 );

%% Errors
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, [], [], 1, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( freq, phase, bins, 150, acf, dist, 1 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_acf_estimate( freq(:, 1:2), phase, bins, 150 ), 'freq' );       % thrown by C++
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins, 1 ), 'n_pos' );                % thrown by C++

% n_pos and the seed are validated by the wrapper before the conversion to an integer type
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins, -1 ), 'Input ''n_pos'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins, NaN ), 'Input ''n_pos'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins, 0 ), 'Input ''n_pos'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, bins, 150, [], [], NaN ), 'Input ''seed''' );
assert_throws( @() quadriga_lib.sos_acf_estimate( freq, phase, [bins(1:end-1), Inf], 150 ), 'bins' );  % thrown by C++
assert_throws( @() dual_without_acf( freq, phase, bins ), 'acf' );                                  % thrown by C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.sos_acf_estimate( varargin{:} ); %#ok<ASGLU>
end

% The two-device estimate needs the ACF table
function dual_without_acf(freq, phase, bins)
[~, ~, ~] = quadriga_lib.sos_acf_estimate( freq, phase, bins, 150 ); %#ok<ASGLU>
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
