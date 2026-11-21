% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_arrayant_generate
% MEX interface test for quadriga_lib.arrayant_generate.
% Scope: each input position once on one type, omitted and [] inputs, outputs, other classes, error paths.
% Physics validation lives in the Catch2 suite (test_arrayant_generate.cpp).

%% Basic call and output fields
ant = quadriga_lib.arrayant_generate( 'omni', 30 );
assertEqual( size(ant.e_theta_re), [7, 13] );
assertEqual( ant.e_theta_re, ones(7, 13) );
assertEqual( ant.e_phi_re, zeros(7, 13) );
assertElementsAlmostEqual( ant.azimuth_grid, linspace(-pi, pi, 13), 'absolute', 1e-14 );       % [1, n_azimuth]
assertElementsAlmostEqual( ant.elevation_grid, linspace(-pi/2, pi/2, 7), 'absolute', 1e-14 );  % [1, n_elevation]
assertEqual( ant.element_pos, [0; 0; 0] );
assertEqual( [ant.coupling_re, ant.coupling_im], [1, 0] );
assertEqual( ant.center_freq, 299792458 );
assertTrue( strcmp(ant.name, 'omni') );

%% Outputs: none sets ans
quadriga_lib.arrayant_generate( 'omni', 30 );
assertEqual( ans, ant );

%% Omitted and [] inputs give the defaults, at every position
ref = quadriga_lib.arrayant_generate( 'omni' );
assertEqual( size(ref.e_theta_re), [19, 37] );                                       % res default 10 for omni
for k = 1 : 12
    args = cell(1, k);
    assertEqual( quadriga_lib.arrayant_generate( 'omni', args{:} ), ref );
end

%% Each input reaches its C++ argument (positions 2 to 13)
a = quadriga_lib.arrayant_generate( 'xpol', 30, 3e9 );                                % res, center_freq
assertEqual( size(a.e_theta_re), [7, 13, 2] );
assertEqual( a.center_freq, 3e9 );
a = quadriga_lib.arrayant_generate( 'omni', 30, NaN );                                % NaN: default
assertEqual( a.center_freq, 299792458 );

a = quadriga_lib.arrayant_generate('3gpp', 30, [], [], [2, 3]);                         % panel [M, N]
assertEqual( size(a.e_theta_re, 3), 6 );
assertElementsAlmostEqual( a.element_pos(:, 1:2), [0, 0; -0.5, -0.5; -0.25, 0.25], 'absolute', 1e-12 );

a = quadriga_lib.arrayant_generate('ula', 30, [], [], [1, 2], 0.7);                     % spacing
assertElementsAlmostEqual( a.element_pos, [0, 0; -0.35, 0.35; 0, 0], 'absolute', 1e-12 );

a = quadriga_lib.arrayant_generate('3gpp', 30, [], [], 4, [], [], 1);                       % ports
assertEqual( size(a.coupling_re), [4, 1] );

a = quadriga_lib.arrayant_generate('3gpp', 30, [], [], [], [], [2, 2]);             % pol [polarization, model]
assertEqual( size(a.e_theta_re, 3), 2 );
assertElementsAlmostEqual( a.e_theta_re(:, :, 2), zeros(7, 13), 'absolute', 1e-12 );   % Model-2, zeta = 90 deg

a = quadriga_lib.arrayant_generate('3gpp', 30, [], [], 2, [], [], 1, [30, 0]);          % steer: eq. 7.3-1, d_V = 0.5
assertElementsAlmostEqual( [a.coupling_re, a.coupling_im], [0.5, -0.5; 0.5, 0.5], 'absolute', 1e-12 );
a = quadriga_lib.arrayant_generate('3gpp', 30, [], [], [1, 2], [], [], [1, 1], [0, 30]);  % pan: d_H = 0.5
assertElementsAlmostEqual( [a.coupling_re, a.coupling_im], [0.5, 0.5; 0.5, -0.5], 'absolute', 1e-12 );

a = quadriga_lib.arrayant_generate('custom', 30, [], [60, 90, 10]); % element, floor 10 dB
b = quadriga_lib.arrayant_generate('custom', 30, [], 60);
p = a.e_theta_re.^2;
assertElementsAlmostEqual( min(p(:)) / max(p(:)), 0.1, 'absolute', 1e-12 );
assertTrue( max(abs(a.e_theta_re(:) - b.e_theta_re(:))) > 0.01 );

pat = quadriga_lib.arrayant_generate('custom', 30, [], [60, 40]);
a = quadriga_lib.arrayant_generate('3gpp', [], [], [], 2, [], [], [], [], [], pat);     % pattern
assertEqual( size(a.e_theta_re), [7, 13, 2] );
assertElementsAlmostEqual( a.e_theta_re(:, :, 2), pat.e_theta_re, 'absolute', 1e-12 );

beams = [ -30, 30; 0, 0 ];
a = quadriga_lib.arrayant_generate('multibeam', 30, [], [], [1, 4], [], [], [], [], beams, [], true); % beams, separate_beams
assertEqual( size(a.coupling_re), [4, 2] );
a = quadriga_lib.arrayant_generate('multibeam', 30, [], [], [1, 4], [], [], [], [], beams);
assertEqual( size(a.coupling_re), [4, 1] );

a = quadriga_lib.arrayant_generate('3gpp', 30, [], [], 2, [], [], 1, [], [], [], [], true); % apply_weights
assertEqual( size(a.e_theta_re), [7, 13] );
assertElementsAlmostEqual( a.element_pos, [0; 0; 0], 'absolute', 1e-12 );

%% Other classes and orientation (one call)
a = quadriga_lib.arrayant_generate('3gpp', 30, 3e9, [], [2; 3], 0.7, [3, 1], [1, 1], [10, 5]);
b = quadriga_lib.arrayant_generate('3gpp', single(30), single(3e9), [], int32([2, 3]), single(0.7), [3; 1], uint8([1; 1]), single([10, 5]));
assertElementsAlmostEqual( b.e_theta_re, a.e_theta_re, 'absolute', 1e-12 );
assertElementsAlmostEqual( b.coupling_im, a.coupling_im, 'absolute', 1e-6 );         % steer rounded to single
assertElementsAlmostEqual( b.element_pos, a.element_pos, 'absolute', 1e-6 );         % spacing rounded to single
a = quadriga_lib.arrayant_generate('multibeam', 30, [], [], 2, [], [], [], [], [], [], 1, int8(1)); % numeric flags
assertEqual( size(a.e_theta_re, 3), 1 );

%% Errors
assert_throws( @() quadriga_lib.arrayant_generate(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.arrayant_generate( 'omni', [], [], [], [], [], [], [], [], [], [], [], [], [] ), ...
    'Wrong number of input arguments.' );
assert_throws( @() out2( 'omni' ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.arrayant_generate( 'omni', [10, 20] ), 'Input ''res'' must be a scalar.' );
assert_throws( @() quadriga_lib.arrayant_generate('3gpp', 30, [], [], -1), ...
    'Entries in ''panel'' must be non-negative integers.' );
assert_throws( @() quadriga_lib.arrayant_generate('3gpp', 30, [], [], [], [], [], 1.5), ...
    'Entries in ''ports'' must be non-negative integers.' );
assert_throws( @() quadriga_lib.arrayant_generate('3gpp', 30, [], [], [], [], NaN), ...
    'Entries in ''pol'' must be non-negative integers.' );
assert_throws( @() quadriga_lib.arrayant_generate('3gpp', 30, [], [], [], [], [], [], [], [], 0.1), ...
    'Input ''pattern'' must be a struct.' );
assert_throws( @() quadriga_lib.arrayant_generate('omni', 30, [], 90), ...
    'Input ''element'' is not used by type ''omni''.' );                                             % C++
assert_throws( @() quadriga_lib.arrayant_generate( 'bla' ), 'Input ''type'' has an unsupported value ''bla''.' ); % C++

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.arrayant_generate( varargin{:} ); %#ok<ASGLU>
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
