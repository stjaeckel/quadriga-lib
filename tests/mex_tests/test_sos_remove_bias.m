% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_sos_remove_bias
% MEX interface test for quadriga_lib.sos_remove_bias.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_remove_bias.cpp).

%% Setup
spatial_freq  = quadriga_lib.sos_preset( 'Comb128' );
phase = quadriga_lib.sos_init_phases( 128, 3, 5 );
[x, y] = meshgrid( 0:5:20, 0:5:15 );
pos   = [x(:), y(:), 1.5 * ones(numel(x), 1)];                                 % 20 positions
pos_b = [y(:) + 3, x(:) - 2, 1.5 * ones(numel(x), 1)];

%% Basic call, one device, all outputs
[ phase_out, offset, mean_before ] = quadriga_lib.sos_remove_bias( spatial_freq, phase, pos );
assertEqual( size(phase_out), [128, 3] );
assertEqual( size(offset), [3, 1] );
assertEqual( size(mean_before), [3, 1] );
assertTrue( isa(phase_out, 'double') && isa(offset, 'double') && isa(mean_before, 'double') );
m0 = mean( quadriga_lib.sos_val( spatial_freq, phase, pos, [], [], [], [], 1 ), 1 );
m1 = mean( quadriga_lib.sos_val( spatial_freq, phase_out, pos, [], [], [], [], 1 ), 1 );
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
p2 = quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, [], [], [] );
assertEqual( p2, phase_out );
p2 = quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, [], [], false );           % reciprocal: no effect for one device
assertEqual( p2, phase_out );

% Two devices: one offset per generator; the mean of sos_val with the same arguments is removed
% Columns of opt: spatial_freq_scale_b, reciprocal
opt = { 1, true; 1, false; 2.5, false };
o_all = zeros(3, 3);
for k = 1 : 3
    [ p3, o3, mb3 ] = quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, pos_b, opt{k, 1}, opt{k, 2} );
    assertEqual( size(p3), [128, 3] );
    assertEqual( size(o3), [3, 1] );
    assertEqual( size(mb3), [3, 1] );
    m3 = mean( quadriga_lib.sos_val( spatial_freq, p3, pos, pos_b, opt{k, 1}, opt{k, 2}, [], 1 ), 1 );
    assertElementsAlmostEqual( m3, [0, 0, 0], 'absolute', 1e-12 );
    m3b = mean( quadriga_lib.sos_val( spatial_freq, phase, pos, pos_b, opt{k, 1}, opt{k, 2}, [], 1 ), 1 );
    assertElementsAlmostEqual( mb3, m3b', 'absolute', 1e-12 );
    d3 = angle( exp( 1j * ( p3 - phase - repmat( o3', 128, 1 ) ) ) );
    assertElementsAlmostEqual( d3, zeros(128, 3), 'absolute', 1e-13 );
    o_all(:, k) = o3;
end

% pos_b, reciprocal and the frequency scale reach the library
assertTrue( max(abs(o_all(:, 1) - offset)) > 1e-3 );
assertTrue( max(abs(o_all(:, 2) - o_all(:, 1))) > 1e-3 );
assertTrue( max(abs(o_all(:, 3) - o_all(:, 2))) > 1e-3 );

%% Outputs: none, and fewer than the maximum
quadriga_lib.sos_remove_bias( spatial_freq, phase, pos );
assertEqual( ans, phase_out );
p5 = quadriga_lib.sos_remove_bias( spatial_freq, phase, pos );
assertEqual( p5, phase_out );
[ ~, o6 ] = quadriga_lib.sos_remove_bias( spatial_freq, phase, pos );
assertEqual( o6, offset );

%% Type handling
p7 = quadriga_lib.sos_remove_bias( single(spatial_freq), single(phase), single(pos) );
assertTrue( isa(p7, 'double') );
assertElementsAlmostEqual( angle( exp( 1j * ( p7 - phase_out ) ) ), zeros(128, 3), 'absolute', 1e-4 );

%% Errors
assert_throws( @() quadriga_lib.sos_remove_bias( spatial_freq, phase ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, [], 1, true, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( spatial_freq, phase, pos ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_remove_bias( spatial_freq, phase(1:5, :), pos ), 'phase' );                 % thrown by C++
assert_throws( @() quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, pos_b(1:5, :) ), ...
    'Number of rows in ''pos_b'' does not match number of rows in ''pos''.' );                                   % thrown by C++
assert_throws( @() quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, [], 2, false ), ...
    'Input ''spatial_freq_scale_b'' other than 1 requires ''pos_b''.' );                                         % thrown by C++
assert_throws( @() quadriga_lib.sos_remove_bias( spatial_freq, phase, pos, pos_b, 2 ), ...
    'Input ''spatial_freq_scale_b'' other than 1 requires ''reciprocal = false''.' );                            % thrown by C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.sos_remove_bias( varargin{:} ); %#ok<ASGLU>
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
