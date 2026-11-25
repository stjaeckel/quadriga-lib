% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_crossfade_segments
% MEX interface test for quadriga_lib.crossfade_segments.
% Scope: argument and output handling, the segments as a struct array, the tap indices as 'par', error paths.
% The cross-fade rules, the tap forms and the edge cases are tested in the Catch2 suite (test_crossfade_segments.cpp).

%% Setup: two segments of 4 snapshots with one path each (rays form), amplitude 1 and 2; the last 2 snapshots of segment 1 are the first 2 of segment 2
a = segment( 1, 4 );
b = segment( 2, 4 );
seg = [ a, b ];
n_overlap = [ 0, 2 ];
par = struct( 'tap', { uint32( 0 ), uint32( 0 ) } );                           % both paths are LOS paths (tap index 0)

%% Basic call: LOS path with |h|^2 = (1 - w) |h_old|^2 + w |h_new|^2 in the ramp (w = 0.25, 0.75 for 2 ramp snapshots)
out = quadriga_lib.crossfade_segments( seg, n_overlap, 1, par );
assertEqual( numel( out ), 1 );
assertEqual( size( out.coeff_re ), [ 1, 1, 1, 6 ] );                            % 4 + (4 - 2) snapshots
amp = squeeze( abs( out.coeff_re + 1i * out.coeff_im ) )';
assertElementsAlmostEqual( amp, [ 1, 1, sqrt( 1.75 ), sqrt( 3.25 ), 2, 2 ], 'absolute', 1e-12 );
assertEqual( size( out.rx_pos ), [ 3, 6 ] );
assertEqual( out.name, 'seg' );
assertEqual( out.center_freq, 3.5e9 );

%% Optional inputs: overlap (default 0.5: one ramp snapshot, w = 0.5) and par (without tap indices all paths are NLOS)
out = quadriga_lib.crossfade_segments( seg, n_overlap, [], par );
amp = squeeze( abs( out.coeff_re + 1i * out.coeff_im ) )';
assertElementsAlmostEqual( amp, [ 1, 1, 1, sqrt( 0.5 * 1 + 0.5 * 4 ), 2, 2 ], 'absolute', 1e-12 );
assertEqual( quadriga_lib.crossfade_segments( seg, n_overlap, 0.5, par ), out );
out = quadriga_lib.crossfade_segments( seg, n_overlap, 1 );                     % NLOS paths of both segments side by side
assertEqual( size( out.coeff_re ), [ 1, 1, 2, 6 ] );
power = squeeze( abs( out.coeff_re + 1i * out.coeff_im ).^2 );               % [path, snapshot]
assertElementsAlmostEqual( sum( power, 1 ), [ 1, 1, 1.75, 3.25, 4, 4 ], 'absolute', 1e-12 );   % the same total power as the LOS rule
out = quadriga_lib.crossfade_segments( seg, n_overlap, 1, [] );
assertEqual( size( out.coeff_re, 3 ), 2 );

%% Outputs: none
quadriga_lib.crossfade_segments( seg, n_overlap, 1, par );
assertEqual( size( ans.coeff_re ), [ 1, 1, 1, 6 ] );

%% Type handling: uint8 and column n_overlap, tap indices as double, single overlap
out = quadriga_lib.crossfade_segments( seg, uint8( [ 0; 2 ] ), single( 1 ), struct( 'tap', { 0, 0 } ) );
amp = squeeze( abs( out.coeff_re + 1i * out.coeff_im ) )';
assertElementsAlmostEqual( amp, [ 1, 1, sqrt( 1.75 ), sqrt( 3.25 ), 2, 2 ], 'absolute', 1e-12 );

%% Errors
assert_throws( @() quadriga_lib.crossfade_segments( seg ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 1, par, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( seg, n_overlap ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.crossfade_segments( 1, n_overlap ), 'Input ''segments'' must be a struct.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 1, 5 ), 'Input ''par'' must be a struct.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 1, struct( 'tapp', { 0, 0 } ) ), 'Input ''par'' has no field ''tap''.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 1, par(1) ), 'Inputs ''par'' and ''segments'' must have the same number of entries.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, [ 1, 2 ] ), 'n_overlap' );                                    % C++
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 2 ), 'overlap' );                                  % C++

end

% A channel of n_snap snapshots with one path of the given amplitude and path gain amplitude^2
function s = segment(amp, n_snap)
s.name = 'seg';
s.tx_pos = [ 0; 0; 10 ];
s.rx_pos = repmat( [ 0; 0; 1.5 ], 1, n_snap );
s.coeff_re = reshape( amp * ones( 1, n_snap ), 1, 1, 1, n_snap );
s.coeff_im = zeros( 1, 1, 1, n_snap );
s.delay = 1e-7 * ones( 1, 1, 1, n_snap );
s.path_gain = amp^2 * ones( 1, n_snap );
s.center_freq = 3.5e9;
end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.crossfade_segments( varargin{:} ); %#ok<ASGLU>
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
