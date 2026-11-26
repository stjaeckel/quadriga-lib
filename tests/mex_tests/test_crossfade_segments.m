% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_crossfade_segments
% MEX interface test for quadriga_lib.crossfade_segments.
% Scope: argument and output handling, the segments as a struct array, the tap indices as 'par', error paths.
% The cross-fade rules, the tap forms and the edge cases are tested in the Catch2 suite (test_crossfade_segments.cpp).

%% Setup: two segments of 4 snapshots with one path each (rays form), amplitude 1 and 2; the last 2 snapshots of segment 1 are the first 2 of segment 2 and form the ramp (w = 0.25, 0.75); levels 6 dB apart, so the power is 1^(1 - w) 4^w
a = segment( 1, 4, 0 );
b = segment( 2, 4, 2 );
seg = [ a, b ];
n_overlap = [ 0, 2 ];
par = struct( 'tap', { uint32( 0 ), uint32( 0 ) } );                           % both paths are LOS paths (tap index 0)

%% Basic call: one LOS path with |h|^2 = 4^w in the ramp
out = quadriga_lib.crossfade_segments( seg, n_overlap, 'rays', par );
assertEqual( numel( out ), 1 );
assertEqual( size( out.coeff_re ), [ 1, 1, 1, 6 ] );                            % 4 + (4 - 2) snapshots
amp = squeeze( abs( out.coeff_re + 1i * out.coeff_im ) )';
assertElementsAlmostEqual( amp, [ 1, 1, 4^0.125, 4^0.375, 2, 2 ], 'absolute', 1e-12 );
assertElementsAlmostEqual( out.path_gain, [ 1, 1, 4^0.25, 4^0.75, 4, 4 ], 'absolute', 1e-12 );
assertEqual( size( out.rx_pos ), [ 3, 6 ] );
assertEqual( out.name, 'seg' );
assertEqual( out.center_freq, 3.5e9 );

%% Optional inputs: output (default 'taps') and par (without it both paths are NLOS clusters, side by side in the ramp)
out = quadriga_lib.crossfade_segments( seg, n_overlap, [], par );
assertTrue( ~isfield( out, 'path_gain' ) || isempty( out.path_gain ) );
amp = squeeze( abs( out.coeff_re + 1i * out.coeff_im ) )';
assertElementsAlmostEqual( amp, [ 1, 1, 4^0.125, 4^0.375, 2, 2 ], 'absolute', 1e-12 );
assertEqual( quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', par ), out );
out = quadriga_lib.crossfade_segments( seg, n_overlap );
assertEqual( size( out.coeff_re ), [ 1, 1, 2, 6 ] );
power = squeeze( abs( out.coeff_re + 1i * out.coeff_im ).^2 );               % [path, snapshot]
assertElementsAlmostEqual( sum( power, 1 ), [ 1, 1, 4^0.25, 4^0.75, 4, 4 ], 'absolute', 1e-12 );   % the same total power as one path
out = quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', [] );
assertEqual( size( out.coeff_re, 3 ), 2 );

%% path_type: a single path of tap index 3 is a GR path for the tap rule (merged); path_type 2 makes it NLOS
out = quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', struct( 'tap', { uint32( 3 ), uint32( 3 ) } ) );
assertEqual( size( out.coeff_re, 3 ), 1 );
out = quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', struct( 'tap', { uint32( 3 ), uint32( 3 ) }, 'path_type', { uint32( 2 ), uint32( 2 ) } ) );
assertEqual( size( out.coeff_re, 3 ), 2 );
out = quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', struct( 'path_type', { 0, 0 } ) );   % path_type alone: both LOS
assertEqual( size( out.coeff_re, 3 ), 1 );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', struct( 'tap', { 0, 0 }, 'path_type', { 5, 2 } ) ), 'path_type' );   % C++

%% Outputs: none
quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', par );
assertEqual( size( ans.coeff_re ), [ 1, 1, 1, 6 ] );

%% Type handling: uint8 and column n_overlap, tap indices as double, a cluster field
out = quadriga_lib.crossfade_segments( seg, uint8( [ 0; 2 ] ), 'rays', struct( 'tap', { 0, 0 }, 'cluster', { 0, 0 } ) );
amp = squeeze( abs( out.coeff_re + 1i * out.coeff_im ) )';
assertElementsAlmostEqual( amp, [ 1, 1, 4^0.125, 4^0.375, 2, 2 ], 'absolute', 1e-12 );

%% n_overlap omitted: the 2 shared snapshots follow from the Rx positions; ends that do not move need n_overlap
assertEqual( quadriga_lib.crossfade_segments( seg ), quadriga_lib.crossfade_segments( seg, n_overlap ) );
assertEqual( quadriga_lib.crossfade_segments( seg, [], 'taps', par ), quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', par ) );
static = [ segment( 1, 4, 0 ), segment( 2, 4, 0 ) ];
static(2).rx_pos = static(1).rx_pos;
static(1).rx_pos = repmat( [ 0; 0; 1.5 ], 1, 4 );
static(2).rx_pos = static(1).rx_pos;
assert_throws( @() quadriga_lib.crossfade_segments( static ), 'give ''n_overlap''' );

%% Errors
assert_throws( @() quadriga_lib.crossfade_segments(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', par, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( seg, n_overlap ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.crossfade_segments( 1, n_overlap ), 'Input ''segments'' must be a struct.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', 5 ), 'Input ''par'' must be a struct.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', struct( 'tapp', { 0, 0 } ) ), 'Input ''par'' has no field ''tap'', ''path_type'' or ''cluster''.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 'taps', par(1) ), 'Inputs ''par'' and ''segments'' must have the same number of entries.' );
assert_throws( @() quadriga_lib.crossfade_segments( seg, [ 1, 2 ] ), 'n_overlap' );                                    % C++
assert_throws( @() quadriga_lib.crossfade_segments( seg, n_overlap, 'paths' ), 'Input ''output'' must be ''taps'' or ''rays''.' );   % C++

end

% A channel of n_snap snapshots with one path of the given amplitude and path gain amplitude^2, Rx at x = x0, x0 + 1, ...
function s = segment(amp, n_snap, x0)
s.name = 'seg';
s.tx_pos = [ 0; 0; 10 ];
s.rx_pos = [ x0 + ( 0 : n_snap - 1 ); zeros( 1, n_snap ); 1.5 * ones( 1, n_snap ) ];
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
