function test_o2i_loss
% MEX interface test for quadriga_lib.o2i_loss.
% Scope: argument and output handling, index conversion, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_o2i_loss.cpp).

%% Setup: 3 UTs, 3 models, 3 frequencies
fc = [ 1e9, 10e9, 30e9 ];
rnd = [ 0.2, 0.5, 0.3, 0.1; 0.7, 0.4, 0.9, -1.2; 0.95, 0.2, 0.6, 0.5 ];
models = { 'low', 'car', 'high' };
weights = [ 0.5, 0.3, 0.2 ];                                                    % UT 1: low, UT 2: car, UT 3: high

%% Basic call and values
[ loss, d2d_in, model_index ] = quadriga_lib.o2i_loss( fc, rnd, models, weights );
assertEqual( size(loss), [3, 3] );
assertEqual( size(d2d_in), [3, 1] );
assertTrue( isa(loss, 'double') && isa(d2d_in, 'double') );
assertEqual( model_index, uint64( [1; 2; 3] ) );                               % 1-based
assertElementsAlmostEqual( d2d_in, [ 0.3 * 25; 0; 0.2 * 25 ], 'absolute', 1e-14 );
assertElementsAlmostEqual( loss(2, :), [3, 3, 3], 'absolute', 1e-14 );          % car: 9 + 5·N, N = -1.2
f = fc / 1e9;                                                                   % low-loss model of UT 1
tw = 5 - 10 * log10( 0.3 * 10.^( -(2 + 0.2 * f) / 10 ) + 0.7 * 10.^( -(5 + 4 * f) / 10 ) );
assertElementsAlmostEqual( loss(1, :), tw + 0.5 * 7.5 + 4.4 * 0.1, 'absolute', 1e-12 );

%% Optional inputs: d2d_in_max
[ loss2, d2d_in2 ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, [] );
assertEqual( loss2, loss );
assertEqual( d2d_in2, d2d_in );
assertEqual( quadriga_lib.o2i_loss( fc, rnd, models, weights, 25 ), loss );
[ loss3, d2d_in3 ] = quadriga_lib.o2i_loss( fc, rnd, models, weights, 10 );
assertElementsAlmostEqual( d2d_in3, [ 3; 0; 2 ], 'absolute', 1e-14 );
assertElementsAlmostEqual( loss3(1, :) - loss(1, :), 0.5 * (3 - 7.5) * [1, 1, 1], 'absolute', 1e-12 );

%% Models as one string or as a cell array; the model index points into 'models'
[ l_c, ~, i_c ] = quadriga_lib.o2i_loss( fc, rnd, 'car', 1 );
[ l_cc, ~, i_cc ] = quadriga_lib.o2i_loss( fc, rnd, { 'car' }, 1 );
assertEqual( l_c, l_cc );
assertEqual( i_c, uint64( [1; 1; 1] ) );
assertEqual( i_cc, i_c );
[ ~, ~, i_r ] = quadriga_lib.o2i_loss( fc, rnd, { 'high', 'car', 'low' }, [ 0.5, 0.3, 0.2 ] );
assertEqual( i_r, uint64( [1; 2; 3] ) );

%% Outputs: none and fewer
quadriga_lib.o2i_loss( fc, rnd, models, weights );
assertEqual( quadriga_lib.o2i_loss( fc, rnd, models, weights ), loss );
[ ~, d2d_in4 ] = quadriga_lib.o2i_loss( fc, rnd, models, weights );
assertEqual( d2d_in4, d2d_in );

%% Type handling and orientation
[ l_t, d_t, i_t ] = quadriga_lib.o2i_loss( single(fc'), single(rnd), models', single(weights'), int32(25) );
assertTrue( isa(l_t, 'double') && isa(i_t, 'uint64') );
assertElementsAlmostEqual( l_t, loss, 'absolute', 1e-5 );                      % rnd rounded to single
assertElementsAlmostEqual( d_t, d2d_in, 'absolute', 1e-5 );
assertEqual( i_t, model_index );

%% Errors
assert_throws( @() quadriga_lib.o2i_loss( fc, rnd, models ), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.o2i_loss( fc, rnd, models, weights, 25, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out4( fc, rnd, models, weights ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.o2i_loss( fc, [ rnd, rnd(:, 1:2) ], models, weights ), ...
    'Input ''rnd'' must have 4 columns.' );                                                         % C++
assert_throws( @() quadriga_lib.o2i_loss( fc, rnd, models, [ 0.5, 0.3, 0.1 ] ), ...
    'Input ''weights'' must sum to 1.' );                                                           % C++
assert_throws( @() quadriga_lib.o2i_loss( fc, rnd, { 'low', 'foo' }, [ 0.5, 0.5 ] ), ...
    'Model ''foo'' is not supported.' );                                                            % C++

end

% Request 4 outputs so that the output-count guard fires
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.o2i_loss( varargin{:} ); %#ok<ASGLU>
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
