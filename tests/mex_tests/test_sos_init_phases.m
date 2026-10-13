function test_sos_init_phases
% MEX interface test for quadriga_lib.sos_init_phases.
% Scope: argument and output handling, defaults, error paths.
% Physics validation lives in the Catch2 suite (test_sos_init_phases.cpp).

%% Basic call
phase = quadriga_lib.sos_init_phases( 7, 3, 42 );
assertEqual( size(phase), [7, 3] );
assertTrue( isa(phase, 'double') );
assertTrue( all(phase(:) >= -pi) && all(phase(:) < pi) );
assertTrue( max(abs(phase(:))) > 1 );
assertTrue( numel(unique(phase(:))) == 21 );

%% A seed is reproducible, the fill order is column by column
phase2 = quadriga_lib.sos_init_phases( 7, 3, 42 );
assertEqual( phase2, phase );
phase3 = quadriga_lib.sos_init_phases( 7, 5, 42 );
assertEqual( phase3(:, 1:3), phase );
phase4 = quadriga_lib.sos_init_phases( 3, 7, 42 );                             % arguments not swapped
assertEqual( size(phase4), [3, 7] );
assertEqual( phase4(:), phase(:) );
phase5 = quadriga_lib.sos_init_phases( 7, 3, 43 );
assertTrue( max(abs(phase5(:) - phase(:))) > 0.1 );

%% Optional inputs: omitted, empty, explicit
p1 = quadriga_lib.sos_init_phases( 9 );                                        % n_gen = 1, random seed
assertEqual( size(p1), [9, 1] );
p2 = quadriga_lib.sos_init_phases( 9, [], [] );
assertEqual( size(p2), [9, 1] );
assertTrue( max(abs(p1 - p2)) > 1e-6 );                                        % seed -1: new values
p3 = quadriga_lib.sos_init_phases( 7, [], 42 );
assertEqual( p3, phase(:, 1) );
p4 = quadriga_lib.sos_init_phases( 7, 3, -1 );
assertTrue( max(abs(p4(:) - phase(:))) > 1e-6 );

%% Outputs: none
quadriga_lib.sos_init_phases( 7, 3, 42 );

%% Type handling
phase6 = quadriga_lib.sos_init_phases( uint32(7), single(3), int32(42) );
assertEqual( phase6, phase );

%% Errors
assert_throws( @() quadriga_lib.sos_init_phases(), 'Wrong number of input arguments.' );
assert_throws( @() quadriga_lib.sos_init_phases( 7, 3, 42, 1 ), 'Wrong number of input arguments.' );
assert_throws( @() out2( 7, 3, 42 ), 'Wrong number of output arguments.' );
assert_throws( @() quadriga_lib.sos_init_phases( 0, 3, 42 ), 'n_coeff' );       % thrown by C++
assert_throws( @() quadriga_lib.sos_init_phases( 7, 0, 42 ), 'n_gen' );         % thrown by C++
assert_throws( @() quadriga_lib.sos_init_phases( 7, 3, -2 ), 'seed' );          % thrown by C++

% Counts and the seed are validated by the wrapper before the conversion to an integer type
assert_throws( @() quadriga_lib.sos_init_phases( -1 ), 'Input ''n_coeff'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_init_phases( NaN ), 'Input ''n_coeff'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_init_phases( [] ), 'Input ''n_coeff'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_init_phases( 7, -1 ), 'Input ''n_gen'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_init_phases( 7, Inf ), 'Input ''n_gen'' must be a finite number >= 1' );
assert_throws( @() quadriga_lib.sos_init_phases( 7, 3, NaN ), 'Input ''seed''' );
assert_throws( @() quadriga_lib.sos_init_phases( 7, 3, Inf ), 'Input ''seed''' );
assert_throws( @() quadriga_lib.sos_init_phases( 7, 3, -Inf ), 'Input ''seed''' );

%% Non-integer scalars are truncated
assertEqual( quadriga_lib.sos_init_phases( 7.9, 3.9, 42.9 ), phase );
p5 = quadriga_lib.sos_init_phases( 7, 3, -1.5 );                               % truncated to -1: random seed
assertTrue( max(abs(p5(:) - phase(:))) > 1e-6 );

end

% Request 2 outputs so that the output-count guard fires
function out2(varargin)
[~, ~] = quadriga_lib.sos_init_phases( varargin{:} ); %#ok<ASGLU>
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
