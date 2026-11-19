% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_fast_sincos
% MEX interface test for quadriga_lib.fast_sincos.
% Physics validation lives in the Catch2 suite (test_fast_sincos.cpp).

% Helper for approximate equality on singles (handles empty inputs gracefully)
assert_close = @(a,b,tol) assertTrue( max([0; abs(single(a(:))-single(b(:)))]) <= tol );

%% Two outputs, single input
x = single(linspace(0,2*pi,1001));
[s,c] = quadriga_lib.fast_sincos(x);
assertTrue(isa(s,'single') && isa(c,'single'));
assertTrue(isequal(size(s),size(x)) && isequal(size(c),size(x)));
assert_close(s, sin(x), 1e-2);
assert_close(c, cos(x), 1e-2);

%% Single output (default = sine), double input accepted, output single
x = linspace(-10*pi,10*pi,2049);
s = quadriga_lib.fast_sincos(x);
assertTrue(isa(s,'single'));
assertTrue(isequal(size(s),size(x)));
assert_close(s, sin(x), 1e-2);

%% Single output cosine via flag
x = single(rand(10,7)*8*pi - 4*pi);
c = quadriga_lib.fast_sincos(x,true);
assertTrue(isa(c,'single'));
assertTrue(isequal(size(c),size(x)));
assert_close(c, cos(x), 1e-2);

%% Single output with flag=false returns sine
x = single(linspace(0,2*pi,257));
s = quadriga_lib.fast_sincos(x,false);
assert_close(s, sin(x), 1e-2);

%% N-D array support
x = single(rand(7,5,3)*4*pi - 2*pi);
[s,c] = quadriga_lib.fast_sincos(x);
assertTrue(isequal(size(s),size(x)) && isequal(size(c),size(x)));
assertTrue(isa(s,'single') && isa(c,'single'));
assert_close(s, sin(x), 1e-2);
assert_close(c, cos(x), 1e-2);

%% Empty input
x = single([]);
[s,c] = quadriga_lib.fast_sincos(x);
assertTrue(isempty(s) && isempty(c));
assertTrue(isa(s,'single') && isa(c,'single'));
s1 = quadriga_lib.fast_sincos(x);
c1 = quadriga_lib.fast_sincos(x,true);
assertTrue(isempty(s1) && isempty(c1));
assertTrue(isa(s1,'single') && isa(c1,'single'));

%% Integer input promoted; outputs remain single
x = int32([-3 -2 -1 0 1 2 3]);
[s,c] = quadriga_lib.fast_sincos(x);
assertTrue(isa(s,'single') && isa(c,'single'));
assertTrue(isequal(size(s),size(x)) && isequal(size(c),size(x)));
assert_close(s, sin(double(x)), 1e-2);
assert_close(c, cos(double(x)), 1e-2);

%% Large magnitude angles (looser tolerance)
x = single(1000*linspace(-10*pi,10*pi,2000));
[s,c] = quadriga_lib.fast_sincos(x);
assert_close(s, sin(x), 2e-2);
assert_close(c, cos(x), 2e-2);

%% Empty cos_only uses the default (sine)
x = [0.1 0.5 1];
s0 = quadriga_lib.fast_sincos(x);
s1 = quadriga_lib.fast_sincos(x, []);
assertTrue(isequal(s1, s0));
assertTrue(isa(s1, 'single'));
assert_close(s1, sin(x), 1e-6);

%% Malformed input is rejected by the MEX helpers
assert_cpp_error( @() quadriga_lib.fast_sincos( sparse([0 0 1.5]) ), 'Input ''x'' is sparse; sparse inputs are not supported.' );
assert_cpp_error( @() quadriga_lib.fast_sincos( [0 0 1.5], sparse(true) ), 'Input ''cos_only'' is sparse' );
assert_cpp_error( @() quadriga_lib.fast_sincos( ones(2,2,2,2,2) ), 'Input ''x'' has more than 4 dimensions.' );
s = quadriga_lib.fast_sincos( ones(2,2,2,2) );                  % 4 dimensions are supported (all 16 values)
assertEqual( numel(s), 16 );

%% use_kernel (input 3) reaches C++: 1 is accepted, 3 is rejected by C++, also for empty input
x = linspace(-100, 100, 101);
[s1, c1] = quadriga_lib.fast_sincos(x, [], 1);
assertEqual( size(s1), size(x) );
assert_cpp_error( @() quadriga_lib.fast_sincos(zeros(0, 1), [], 3), 'Input ''use_kernel'' must be 0, 1 or 2.' );
assert_cpp_error( @() quadriga_lib.fast_sincos(x, [], 3), 'Input ''use_kernel'' must be 0, 1 or 2.' );
end

% Assert that fn() raises quadriga_lib:CPPerror with a message that contains expected (substring).
% One output is requested: without outputs, the wrapper returns before it reads the inputs.
function assert_cpp_error(fn, expected)
try
    [~] = fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if ~strcmp(ME.identifier, 'quadriga_lib:CPPerror') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: quadriga_lib:CPPerror "', expected, '", GOT: ', ...
            ME.identifier, ' "', ME.message, '"']);
    end
end
end
