function test_arrayant_combine_pattern

% ==== Single-frequency tests ====

azimuth_grid    = [-1.5,0,1.5,2] * pi/2;
elevation_grid  = [-0.9,0,0.9] * pi/2;
e_theta_re      = ones(3,4,2);
e_theta_im      = ones(3,4,2)*2;
e_phi_re        = ones(3,4,2)*3;
e_phi_im        = ones(3,4,2)*4;
element_pos     = zeros(3,2);
coupling_re     = [1;1];
coupling_im     = [0;0];
center_freq     = 2e9;

ant_in = struct('e_theta_re', e_theta_re, 'e_theta_im', e_theta_im, 'e_phi_re', e_phi_re, 'e_phi_im', e_phi_im, ...
   'azimuth_grid', azimuth_grid, 'elevation_grid', elevation_grid, 'element_pos', element_pos, ...
   'coupling_re', coupling_re, 'coupling_im', coupling_im);

% Separate inputs, combined output
ant = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq, 'bla');

assertElementsAlmostEqual( ant.e_theta_re, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_theta_im, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_re, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_im, ones(3,4)*8, 'absolute', 1e-14 );
assertEqual( ant.center_freq, center_freq );
assertEqual( ant.name, 'bla' );

% Combined inputs, combined output
ant = quadriga_lib.arrayant_combine_pattern( ant_in, 3e9 );
assertElementsAlmostEqual( ant.e_theta_re, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_theta_im, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_re, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( ant.e_phi_im, ones(3,4)*8, 'absolute', 1e-14 );
assertEqual( ant.center_freq, 3e9 );

% Combined inputs, separate output
[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_combine_pattern( ant_in, 3e9 );
assertElementsAlmostEqual( A, ones(3,4)*2, 'absolute', 1e-14 );
assertElementsAlmostEqual( B, ones(3,4)*4, 'absolute', 1e-14 );
assertElementsAlmostEqual( C, ones(3,4)*6, 'absolute', 1e-14 );
assertElementsAlmostEqual( D, ones(3,4)*8, 'absolute', 1e-14 );
assertEqual( J, 3e9 );

e_theta_re      = ones(3,4);
e_theta_im      = zeros(3,4);
e_phi_re        = ones(3,4);
e_phi_im        = zeros(3,4);
element_pos     = ones(3,1);
coupling_re     = 1;
coupling_im     = 0;
center_freq     = 2e9;

[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im, center_freq);

assertTrue( A(1,1) ~= 0 );
assertTrue( B(1,1) ~= 0 );
assertTrue( C(1,1) ~= 0 );
assertTrue( D(1,1) ~= 0 );

[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.arrayant_combine_pattern( [], [], [], [],  e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
    azimuth_grid, elevation_grid, [0;0;0], 1, 0);

assertElementsAlmostEqual( A, e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( B, e_theta_im, 'absolute', 1e-14 );
assertElementsAlmostEqual( C, e_phi_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( D, e_phi_im, 'absolute', 1e-14 );

% Grid inerpolation
ant = quadriga_lib.arrayant_generate('3gpp', 10, [], [], [], [], 2, [], [], [], 0.0);
ant.e_theta_re(:,:,2) = 2 * ant.e_theta_re(:,:,2);
ant.coupling_re = [1;1];
ant.coupling_im = [0;0];

out = quadriga_lib.arrayant_combine_pattern(ant);
assertElementsAlmostEqual( out.e_theta_re, 3*ant.e_theta_re(:,:,1), 'absolute', 1e-14 );

az_grid = linspace(-pi, pi, 73);
out = quadriga_lib.arrayant_combine_pattern(ant, 3.0e9, az_grid);
assertElementsAlmostEqual( out.e_theta_re(:,1:2:end), 3*ant.e_theta_re(:,:,1), 'absolute', 1e-14 );

el_grid = linspace(-pi/2, pi/2, 37);
out = quadriga_lib.arrayant_combine_pattern(ant, 3.0e9, [], el_grid);
assertElementsAlmostEqual( out.e_theta_re(1:2:end,:), 3*ant.e_theta_re(:,:,1), 'absolute', 1e-14 );

directivity_ant = quadriga_lib.arrayant_calc_directivity(ant,1);
directivity_out = quadriga_lib.arrayant_calc_directivity(out);
assertElementsAlmostEqual( directivity_ant, directivity_out, 'absolute', 0.1 );

ant.element_pos(2,:) = [-0.25, 0.25];
out = quadriga_lib.arrayant_combine_pattern(ant);
directivity_out = quadriga_lib.arrayant_calc_directivity(out);
assertTrue(directivity_out > directivity_ant + 1.2);

% Errors
try
    out = quadriga_lib.arrayant_combine_pattern( ant, [],[],[],[]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~] = quadriga_lib.arrayant_combine_pattern( ant );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re(:,2:end), e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, coupling_re, coupling_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sizes of ''e_theta_re'' and ''e_theta_im'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid(1:2), elevation_grid, element_pos, coupling_re, coupling_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''azimuth_grid'' does not match number of columns in pattern data.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, 1, coupling_re, coupling_im);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Size of ''element_pos'' must be either empty or match [3, n_elements]';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, [1;1], [0;0]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''Coupling'' must be a matrix with rows equal to number of elements';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, [], 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Imaginary part of coupling matrix (phase component) defined without real part (absolute component)';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    out = quadriga_lib.arrayant_combine_pattern( [], [], [], [], e_theta_re, e_theta_im, e_phi_re, e_phi_im, ...
        azimuth_grid, elevation_grid, element_pos, 1, [1,1]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''coupling_im'' must be empty or its size must match ''coupling_re''';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end


% ==== Multi-frequency tests ====

% Build a 3-entry multi-freq struct array (scale=1,2,3 at 1/2/3 GHz)
freqs_mf = [1e9, 2e9, 3e9];
clear ant_mf;
for i = 1:length(freqs_mf)
    ant_mf(i).e_theta_re      = ones(3,4) * i;
    ant_mf(i).e_theta_im      = zeros(3,4);
    ant_mf(i).e_phi_re        = zeros(3,4);
    ant_mf(i).e_phi_im        = zeros(3,4);
    ant_mf(i).azimuth_grid    = azimuth_grid;
    ant_mf(i).elevation_grid  = elevation_grid;
    ant_mf(i).element_pos     = zeros(3,1);
    ant_mf(i).coupling_re     = 1;
    ant_mf(i).coupling_im     = 0;
    ant_mf(i).center_freq     = freqs_mf(i);
end

% Multi-freq input → struct array output (default = 1:1 with input)
out = quadriga_lib.arrayant_combine_pattern(ant_mf);
assertEqual( length(out), 3 );
for i = 1:3
    assertElementsAlmostEqual( out(i).e_theta_re, ones(3,4) * i, 'absolute', 1e-12 );
    assertEqual( out(i).center_freq, freqs_mf(i) );
end

% Multi-freq input + scalar freq → single struct output (exact match)
out = quadriga_lib.arrayant_combine_pattern(ant_mf, 2e9);
assertEqual( length(out), 1 );
assertTrue( isstruct(out) );
assertElementsAlmostEqual( out.e_theta_re, ones(3,4) * 2, 'absolute', 1e-12 );
assertEqual( out.center_freq, 2e9 );

% Multi-freq input + vector freq → struct array output (interpolated)
out = quadriga_lib.arrayant_combine_pattern(ant_mf, [1.5e9, 2.5e9]);
assertEqual( length(out), 2 );
% SLERP of in-phase reals collapses to linear blend → midpoints = 1.5 and 2.5
assertElementsAlmostEqual( out(1).e_theta_re, ones(3,4) * 1.5, 'absolute', 1e-10 );
assertElementsAlmostEqual( out(2).e_theta_re, ones(3,4) * 2.5, 'absolute', 1e-10 );
assertEqual( out(1).center_freq, 1.5e9 );
assertEqual( out(2).center_freq, 2.5e9 );

% Single struct input + vector freq → struct array output (all clamped to that entry)
out = quadriga_lib.arrayant_combine_pattern(ant_mf(2), [1e9, 2e9, 3e9]);
assertEqual( length(out), 3 );
for i = 1:3
    assertElementsAlmostEqual( out(i).e_theta_re, ones(3,4) * 2, 'absolute', 1e-12 );
end
assertEqual( out(1).center_freq, 1e9 );
assertEqual( out(2).center_freq, 2e9 );
assertEqual( out(3).center_freq, 3e9 );

% Clamping at boundaries
out = quadriga_lib.arrayant_combine_pattern(ant_mf, [0.5e9, 10e9]);
assertEqual( length(out), 2 );
assertElementsAlmostEqual( out(1).e_theta_re, ones(3,4) * 1, 'absolute', 1e-12 );
assertElementsAlmostEqual( out(2).e_theta_re, ones(3,4) * 3, 'absolute', 1e-12 );

% Default output preserves input order even when center_freqs are unsorted
clear ant_unsorted;
for i = 1:3
    ant_unsorted(i).e_theta_re      = ones(3,4) * i;
    ant_unsorted(i).e_theta_im      = zeros(3,4);
    ant_unsorted(i).e_phi_re        = zeros(3,4);
    ant_unsorted(i).e_phi_im        = zeros(3,4);
    ant_unsorted(i).azimuth_grid    = azimuth_grid;
    ant_unsorted(i).elevation_grid  = elevation_grid;
    ant_unsorted(i).element_pos     = zeros(3,1);
    ant_unsorted(i).coupling_re     = 1;
    ant_unsorted(i).coupling_im     = 0;
end
ant_unsorted(1).center_freq = 2e9;
ant_unsorted(2).center_freq = 1e9;
ant_unsorted(3).center_freq = 3e9;
out = quadriga_lib.arrayant_combine_pattern(ant_unsorted);
assertEqual( length(out), 3 );
assertEqual( out(1).center_freq, 2e9 );
assertEqual( out(2).center_freq, 1e9 );
assertEqual( out(3).center_freq, 3e9 );

% Unsorted freq_grid_new is allowed; outputs match positionally
out = quadriga_lib.arrayant_combine_pattern(ant_mf, [3e9, 1e9, 2e9]);
assertEqual( length(out), 3 );
assertElementsAlmostEqual( out(1).e_theta_re, ones(3,4) * 3, 'absolute', 1e-12 );
assertElementsAlmostEqual( out(2).e_theta_re, ones(3,4) * 1, 'absolute', 1e-12 );
assertElementsAlmostEqual( out(3).e_theta_re, ones(3,4) * 2, 'absolute', 1e-12 );

% Custom azimuth/elevation grids with multi-freq input
az_new = linspace(-pi, pi, 9);
el_new = linspace(-pi/2, pi/2, 5);
out = quadriga_lib.arrayant_combine_pattern(ant_mf, [], az_new, el_new);
assertEqual( length(out), 3 );
for i = 1:3
    assertEqual( length(out(i).azimuth_grid), 9 );
    assertEqual( length(out(i).elevation_grid), 5 );
    assertEqual( size(out(i).e_theta_re, 1), 5 );
    assertEqual( size(out(i).e_theta_re, 2), 9 );
end

% Coupling SLERP across frequencies (constant pattern, varying coupling)
clear ant_cpl;
for i = 1:2
    ant_cpl(i).e_theta_re      = ones(3,4);
    ant_cpl(i).e_theta_im      = zeros(3,4);
    ant_cpl(i).e_phi_re        = zeros(3,4);
    ant_cpl(i).e_phi_im        = zeros(3,4);
    ant_cpl(i).azimuth_grid    = azimuth_grid;
    ant_cpl(i).elevation_grid  = elevation_grid;
    ant_cpl(i).element_pos     = zeros(3,1);
    ant_cpl(i).coupling_im     = 0;
end
ant_cpl(1).coupling_re = 1; ant_cpl(1).center_freq = 1e9;
ant_cpl(2).coupling_re = 3; ant_cpl(2).center_freq = 2e9;
% Midpoint: coupling SLERP of in-phase reals = 2; pattern = 1 → result = 2
out = quadriga_lib.arrayant_combine_pattern(ant_cpl, 1.5e9);
assertElementsAlmostEqual( out.e_theta_re, ones(3,4) * 2, 'absolute', 1e-10 );

% Polarimetric V and H both propagated through multi-freq combine
clear ant_pol;
for i = 1:2
    ant_pol(i).e_theta_re      = ones(3,4) * i;
    ant_pol(i).e_theta_im      = zeros(3,4);
    ant_pol(i).e_phi_re        = ones(3,4) * (-i);
    ant_pol(i).e_phi_im        = zeros(3,4);
    ant_pol(i).azimuth_grid    = azimuth_grid;
    ant_pol(i).elevation_grid  = elevation_grid;
    ant_pol(i).element_pos     = zeros(3,1);
    ant_pol(i).coupling_re     = 1;
    ant_pol(i).coupling_im     = 0;
end
ant_pol(1).center_freq = 1e9;
ant_pol(2).center_freq = 2e9;
out = quadriga_lib.arrayant_combine_pattern(ant_pol);
assertEqual( length(out), 2 );
assertElementsAlmostEqual( out(1).e_theta_re, ones(3,4) * 1,    'absolute', 1e-12 );
assertElementsAlmostEqual( out(1).e_phi_re,   ones(3,4) * (-1), 'absolute', 1e-12 );
assertElementsAlmostEqual( out(2).e_theta_re, ones(3,4) * 2,    'absolute', 1e-12 );
assertElementsAlmostEqual( out(2).e_phi_re,   ones(3,4) * (-2), 'absolute', 1e-12 );

% Multi-element antenna with non-trivial coupling, multi-freq
% 2 elements summed by coupling [1;1] → 1 port → 1 element output
clear ant_2e;
for i = 1:2
    ant_2e(i).e_theta_re      = ones(3,4,2) * i;   % 2 elements
    ant_2e(i).e_theta_im      = zeros(3,4,2);
    ant_2e(i).e_phi_re        = zeros(3,4,2);
    ant_2e(i).e_phi_im        = zeros(3,4,2);
    ant_2e(i).azimuth_grid    = azimuth_grid;
    ant_2e(i).elevation_grid  = elevation_grid;
    ant_2e(i).element_pos     = zeros(3,2);
    ant_2e(i).coupling_re     = [1;1];
    ant_2e(i).coupling_im     = [0;0];
end
ant_2e(1).center_freq = 1e9;
ant_2e(2).center_freq = 2e9;
out = quadriga_lib.arrayant_combine_pattern(ant_2e);
assertEqual( length(out), 2 );
% Each entry: elem0+elem1 = 2*scale → 2 for entry 1, 4 for entry 2
assertElementsAlmostEqual( out(1).e_theta_re, ones(3,4) * 2, 'absolute', 1e-12 );
assertElementsAlmostEqual( out(2).e_theta_re, ones(3,4) * 4, 'absolute', 1e-12 );
% Output coupling is identity, output has 1 port → coupling is [1x1]
assertEqual( size(out(1).coupling_re, 1), 1 );
assertEqual( size(out(1).coupling_re, 2), 1 );

% Element-position-induced gain at multi-freq → 2 elements with offset → directivity gain
clear ant_dir;
ant_3gpp = quadriga_lib.arrayant_generate('3gpp', 10, [], [], [], [], 2, [], [], [], 0.0);
ant_3gpp.coupling_re = [1;1];
ant_3gpp.coupling_im = [0;0];
ant_3gpp.element_pos(2,:) = [-0.25, 0.25];
for i = 1:2
    ant_dir(i) = ant_3gpp;
    ant_dir(i).center_freq = (i==1)*2e9 + (i==2)*3e9;
end
out = quadriga_lib.arrayant_combine_pattern(ant_dir);
assertEqual( length(out), 2 );
d0 = quadriga_lib.arrayant_calc_directivity(ant_3gpp, 1);
d1 = quadriga_lib.arrayant_calc_directivity(out(1));
d2 = quadriga_lib.arrayant_calc_directivity(out(2));
assertTrue( d1 > d0 + 1.2 );
assertTrue( d2 > d0 + 1.2 );

% ==== Multi-frequency error tests ====

% Multi-freq output cannot be returned through 11 separate outputs
try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_combine_pattern(ant_mf);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Multi-frequency output supports only struct output';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Same error path triggered by single-struct + vector freq + 11 LHS
try
    [~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_combine_pattern(ant_mf(1), [1e9, 2e9]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Multi-frequency output supports only struct output';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Separate inputs + vector freq is rejected
e_theta_re_se = ones(3,4);
e_theta_im_se = zeros(3,4);
e_phi_re_se   = ones(3,4);
e_phi_im_se   = zeros(3,4);
try
    out = quadriga_lib.arrayant_combine_pattern( [], [1e9, 2e9], [], [], ...
        e_theta_re_se, e_theta_im_se, e_phi_re_se, e_phi_im_se, ...
        azimuth_grid, elevation_grid, [0;0;0], 1, 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Multi-frequency mode (vector freq) requires struct input';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Invalid multi-freq input (mismatched azimuth grids across entries) caught by validation
clear ant_bad;
for i = 1:2
    ant_bad(i).e_theta_re      = ones(3,4);
    ant_bad(i).e_theta_im      = zeros(3,4);
    ant_bad(i).e_phi_re        = zeros(3,4);
    ant_bad(i).e_phi_im        = zeros(3,4);
    ant_bad(i).elevation_grid  = elevation_grid;
    ant_bad(i).element_pos     = zeros(3,1);
    ant_bad(i).coupling_re     = 1;
    ant_bad(i).coupling_im     = 0;
    ant_bad(i).center_freq     = (i==1)*1e9 + (i==2)*2e9;
end
ant_bad(1).azimuth_grid = azimuth_grid;
ant_bad(2).azimuth_grid = [-pi/2, 0, pi/2];   % wrong length / content
try
    out = quadriga_lib.arrayant_combine_pattern(ant_bad);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected the C++ multi-freq validator to throw.');
    end
end

end
