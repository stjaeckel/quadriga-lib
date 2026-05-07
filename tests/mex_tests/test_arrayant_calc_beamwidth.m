function test_arrayant_calc_beamwidth

% --- Short dipole: known elevation 3dB beamwidth = 90° (cos^2(el) shape) ---
ant = quadriga_lib.arrayant_generate('dipole');
[bw_az, bw_el, ~, el_pt] = quadriga_lib.arrayant_calc_beamwidth(ant);
assertElementsAlmostEqual( bw_az, 360.0, 'absolute', 0.05 );  % omni in azimuth
assertElementsAlmostEqual( bw_el,  90.0, 'absolute', 0.5  );  % cos^2 -> ±45°
assertElementsAlmostEqual( el_pt,   0.0, 'absolute', 0.5  );  % broadside

% --- Custom antenna recovers its input 3dB beamwidth ---
ant_c = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0);
[bw_az, bw_el, az_pt, el_pt] = quadriga_lib.arrayant_calc_beamwidth(ant_c);
assertElementsAlmostEqual( bw_az, 30.0, 'absolute', 0.5 );
assertElementsAlmostEqual( bw_el, 20.0, 'absolute', 0.5 );
assertElementsAlmostEqual( az_pt,  0.0, 'absolute', 0.1 );
assertElementsAlmostEqual( el_pt,  0.0, 'absolute', 0.1 );

% --- Threshold parameter: 6 dB clearly wider than 3 dB ---
bw3 = quadriga_lib.arrayant_calc_beamwidth(ant_c, [], 3);
bw6 = quadriga_lib.arrayant_calc_beamwidth(ant_c, [], 6);
assert( bw6 > bw3 + 5 );

% Empty threshold falls back to default (= 3 dB)
bw3b = quadriga_lib.arrayant_calc_beamwidth(ant_c, [], []);
assertElementsAlmostEqual( bw3, bw3b, 'absolute', 1e-9 );

% --- Multi-element antenna: per-element pointing via rotation ---
% Build 2 elements pointing at +45° and -45° in azimuth
ant2 = ant_c;
ant2.e_theta_re = repmat(ant_c.e_theta_re, [1 1 2]);
ant2.e_theta_im = repmat(ant_c.e_theta_im, [1 1 2]);
ant2.e_phi_re   = repmat(ant_c.e_phi_re,   [1 1 2]);
ant2.e_phi_im   = repmat(ant_c.e_phi_im,   [1 1 2]);
ant2.element_pos = zeros(3, 2);
ant2.coupling_re = eye(2);
ant2.coupling_im = zeros(2, 2);
ant2 = quadriga_lib.arrayant_rotate_pattern(ant2, 0, 0,  45, 0, 1);
ant2 = quadriga_lib.arrayant_rotate_pattern(ant2, 0, 0, -45, 0, 2);

[~, ~, az_pt] = quadriga_lib.arrayant_calc_beamwidth(ant2);
assertElementsAlmostEqual( az_pt, [45; -45], 'absolute', 0.5 );

% Subset selection picks only the requested element
[~, ~, az_pt] = quadriga_lib.arrayant_calc_beamwidth(ant2, 2);
assertElementsAlmostEqual( az_pt, -45, 'absolute', 0.5 );

% --- Split mode (separate inputs) ---
[A,B,C,D,E,F,~,~,~,~,~] = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0);
[bw_az, bw_el] = quadriga_lib.arrayant_calc_beamwidth(A,B,C,D,E,F);
assertElementsAlmostEqual( bw_az, 30.0, 'absolute', 0.5 );
assertElementsAlmostEqual( bw_el, 20.0, 'absolute', 0.5 );

% Split mode with i_element and threshold
bw_az = quadriga_lib.arrayant_calc_beamwidth(A,B,C,D,E,F, 1, 3);
assertElementsAlmostEqual( bw_az, 30.0, 'absolute', 0.5 );

% --- xpol: 2 cross-polarized isotropic elements -> full grid for both ---
ant_xp = quadriga_lib.arrayant_generate('xpol');
[bw_az, bw_el] = quadriga_lib.arrayant_calc_beamwidth(ant_xp);
assertElementsAlmostEqual( bw_az, [360; 360], 'absolute', 0.05 );
assertElementsAlmostEqual( bw_el, [180; 180], 'absolute', 0.05 );

% Empty i_element -> all elements (struct mode)
bw_az = quadriga_lib.arrayant_calc_beamwidth(ant_xp, []);
assertElementsAlmostEqual( bw_az, [360; 360], 'absolute', 0.05 );

% Empty i_element -> all elements (split mode)
[A2,B2,C2,D2,E2,F2,~,~,~,~,~] = quadriga_lib.arrayant_generate('xpol');
bw_az = quadriga_lib.arrayant_calc_beamwidth(A2,B2,C2,D2,E2,F2, []);
assertElementsAlmostEqual( bw_az, [360; 360], 'absolute', 0.05 );

% --- Multi-frequency struct array (n_freq = 2) ---
ant_mf = [ant_c, ant_c];
[bw_az, bw_el] = quadriga_lib.arrayant_calc_beamwidth(ant_mf);
assertElementsAlmostEqual( bw_az, [30, 30], 'absolute', 0.5 );
assertElementsAlmostEqual( bw_el, [20, 20], 'absolute', 0.5 );

% Multi-frequency with i_element selection
bw_az = quadriga_lib.arrayant_calc_beamwidth(ant_mf, [1, 1]);
assertElementsAlmostEqual( bw_az, [30, 30; 30, 30], 'absolute', 0.5 );

% --- Selective output requests (nlhs variants) ---
bw_az_only = quadriga_lib.arrayant_calc_beamwidth(ant_c);
assertElementsAlmostEqual( bw_az_only, 30.0, 'absolute', 0.5 );

[~, ~, ~, el_pt_only] = quadriga_lib.arrayant_calc_beamwidth(ant_c);
assertElementsAlmostEqual( el_pt_only, 0.0, 'absolute', 0.5 );

% --- Errors ---

% Non-struct input with too few args for split mode
try
    quadriga_lib.arrayant_calc_beamwidth( A, B );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input must be a struct.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Wrong number of input arguments (nrhs = 5 invalid)
try
    quadriga_lib.arrayant_calc_beamwidth( A, B, C, D, E );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Element index out of bound
try
    [~] = quadriga_lib.arrayant_calc_beamwidth( ant_c, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Element index out of bound.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many output arguments (nlhs > 4)
try
    [~,~,~,~,~] = quadriga_lib.arrayant_calc_beamwidth( ant_c );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% i_element = 0 (1-based violation)
try
    [~] = quadriga_lib.arrayant_calc_beamwidth( ant_c, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = '''i_element'' cannot be 0';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Struct input mixed with split-mode extras
try
    [~] = quadriga_lib.arrayant_calc_beamwidth( ant_c, B, C, D, E, F );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Cannot mix struct input with separate arrayant inputs.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
