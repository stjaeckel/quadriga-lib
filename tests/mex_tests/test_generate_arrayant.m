function test_generate_arrayant

quadriga_lib.generate_arrayant('omni', 10);

% Omni antenna, 10 deg
data = quadriga_lib.generate_arrayant('omni', 10);

assertElementsAlmostEqual( data.e_theta_re, ones(19,37), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_im, zeros(19, 37), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_re, zeros(19, 37), 'absolute', 1e-14)
assertElementsAlmostEqual( data.e_phi_im, zeros(19, 37), 'absolute', 1e-14)
assertElementsAlmostEqual( data.center_freq, 299792458.0, 'absolute', 1e-14)
assertElementsAlmostEqual( data.azimuth_grid, linspace(-pi, pi, 37), 'absolute', 1e-14)
assertElementsAlmostEqual( data.elevation_grid, linspace(-pi/2, pi/2, 19), 'absolute', 1e-14)
assertElementsAlmostEqual( data.coupling_re, 1.0, 'absolute', 1e-14)
assertElementsAlmostEqual( data.coupling_im, 0.0, 'absolute', 1e-14)
assertTrue( strcmp(data.name,'omni') );

% Xpol antenna
data = quadriga_lib.generate_arrayant('xpol', 400.0, 3.0e9);

assertElementsAlmostEqual( data.e_theta_re(:,:,1), ones(3, 5), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_re(:,:,2), zeros(3, 5), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_re(:,:,2), ones(3, 5), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_re(:,:,1), zeros(3, 5), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_im, zeros(3, 5, 2), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_im, zeros(3, 5, 2), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.center_freq, 3.0e9, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.azimuth_grid, linspace(-pi, pi, 5), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.elevation_grid, linspace(-pi/2, pi/2, 3), 'absolute', 1e-14 );
assertTrue( strcmp(data.name, 'xpol') );

% Custom antenna
data = quadriga_lib.generate_arrayant('custom', 10, [], 90, 90, 0);
assert( isequal(size(data.e_theta_re), [19, 37]) );
assertTrue( strcmp(data.name, 'custom') );

% 3GPP default
data = quadriga_lib.generate_arrayant('3gpp');
assert( isequal(size(data.e_theta_re), [181, 361]) );
assertElementsAlmostEqual( data.e_theta_re(1,1,1), 0.0794328, 'absolute', 1e-7 );
assertElementsAlmostEqual( data.e_theta_re(91,181,1), 2.51188, 'absolute', 1e-5 );
assertTrue( strcmp(data.name, '3gpp') );

[A,B,C,D,E,F,G,H,I,J,K] = quadriga_lib.generate_arrayant('3gpp');
assertElementsAlmostEqual( data.e_theta_re, A, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_im, B, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_re, C, 'absolute', 1e-14)
assertElementsAlmostEqual( data.e_phi_im, D, 'absolute', 1e-14)
assertElementsAlmostEqual( data.azimuth_grid, E, 'absolute', 1e-14)
assertElementsAlmostEqual( data.elevation_grid, F, 'absolute', 1e-14)
assertElementsAlmostEqual( data.element_pos, G, 'absolute', 1e-14)
assertElementsAlmostEqual( data.coupling_re, H, 'absolute', 1e-14)
assertElementsAlmostEqual( data.coupling_im, I, 'absolute', 1e-14)
assertElementsAlmostEqual( data.center_freq, J, 'absolute', 1e-14)
assertTrue( strcmp(K, '3gpp') );

data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], [], [], 2);
assertElementsAlmostEqual( data.e_theta_re(1,1,1), 0.0794328, 'absolute', 1e-7 );
assertElementsAlmostEqual( data.e_theta_re(91,181,1), 2.51188, 'absolute', 1e-5 );
assertElementsAlmostEqual( data.e_phi_re(1,1,2), -0.0794328, 'absolute', 1e-7 );
assertElementsAlmostEqual( data.e_phi_re(91,181,2), 2.51188, 'absolute', 1e-5 );

data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], [], [], 3);
assertElementsAlmostEqual( data.e_theta_re(91,181,1), 1.776172, 'absolute', 1e-5 );
assertElementsAlmostEqual( data.e_theta_re(91,181,2), 1.776172, 'absolute', 1e-5 );
assertElementsAlmostEqual( data.e_phi_re(91,181,1), 1.776172, 'absolute', 1e-5 );
assertElementsAlmostEqual( data.e_phi_re(91,181,2), -1.776172, 'absolute', 1e-5 );

% 3GPP custom
pat = quadriga_lib.generate_arrayant('custom', 10);

data = quadriga_lib.generate_arrayant('3gpp', 10, [], 90, 90, 0, [], 2);
assertElementsAlmostEqual( data.e_theta_re(:,:,1), pat.e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_re(:,:,2), pat.e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.element_pos, [0,0 ; -0.25,0.25 ; 0,0], 'absolute', 1e-14 );

data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], 2, 1, [], [], [], [], [], [], [], pat);
assertElementsAlmostEqual( data.e_theta_re(:,:,1), pat.e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_re(:,:,2), pat.e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.element_pos, [0,0 ; 0,0 ; -0.25,0.25], 'absolute', 1e-14 );

pat.e_theta_re = single(pat.e_theta_re);
pat.e_theta_im = uint32(pat.e_theta_im);

data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], 2, 1, [], [], [], [], [], [], [], pat);
assertElementsAlmostEqual( data.e_theta_re(:,:,1), pat.e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_theta_re(:,:,2), pat.e_theta_re, 'absolute', 1e-14 );
assertElementsAlmostEqual( data.element_pos, [0,0 ; 0,0 ; -0.25,0.25], 'absolute', 1e-14 );

% Errors
try 
    [~,~] = quadriga_lib.generate_arrayant('omni');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

trash = struct('e_theta_re', 0.0);
try 
    data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], 2, 1, [], [], [], [], [], [], [], trash);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Field ''e_theta_im'' not found at index 0!';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

trash = struct('e_theta_re', 'Bla');
try 
    data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], 2, 1, [], [], [], [], [], [], [], trash);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Unsupported data type.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try 
    data = quadriga_lib.generate_arrayant('3gpp', [], [], [], [], [], 2, 1, [], [], [], [], [], [], [], 0.1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input must be a struct.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try 
    data = quadriga_lib.generate_arrayant('bla');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Array type not supported!';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- ULA: default (pol=1, v-pol isotropic) ---
data = quadriga_lib.generate_arrayant('ula', 10, [], [], [], [], [], 4);
assert( isequal(size(data.e_theta_re), [19, 37, 4]) );
assertElementsAlmostEqual( data.e_theta_re, ones(19, 37, 4), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_re, zeros(19, 37, 4), 'absolute', 1e-14 );
% Horizontal spacing along y-axis, centered at 0 (adjust if your ULA convention differs)
assertElementsAlmostEqual( data.element_pos(1,:), zeros(1,4), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.element_pos(3,:), zeros(1,4), 'absolute', 1e-14 );
assertTrue( strcmp(data.name, 'ula') );

% --- ULA: pol=2 (H/V) ---
data = quadriga_lib.generate_arrayant('ula', 10, [], [], [], [], [], 2, 2);
assert( isequal(size(data.e_theta_re), [19, 37, 4]) ); % 2N elements for pol=2
% element 1 is V, element 2 is H after rotation
assertElementsAlmostEqual( data.e_theta_re(:,:,1), ones(19,37), 'absolute', 1e-14 );
assertElementsAlmostEqual( data.e_phi_re(:,:,2),   ones(19,37), 'absolute', 1e-14 );

% --- ULA: M should be ignored ---
data = quadriga_lib.generate_arrayant('ula', 10, [], [], [], [], 5, 3); % M=5 should be ignored
assert( size(data.e_theta_re, 3) == 3 ); % only N=3 elements

% --- multibeam: single beam at boresight ---
data = quadriga_lib.generate_arrayant('multibeam', 10, [], [], [], [], 1, 4, 1, [0;0;1]);
assert( isequal(size(data.e_theta_re), [19, 37, 4]) );

% --- multibeam_sep: two separate beams ---
beams = [-30, 30; 0, 0; 1, 1];  % two beams at ±30° azimuth
data = quadriga_lib.generate_arrayant('multibeam_sep', 10, [], [], [], [], 1, 4, 1, beams);
% With separate_beams=true, expect n_ports > 1 in coupling matrix
assert( size(data.coupling_re, 2) == 2 );

% --- multibeam: beam_angles with only 1 row should error ---
try
    bad = [0];  % only azimuth, no elevation
    data = quadriga_lib.generate_arrayant('multibeam', 10, [], [], [], [], 1, 4, 1, bad);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''beam_angles'' must have at least 2 rows';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- custom: default beamwidth (tests the 90° substitution fix) ---
data_default = quadriga_lib.generate_arrayant('custom', 10);
data_explicit = quadriga_lib.generate_arrayant('custom', 10, [], 90, 90, 0);
assertElementsAlmostEqual( data_default.e_theta_re, data_explicit.e_theta_re, 'absolute', 1e-14 );

% --- no-input error ---
try
    data = quadriga_lib.generate_arrayant();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Arrayant type name missing.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end