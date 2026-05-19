function test_arrayant_concat

% --- Single frequency: basic append (xpol + xpol = 4 elements) ---
ant1 = quadriga_lib.arrayant_generate('xpol', 1);  % 2 elements
ant2 = quadriga_lib.arrayant_generate('xpol', 1);  % 2 elements
ant_out = quadriga_lib.arrayant_concat(ant1, ant2);

assert( size(ant_out.e_theta_re, 3) == 4 );
% First 2 slices from ant1
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant1.e_theta_re(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,2), ant1.e_theta_re(:,:,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_im(:,:,1), ant1.e_theta_im(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,1),   ant1.e_phi_re(:,:,1),   'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_im(:,:,1),   ant1.e_phi_im(:,:,1),   'absolute', 1e-12 );
% Last 2 slices from ant2
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,3), ant2.e_theta_re(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,4), ant2.e_theta_re(:,:,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_im(:,:,3), ant2.e_theta_im(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,3),   ant2.e_phi_re(:,:,1),   'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_im(:,:,3),   ant2.e_phi_im(:,:,1),   'absolute', 1e-12 );
% Grids preserved
assertElementsAlmostEqual( ant_out.azimuth_grid,   ant1.azimuth_grid,   'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.elevation_grid, ant1.elevation_grid, 'absolute', 1e-12 );

% --- Coupling is block-diagonal ---
assert( all(size(ant_out.coupling_re) == [4, 4]) );
assert( all(size(ant_out.coupling_im) == [4, 4]) );
% Top-left block from ant1
assertElementsAlmostEqual( ant_out.coupling_re(1:2,1:2), ant1.coupling_re, 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_im(1:2,1:2), ant1.coupling_im, 'absolute', 1e-12 );
% Bottom-right block from ant2
assertElementsAlmostEqual( ant_out.coupling_re(3:4,3:4), ant2.coupling_re, 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_im(3:4,3:4), ant2.coupling_im, 'absolute', 1e-12 );
% Off-diagonal blocks zero
assertElementsAlmostEqual( ant_out.coupling_re(1:2,3:4), zeros(2,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_re(3:4,1:2), zeros(2,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_im(1:2,3:4), zeros(2,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_im(3:4,1:2), zeros(2,2), 'absolute', 1e-12 );

% --- element_pos concatenation ---
ant1_p = ant1;
ant1_p.element_pos = [0.1, 0.2; 0.3, 0.4; 0.5, 0.6];
ant2_p = ant2;
ant2_p.element_pos = [0.7, 0.8; 0.9, 1.0; 1.1, 1.2];
ant_out = quadriga_lib.arrayant_concat(ant1_p, ant2_p);
assert( size(ant_out.element_pos, 2) == 4 );
expected_pos = [0.1, 0.2, 0.7, 0.8; ...
                0.3, 0.4, 0.9, 1.0; ...
                0.5, 0.6, 1.1, 1.2];
assertElementsAlmostEqual( ant_out.element_pos, expected_pos, 'absolute', 1e-12 );

% --- center_freq inherited from ant1 ---
ant1_f = ant1;  ant1_f.center_freq = 2.4e9;
ant2_f = ant2;  ant2_f.center_freq = 5.8e9;  % different on purpose
ant_out = quadriga_lib.arrayant_concat(ant1_f, ant2_f);
assertElementsAlmostEqual( ant_out.center_freq, 2.4e9, 'absolute', 1e-3 );

% --- name inherited from ant1 ---
ant1_n = ant1;  ant1_n.name = 'first';
ant2_n = ant2;  ant2_n.name = 'second';
ant_out = quadriga_lib.arrayant_concat(ant1_n, ant2_n);
assert( strcmp(ant_out.name, 'first') );

% --- Asymmetric element counts: 1 + 2 = 3 ---
ant_c = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0);  % 1 element
ant_out = quadriga_lib.arrayant_concat(ant_c, ant1);
assert( size(ant_out.e_theta_re, 3) == 3 );
assert( all(size(ant_out.coupling_re) == [3, 3]) );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant_c.e_theta_re(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,2), ant1.e_theta_re(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,3), ant1.e_theta_re(:,:,2), 'absolute', 1e-12 );
% Block-diagonal coupling: 1x1 block then 2x2 block
assertElementsAlmostEqual( ant_out.coupling_re(2:3,2:3), ant1.coupling_re, 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_re(1,2:3),   zeros(1,2),       'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_re(2:3,1),   zeros(2,1),       'absolute', 1e-12 );

% --- Multi-frequency: basic 2-entry concat ---
ant_mf1 = [ant1, ant1];
ant_mf2 = [ant2, ant2];
ant_mf_out = quadriga_lib.arrayant_concat(ant_mf1, ant_mf2);
assert( numel(ant_mf_out) == 2 );
for k = 1:2
    assert( size(ant_mf_out(k).e_theta_re, 3) == 4 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,1), ant1.e_theta_re(:,:,1), 'absolute', 1e-12 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,3), ant2.e_theta_re(:,:,1), 'absolute', 1e-12 );
    assert( all(size(ant_mf_out(k).coupling_re) == [4, 4]) );
    assertElementsAlmostEqual( ant_mf_out(k).coupling_re(1:2,3:4), zeros(2,2), 'absolute', 1e-12 );
end

% --- Multi-frequency with distinct content per entry ---
ant_a1 = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0);
ant_a2 = quadriga_lib.arrayant_generate('custom', [], [], 60, 40, 0);
ant_b1 = quadriga_lib.arrayant_generate('custom', [], [], 90, 50, 0);
ant_b2 = quadriga_lib.arrayant_generate('custom', [], [], 45, 30, 0);
ant_mf_a = [ant_a1, ant_a2];
ant_mf_b = [ant_b1, ant_b2];
ant_mf_out = quadriga_lib.arrayant_concat(ant_mf_a, ant_mf_b);
assert( numel(ant_mf_out) == 2 );
% Entry 1 = ant_a1 ++ ant_b1
assertElementsAlmostEqual( ant_mf_out(1).e_theta_re(:,:,1), ant_a1.e_theta_re(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_mf_out(1).e_theta_re(:,:,2), ant_b1.e_theta_re(:,:,1), 'absolute', 1e-12 );
% Entry 2 = ant_a2 ++ ant_b2
assertElementsAlmostEqual( ant_mf_out(2).e_theta_re(:,:,1), ant_a2.e_theta_re(:,:,1), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_mf_out(2).e_theta_re(:,:,2), ant_b2.e_theta_re(:,:,1), 'absolute', 1e-12 );
% Entries differ from each other
assert( ~isequal(ant_mf_out(1).e_theta_re(:,:,1), ant_mf_out(2).e_theta_re(:,:,1)) );

% --- Errors ---

% Too few input arguments
try
    quadriga_lib.arrayant_concat( ant1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many input arguments
try
    quadriga_lib.arrayant_concat( ant1, ant2, ant1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many output arguments
try
    [~, ~] = quadriga_lib.arrayant_concat( ant1, ant2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Non-struct input for ant1
try
    quadriga_lib.arrayant_concat( 1.0, ant2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'must be a struct';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Non-struct input for ant2
try
    quadriga_lib.arrayant_concat( ant1, 1.0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'must be a struct';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Empty struct array (0 elements)
try
    empty_ant = ant1(1:0);  % 1x0 struct array
    quadriga_lib.arrayant_concat( empty_ant, ant2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'cannot be empty';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Mismatched frequency entry count (scalar vs 2-element multi)
try
    quadriga_lib.arrayant_concat( ant1, [ant2, ant2] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'same number of entries';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Single-freq: mismatched azimuth grid (value mismatch)
try
    ant_bad = ant2;
    ant_bad.azimuth_grid = ant_bad.azimuth_grid + 0.001;  % shift values, same size
    quadriga_lib.arrayant_concat( ant1, ant_bad );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Values of ''azimuth_grid'' must be between -pi and pi';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Single-freq: mismatched elevation grid (value mismatch)
try
    ant_bad = ant2;
    ant_bad.elevation_grid = ant_bad.elevation_grid + 0.001;
    quadriga_lib.arrayant_concat( ant1, ant_bad );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Values of ''elevation_grid'' must be between -pi/2 and pi/2'; 
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Multi-freq: azimuth grid size mismatch
try
    ant_bad = ant2;
    ant_bad.azimuth_grid = ant_bad.azimuth_grid(1:end-1);
    ant_bad.e_theta_re  = ant_bad.e_theta_re(:,1:end-1,:);
    ant_bad.e_theta_im  = ant_bad.e_theta_im(:,1:end-1,:);
    ant_bad.e_phi_re    = ant_bad.e_phi_re(:,1:end-1,:);
    ant_bad.e_phi_im    = ant_bad.e_phi_im(:,1:end-1,:);
    quadriga_lib.arrayant_concat( [ant1, ant1], [ant_bad, ant_bad] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Azimuth grid sizes do not match';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Multi-freq: elevation grid size mismatch
try
    ant_bad = ant2;
    ant_bad.elevation_grid = ant_bad.elevation_grid(1:end-1);
    ant_bad.e_theta_re  = ant_bad.e_theta_re(1:end-1,:,:);
    ant_bad.e_theta_im  = ant_bad.e_theta_im(1:end-1,:,:);
    ant_bad.e_phi_re    = ant_bad.e_phi_re(1:end-1,:,:);
    ant_bad.e_phi_im    = ant_bad.e_phi_im(1:end-1,:,:);
    quadriga_lib.arrayant_concat( [ant1, ant1], [ant_bad, ant_bad] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Elevation grid sizes'; 
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Multi-freq: azimuth grid value mismatch (same size, shifted values)
try
    ant_bad = ant2;
    ant_bad.azimuth_grid = ant_bad.azimuth_grid + 0.001;
    quadriga_lib.arrayant_concat( [ant1, ant1], [ant_bad, ant_bad] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Second input invalid: Entry 0: Values of ''azimuth_grid'' must be between -pi and pi';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Multi-freq: elevation grid value mismatch (same size, shifted values)
try
    ant_bad = ant2;
    ant_bad.elevation_grid = ant_bad.elevation_grid + 0.001;
    quadriga_lib.arrayant_concat( [ant1, ant1], [ant_bad, ant_bad] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Second input invalid: Entry 0: Values of ''elevation_grid'' must be between -pi/2 and pi/2';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end