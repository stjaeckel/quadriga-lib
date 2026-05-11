function test_arrayant_copy_element

% --- Single frequency: in-place copy (overwrite existing element) ---
ant = quadriga_lib.arrayant_generate('xpol');  % 2 elements: V-pol and H-pol
% Sanity: input elements actually differ
assert( ~isequal(ant.e_theta_re(:,:,1), ant.e_theta_re(:,:,2)) || ...
        ~isequal(ant.e_phi_re(:,:,1),   ant.e_phi_re(:,:,2)) );

ant_out = quadriga_lib.arrayant_copy_element(ant, 1, 2);
assert( size(ant_out.e_theta_re, 3) == 2 );  % unchanged element count
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant_out.e_theta_re(:,:,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_im(:,:,1), ant_out.e_theta_im(:,:,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,1),   ant_out.e_phi_re(:,:,2),   'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_im(:,:,1),   ant_out.e_phi_im(:,:,2),   'absolute', 1e-12 );

% --- Single frequency: resize via copy to a new slot ---
ant_out = quadriga_lib.arrayant_copy_element(ant, 1, 3);
assert( size(ant_out.e_theta_re, 3) == 3 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant_out.e_theta_re(:,:,3), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,1),   ant_out.e_phi_re(:,:,3),   'absolute', 1e-12 );
% Element 2 untouched
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,2), ant.e_theta_re(:,:,2), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,2),   ant.e_phi_re(:,:,2),   'absolute', 1e-12 );
% Coupling extended; new diagonal entry = 1, new off-diagonals = 0
assert( all(size(ant_out.coupling_re) == [3, 3]) );
assert( all(size(ant_out.coupling_im) == [3, 3]) );
assertElementsAlmostEqual( ant_out.coupling_re(3,3), 1.0, 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_re(1,3), 0.0, 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_re(3,1), 0.0, 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.coupling_im(3,3), 0.0, 'absolute', 1e-12 );
% element_pos extended to 3 columns
assert( size(ant_out.element_pos, 2) == 3 );

% --- Single frequency: copy one source to multiple destinations (vector dest) ---
ant_out = quadriga_lib.arrayant_copy_element(ant, 1, [3, 4]);
assert( size(ant_out.e_theta_re, 3) == 4 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant_out.e_theta_re(:,:,3), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant_out.e_theta_re(:,:,4), 'absolute', 1e-12 );
% Element 2 untouched
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,2), ant.e_theta_re(:,:,2), 'absolute', 1e-12 );

% --- Single frequency: pairwise (source vector + dest vector, equal length) ---
ant_out = quadriga_lib.arrayant_copy_element(ant, [1, 2], [3, 4]);
assert( size(ant_out.e_theta_re, 3) == 4 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,1), ant_out.e_theta_re(:,:,3), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,1),   ant_out.e_phi_re(:,:,3),   'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_theta_re(:,:,2), ant_out.e_theta_re(:,:,4), 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.e_phi_re(:,:,2),   ant_out.e_phi_re(:,:,4),   'absolute', 1e-12 );

% --- Existing element_pos preserved on resize ---
ant_p = ant;
ant_p.element_pos = [0.1, 0.2; 0.3, 0.4; 0.5, 0.6];
ant_out = quadriga_lib.arrayant_copy_element(ant_p, 1, 3);
assert( size(ant_out.element_pos, 2) == 3 );
assertElementsAlmostEqual( ant_out.element_pos(:,1), [0.1; 0.3; 0.5], 'absolute', 1e-12 );
assertElementsAlmostEqual( ant_out.element_pos(:,2), [0.2; 0.4; 0.6], 'absolute', 1e-12 );

% --- Multi-frequency struct array: basic resize ---
ant_mf = [ant, ant];
ant_mf_out = quadriga_lib.arrayant_copy_element(ant_mf, 1, 3);
assert( numel(ant_mf_out) == 2 );
for k = 1:2
    assert( size(ant_mf_out(k).e_theta_re, 3) == 3 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,1), ant_mf_out(k).e_theta_re(:,:,3), 'absolute', 1e-12 );
    assertElementsAlmostEqual( ant_mf_out(k).e_phi_re(:,:,1),   ant_mf_out(k).e_phi_re(:,:,3),   'absolute', 1e-12 );
    assertElementsAlmostEqual( ant_mf_out(k).coupling_re(3,3), 1.0, 'absolute', 1e-12 );
end

% --- Multi-frequency: copy one source to multiple destinations ---
ant_mf_out = quadriga_lib.arrayant_copy_element(ant_mf, 1, [3, 4]);
assert( numel(ant_mf_out) == 2 );
for k = 1:2
    assert( size(ant_mf_out(k).e_theta_re, 3) == 4 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,1), ant_mf_out(k).e_theta_re(:,:,3), 'absolute', 1e-12 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,1), ant_mf_out(k).e_theta_re(:,:,4), 'absolute', 1e-12 );
end

% --- Multi-frequency: pairwise ---
ant_mf_out = quadriga_lib.arrayant_copy_element(ant_mf, [1, 2], [3, 4]);
assert( numel(ant_mf_out) == 2 );
for k = 1:2
    assert( size(ant_mf_out(k).e_theta_re, 3) == 4 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,1), ant_mf_out(k).e_theta_re(:,:,3), 'absolute', 1e-12 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,2), ant_mf_out(k).e_theta_re(:,:,4), 'absolute', 1e-12 );
end

% --- Multi-frequency with distinct entries: each modified independently ---
ant_a = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0);
ant_b = quadriga_lib.arrayant_generate('custom', [], [], 60, 40, 0);
ant_mf_diff = [ant_a, ant_b];
ant_mf_out = quadriga_lib.arrayant_copy_element(ant_mf_diff, 1, 2);
assert( numel(ant_mf_out) == 2 );
for k = 1:2
    assert( size(ant_mf_out(k).e_theta_re, 3) == 2 );
    assertElementsAlmostEqual( ant_mf_out(k).e_theta_re(:,:,1), ant_mf_out(k).e_theta_re(:,:,2), 'absolute', 1e-12 );
end
% Entries still differ from each other (source patterns were different)
assert( ~isequal(ant_mf_out(1).e_theta_re(:,:,1), ant_mf_out(2).e_theta_re(:,:,1)) );

% --- Errors ---

% Too few input arguments
try
    quadriga_lib.arrayant_copy_element( ant, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many input arguments
try
    quadriga_lib.arrayant_copy_element( ant, 1, 2, 3 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many output arguments
try
    [~, ~] = quadriga_lib.arrayant_copy_element( ant, 1, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Non-struct input
try
    quadriga_lib.arrayant_copy_element( 1.0, 1, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'must be a struct';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% source_element = 0 (1-based violation)
try
    quadriga_lib.arrayant_copy_element( ant, 0, 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'cannot be 0';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% dest_element = 0 (1-based violation)
try
    quadriga_lib.arrayant_copy_element( ant, 1, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'cannot be 0';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Empty source_element
try
    quadriga_lib.arrayant_copy_element( ant, [], 2 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'cannot be empty';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Empty dest_element
try
    quadriga_lib.arrayant_copy_element( ant, 1, [] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'cannot be empty';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Mismatched source/dest vector lengths
try
    quadriga_lib.arrayant_copy_element( ant, [1, 2], [3, 4, 5] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'same length';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Source index out of bound (only 2 elements in ant)
try
    quadriga_lib.arrayant_copy_element( ant, 5, 3 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    % Underlying C++ error; just verify an error was raised
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected an error for out-of-bound source index!');
    end
end

end
