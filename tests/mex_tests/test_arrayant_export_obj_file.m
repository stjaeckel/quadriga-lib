function test_arrayant_export_obj_file

% --- Basic file creation: .obj and .mtl produced ---
ant = quadriga_lib.arrayant_generate('xpol');  % 2 elements
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant);
assertTrue(exist('test_mex.obj', 'file') == 2);
assertTrue(exist('test_mex.mtl', 'file') == 2);

% --- Read back and validate content ---
[mesh, mtl_prop, vert_list, face_ind, obj_ind, mtl_ind, obj_names] = ...
    quadriga_lib.obj_file_read('test_mex.obj');

assert( size(mesh, 1) > 0 );
assert( size(mesh, 2) == 9 );                          % {X1,Y1,Z1,X2,Y2,Z2,X3,Y3,Z3} per triangle
assert( size(vert_list, 2) == 3 );                     % xyz columns
assert( size(face_ind, 1) == size(mesh, 1) );          % one face per triangle
assert( size(face_ind, 2) == 3 );                      % 3 vertex indices per face
assert( numel(obj_ind) == size(mesh, 1) );
assert( numel(mtl_ind) == size(mesh, 1) );
assert( size(mtl_prop, 1) == size(mesh, 1) );
assert( size(mtl_prop, 2) == 9 );
% Face indices reference valid vertices
assert( max(face_ind(:)) <= size(vert_list, 1) );
assert( min(face_ind(:)) >= 1 );
% xpol has 2 elements -> expect 2 objects
assert( max(obj_ind) == 2 );
assert( numel(obj_names) == 2 );

delete('test_mex.obj'); delete('test_mex.mtl');

% --- icosphere_n_div: higher subdivision -> roughly 4x more triangles per step ---
ant_c = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0.1);  % 1 element

quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 1.0, 1);
mesh_low = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 1.0, 3);
mesh_high = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

% n_div=1 -> 80 faces/icosphere, n_div=3 -> 1280 -> ratio 16x (allow slack)
assert( size(mesh_high, 1), 180 );
assert( size(mesh_low, 1), 20 );

% --- object_radius scales the mesh linearly ---
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 1.0, 2);
[~, ~, vert_r1] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 3.0, 2);
[~, ~, vert_r3] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

max_r1 = max(abs(vert_r1(:)));
max_r3 = max(abs(vert_r3(:)));
assertElementsAlmostEqual( max_r3 / max_r1, 3.0, 'relative', 0.05 );

% --- i_element: all elements vs single element ---
ant_xp = quadriga_lib.arrayant_generate('xpol');  % 2 elements

quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_xp, 30, 'jet', 1.0, 2, []);
[~, ~, ~, ~, obj_ind_all] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_xp, 30, 'jet', 1.0, 2, 1);
[~, ~, ~, ~, obj_ind_one] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

assert( max(obj_ind_all) == 2 );
assert( max(obj_ind_one) == 1 );
assert( numel(obj_ind_one) < numel(obj_ind_all) );

% --- Colormap parameter: every supported colormap loads without error ---
for cmap = {'jet', 'parula', 'winter', 'hot', 'turbo', 'copper', 'spring', 'cool', 'gray', 'autumn', 'summer'}
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, cmap{1}, 1.0, 2);
    assertTrue(exist('test_mex.obj', 'file') == 2);
    delete('test_mex.obj'); delete('test_mex.mtl');
end

% --- directivity_range: different range values, file still valid ---
for drange = [10, 30, 50]
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, drange, 'jet', 1.0, 2);
    assertTrue(exist('test_mex.obj', 'file') == 2);
    delete('test_mex.obj'); delete('test_mex.mtl');
end

% --- Return value: optional success indicator (= 1) ---
ok = quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c);
assertElementsAlmostEqual( ok, 1.0, 'absolute', 1e-12 );
delete('test_mex.obj'); delete('test_mex.mtl');

% --- Multi-frequency dispatch via 'freq' input ---
ant_a = quadriga_lib.arrayant_generate('custom', [], [], 30, 20, 0);
ant_b = quadriga_lib.arrayant_generate('custom', [], [], 90, 60, 0);
ant_mf = [ant_a, ant_b];

% freq=1 -> uses ant_a entry
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_mf, 30, 'jet', 1.0, 2, [], 1);
[~, ~, vert_freq1] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

% freq=2 -> uses ant_b entry
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_mf, 30, 'jet', 1.0, 2, [], 2);
[~, ~, vert_freq2] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

% Geometries differ because beam patterns differ
assert( ~isequal(vert_freq1, vert_freq2) );

% Each multi-freq export matches its direct single-freq counterpart
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_a, 30, 'jet', 1.0, 2);
[~, ~, vert_a] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_b, 30, 'jet', 1.0, 2);
[~, ~, vert_b] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');

assertElementsAlmostEqual( vert_freq1, vert_a, 'absolute', 1e-10 );
assertElementsAlmostEqual( vert_freq2, vert_b, 'absolute', 1e-10 );

% Omitted freq defaults to 1 (matches explicit freq=1)
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_mf, 30, 'jet', 1.0, 2);
[~, ~, vert_default] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');
assertElementsAlmostEqual( vert_default, vert_freq1, 'absolute', 1e-12 );

% Empty freq input also defaults to 1
quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_mf, 30, 'jet', 1.0, 2, [], []);
[~, ~, vert_empty] = quadriga_lib.obj_file_read('test_mex.obj');
delete('test_mex.obj'); delete('test_mex.mtl');
assertElementsAlmostEqual( vert_empty, vert_freq1, 'absolute', 1e-12 );

% --- Errors ---

% Too few input arguments (filename only)
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many input arguments
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 1.0, 2, [], 1, 'extra');
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many output arguments
try
    [~, ~] = quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Non-struct arrayant input
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', 1.0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'must be a struct';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% i_element = 0 (1-based violation)
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 1.0, 2, 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'cannot be 0';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% freq = 0 (out of bound, must be >= 1)
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_mf, 30, 'jet', 1.0, 2, [], 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'out of bound';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% freq > n_freq (out of bound for 2-entry struct array)
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_mf, 30, 'jet', 1.0, 2, [], 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'out of bound';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% freq = 2 for single-frequency input (n_freq = 1)
try
    quadriga_lib.arrayant_export_obj_file('test_mex.obj', ant_c, 30, 'jet', 1.0, 2, [], 2);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'out of bound';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% --- Cleanup any leftovers ---
if exist('test_mex.obj', 'file'), delete('test_mex.obj'); end
if exist('test_mex.mtl', 'file'), delete('test_mex.mtl'); end

end
