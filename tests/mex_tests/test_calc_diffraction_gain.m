function test_calc_diffraction_gain

cube = [  -1     1     1   ,    1    -1     1   ,    1     1     1;   %  1 Top NorthEast
           1    -1     1   ,   -1    -1    -1   ,    1    -1    -1;   %  2 South Lower
          -1    -1     1   ,   -1     1    -1   ,   -1    -1    -1;   %  3 West Lower
           1     1    -1   ,   -1    -1    -1   ,   -1     1    -1;   %  4 Bottom NorthWest
           1     1     1   ,    1    -1    -1   ,    1     1    -1;   %  5 East Lower
          -1     1     1   ,    1     1    -1   ,   -1     1    -1;   %  6 North Lower
          -1     1     1   ,   -1    -1     1   ,    1    -1     1;   %  7 Top SouthWest
           1    -1     1   ,   -1    -1     1   ,   -1    -1    -1;   %  8 South Upper
          -1    -1     1   ,   -1     1     1   ,   -1     1    -1;   %  9 West Upper
           1     1    -1   ,    1    -1    -1   ,   -1    -1    -1;   % 10 Bottom SouthEast
           1     1     1   ,    1    -1     1   ,    1    -1    -1;   % 11 East Upper
          -1     1     1   ,    1     1     1   ,    1     1    -1 ]; % 12 North Upper

mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 3.0],12,1);

orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];    % FBS West Upper (9), SBS East Upper (11)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];    % FBS East Lower Top (5), SBS West Lower (3)

%% Basic diffraction gain, lod = 0
gain = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0 );
assertElementsAlmostEqual( gain, [10^(-0.3);10^(-0.3)], 'absolute', 1e-14 );

%% 0 outputs should be fine
quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9 );

%% 1 output only
gain_only = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 2 );
assertTrue( numel(gain_only) == 2 );

%% 2 outputs, lod = 5
[gain5, coord5] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 5 );
assertElementsAlmostEqual( gain5, [10^(-0.3);10^(-0.3)], 'absolute', 1e-14 );
assertElementsAlmostEqual( coord5, permute([0,0 ; 0,0 ; 0.5,-0.5],[1,3,2]), 'absolute', 1e-14 );

%% LOS (unobstructed) path: TX and RX above the cube, gain should be ~1.0
orig_los = [ 0, 0, 5 ];
dest_los = [ 0, 0, 10 ];
gain_los = quadriga_lib.calc_diffraction_gain( orig_los, dest_los, cube, mtl_prop, 1e9, 2 );
assertElementsAlmostEqual( gain_los, 1.0, 'absolute', 1e-6 );

%% Single-precision inputs should work (cast to double internally)
gain_single = quadriga_lib.calc_diffraction_gain( single(orig), single(dest), ...
    single(cube), single(mtl_prop), 1e9, 2 );
assertElementsAlmostEqual( gain_only, gain_single, 'absolute', 1e-5 );

%% Empty sub-mesh index
[~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, [] );
[~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, uint32(0) );

%% sub_mesh_index with non-uint32 numeric (typecast path in new wrapper)
gain_smi = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, int32(0) );
assertElementsAlmostEqual( gain, gain_smi, 'absolute', 1e-14 );

%% use_kernel = 1 (GENERIC), should match default
gain_generic = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 2, 0, [], 1 );
assertElementsAlmostEqual( gain_only, gain_generic, 'absolute', 1e-14 );

%% use_kernel = 1 with gpu_id = 0 (all 10 args)
gain_full = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 2, 0, [], 1, 0 );
assertElementsAlmostEqual( gain_only, gain_full, 'absolute', 1e-14 );

%% Verify coord dimensions for each lod value
[~, c1] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 1 );
assertTrue( isequal(size(c1), [3, 2, 2]) );   % lod 1 -> n_seg=2

[~, c2] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 2 );
assertTrue( isequal(size(c2), [3, 2, 2]) );   % lod 2 -> n_seg=2

[~, c3] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 3 );
assertTrue( isequal(size(c3), [3, 3, 2]) );   % lod 3 -> n_seg=3

[~, c4] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 4 );
assertTrue( isequal(size(c4), [3, 4, 2]) );   % lod 4 -> n_seg=4

[~, c6] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 6 );
assertTrue( isequal(size(c6), [3, 1, 2]) );   % lod 6 -> n_seg=1

%% 5 args (explicit center_freq)
gain5a = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 2e9 );

%% Error: too few inputs (4 args)
try
    [~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: too many inputs (11 args)
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, [], 0, 0, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: 3 outputs
try
    [~, ~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 5 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: wrong dest size
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest(1,:), cube, mtl_prop, 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: wrong mtl_prop columns
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop(:,1), 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mtl_prop'' must have 5 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: wrong mtl_prop rows
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop(1,:), 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''mesh'' and ''mtl_prop'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: sub_mesh_index first element not 0
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, uint32(1));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-mesh must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

%% Error: sub_mesh_index exceeds mesh count
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, uint32([0,32]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-mesh indices cannot exceed number of faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end