function test_ray_triangle_intersect

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


orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];    % FBS West Upper (9), SBS East Upper (11)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];    % FBS East Lower Top (5), SBS West Lower (3)
orig(3,:) = [ -10.0,  0.0,   2.0 ]; dest(3,:) = [  10.0,  0.0,   2.0];    % Miss
orig(4,:) = [   0.5,  0.0,  10.0 ]; dest(4,:) = [  0.5,   0.0, -10.0];    % FBS Top NorthEast (1), SBS Bottom SouthEast (10)
orig(5,:) = [   0.0, 10.0,  -0.5 ]; dest(5,:) = [  0.0, -10.0,  -0.5];    % FBS North Lower (6), SBS South Lower (2)
orig(6,:) = [0,0,0]; dest(6,:) = [0,0,0];                                 % Edge case : co-located

[ fbs, sbs, no_hit, ifbs, isbs ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube );

% ray 1
assertElementsAlmostEqual( fbs(1,:), [-1, 0, 0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs(1,:), [ 1, 0, 0.5], 'absolute', 1e-6 );
assertEqual( no_hit(1), uint32(2) );
assertEqual( ifbs(1), uint32(9) );
assertEqual( isbs(1), uint32(11) );

% ray 2
assertElementsAlmostEqual( fbs(2,:), [ 1, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs(2,:), [-1, 0, -0.5], 'absolute', 1e-6 );
assertEqual( no_hit(2), uint32(2) );
assertEqual( ifbs(2), uint32(5) );
assertEqual( isbs(2), uint32(3) );

% ray 3 (miss)
assertElementsAlmostEqual( fbs(3,:), dest(3,:), 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs(3,:), dest(3,:), 'absolute', 1e-6 );
assertEqual( no_hit(3), uint32(0) );
assertEqual( ifbs(3), uint32(0) );
assertEqual( isbs(3), uint32(0) );

% ray 4
assertElementsAlmostEqual( fbs(4,:), [ 0.5, 0, 1], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs(4,:), [ 0.5, 0,-1], 'absolute', 1e-6 );
assertEqual( no_hit(4), uint32(2) );
assertEqual( ifbs(4), uint32(1) );
assertEqual( isbs(4), uint32(10) );

% ray 5
assertElementsAlmostEqual( fbs(5,:), [ 0,1,-0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs(5,:), [ 0,-1,-0.5], 'absolute', 1e-6 );
assertEqual( no_hit(5), uint32(2) );
assertEqual( ifbs(5), uint32(6) );
assertEqual( isbs(5), uint32(2) );

% ray 6
assertElementsAlmostEqual( fbs(6,:), [ 0,0,0 ], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs(6,:), [ 0,0,0 ], 'absolute', 1e-6 );
assertEqual( no_hit(6), uint32(0) );
assertEqual( ifbs(6), uint32(0) );
assertEqual( isbs(6), uint32(0) );

% Things that should work:
quadriga_lib.ray_triangle_intersect( orig, dest, cube );
fbs_only = quadriga_lib.ray_triangle_intersect( orig, dest, cube );
[ ~,~ ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube );
[ ~,~,~,~ ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube );

% 6 outputs
try
    [ ~,~,~,~,~,~ ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% 2 inputs
try
    quadriga_lib.ray_triangle_intersect( orig, dest );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% 8 inputs (too many)
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube, [], [], 0, 0, 'extra' );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Ray mismatch
try
    fbs_only = quadriga_lib.ray_triangle_intersect( orig, dest(2:end,:), cube );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Empty rays
try
    fbs_only = quadriga_lib.ray_triangle_intersect( [], [], cube );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Inputs cannot be empty.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% NaNs in orig and dest
[ fbs2, sbs2, no_hit2, ifbs2, isbs2 ] = quadriga_lib.ray_triangle_intersect( [nan nan nan;orig(2:end,:);nan nan nan], [dest;nan nan nan], cube );
assertTrue( all(isnan(fbs2(1,:))) );
assertTrue( all(isnan(fbs2(end,:))) );
assertTrue( all(isnan(sbs2(1,:))) );
assertTrue( all(isnan(sbs2(end,:))) );

assertEqual( no_hit2(1), uint32(0) );
assertEqual( ifbs2(1), uint32(0) );
assertEqual( isbs2(1), uint32(0) );
assertEqual( no_hit2(end), uint32(0) );
assertEqual( ifbs2(end), uint32(0) );
assertEqual( isbs2(end), uint32(0) );

assertElementsAlmostEqual( fbs2(2,:), [ 1, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs2(2,:), [-1, 0, -0.5], 'absolute', 1e-6 );
assertEqual( no_hit2(2), uint32(2) );
assertEqual( ifbs2(2), uint32(5) );
assertEqual( isbs2(2), uint32(3) );

% NaNs in cube, but no hit (last row)
[ fbs2, sbs2, no_hit2, ifbs2, isbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, [cube(1:end-1,:);nan(1,9)] );
assertEqual( fbs, fbs2 );
assertEqual( sbs2, sbs );
assertEqual( no_hit2, no_hit );
assertEqual( ifbs2, ifbs );
assertEqual( isbs2, isbs );

% NaNs in cube, but hit (first row)
[ fbs2, sbs2, no_hit2, ifbs2, isbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, [nan(1,9);cube(2:end,:)] );

% ray 4 (hits face 1 at FBS)
assertElementsAlmostEqual( fbs2(4,:), [ 0.5, 0,-1], 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs2(4,:), dest(4,:), 'absolute', 1e-6 );
assertEqual( no_hit2(4), uint32(1) );
assertEqual( ifbs2(4), uint32(10) );
assertEqual( isbs2(4), uint32(0) );

% Col mismatch
try
    fbs_only = quadriga_lib.ray_triangle_intersect( orig(:,2:end), dest, cube );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orig'' must have at least 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end
try
    fbs_only = quadriga_lib.ray_triangle_intersect( orig, dest(:,2:end), cube );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''dest'' must have at least 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end
try
    fbs_only = quadriga_lib.ray_triangle_intersect( orig, dest,cube(:,2:end) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mesh'' must have at least 9 columns containing x,y,z coordinates of 3 vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Single-precision inputs (type casting via qd_mex_get_double_Mat) ----
[ fbs2, sbs2, no_hit2, ifbs2, isbs2 ] = quadriga_lib.ray_triangle_intersect( single(orig), single(dest), single(cube) );
assertElementsAlmostEqual( fbs2, fbs, 'absolute', 1e-5 );
assertElementsAlmostEqual( sbs2, sbs, 'absolute', 1e-5 );
assertEqual( no_hit2, no_hit );
assertEqual( ifbs2, ifbs );
assertEqual( isbs2, isbs );

% ---- Sub-mesh index (uint32) ----
[ cube_seg, smi ] = quadriga_lib.triangle_mesh_segmentation( cube, 4 );
aabb_seg = quadriga_lib.triangle_mesh_aabb( cube_seg, smi );

[ fbs_seg, sbs_seg, no_hit_seg, ifbs_seg, isbs_seg ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, smi );

% Segmentation reorders the mesh, so FBS/SBS coordinates must match but indices may differ.
% Verify intersection points and hit counts match the unsegmented result.
assertElementsAlmostEqual( fbs_seg, fbs, 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs_seg, sbs, 'absolute', 1e-6 );
assertEqual( no_hit_seg, no_hit );

% ---- Sub-mesh index as double (typecast path via qd_mex_typecast_Col) ----
[ fbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, double(smi) );
assertElementsAlmostEqual( fbs2, fbs_seg, 'absolute', 1e-6 );

% ---- Sub-mesh index as int32 (typecast path) ----
[ fbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, int32(smi) );
assertElementsAlmostEqual( fbs2, fbs_seg, 'absolute', 1e-6 );

% ---- Sub-mesh index validation: first index not 0 ----
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube, uint32([2; 6]) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-mesh must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Sub-mesh index validation: not sorted ----
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube, uint32([1; 6; 3]) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-mesh indices must be sorted in ascending order.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Sub-mesh index validation: exceeds mesh size ----
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube, uint32([1; 99]) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-mesh indices cannot exceed number of mesh elements.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Pre-computed AABB ----
[ fbs2, sbs2, no_hit2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, smi, aabb_seg );
assertElementsAlmostEqual( fbs2, fbs_seg, 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs2, sbs_seg, 'absolute', 1e-6 );
assertEqual( no_hit2, no_hit_seg );

% ---- AABB row count mismatch ----
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, smi, aabb_seg(1,:) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''aabb'' must match number of sub-meshes.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- AABB column count mismatch ----
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, smi, aabb_seg(:,1:3) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''aabb'' must have at least 6 columns';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Skipping optional args with [] ----
[ fbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube, [], [], 1, 0 );
assertElementsAlmostEqual( fbs2, fbs, 'absolute', 1e-6 );

[ fbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, smi, [], 1 );
assertElementsAlmostEqual( fbs2, fbs_seg, 'absolute', 1e-6 );

% ---- Kernel selector: GENERIC (1) ----
[ fbs2, sbs2, no_hit2, ifbs2, isbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube, [], [], 1 );
assertElementsAlmostEqual( fbs2, fbs, 'absolute', 1e-6 );
assertElementsAlmostEqual( sbs2, sbs, 'absolute', 1e-6 );
assertEqual( no_hit2, no_hit );
assertEqual( ifbs2, ifbs );
assertEqual( isbs2, isbs );

% ---- Kernel selector: AVX2 (2) ----
% AVX2 may not be available; if so, an error is expected
try
    [ fbs2, sbs2, no_hit2, ifbs2, isbs2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube, [], [], 2 );
    % If we get here, AVX2 is available - verify results
    assertElementsAlmostEqual( fbs2, fbs, 'absolute', 1e-5 );
    assertElementsAlmostEqual( sbs2, sbs, 'absolute', 1e-5 );
    assertEqual( no_hit2, no_hit );
    assertEqual( ifbs2, ifbs );
    assertEqual( isbs2, isbs );
catch ME
    expectedErrorMessage = 'AVX2 kernel requested but not available';
    if isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- Kernel selector: CUDA (3) - expected to fail without GPU ----
try
    quadriga_lib.ray_triangle_intersect( orig, dest, cube, [], [], 3 );
catch ME
    expectedErrorMessage = 'CUDA kernel requested but not available';
    if isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% ---- AVX2 kernel with sub-mesh and pre-computed AABB ----
try
    [ fbs2, sbs2, no_hit2 ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube_seg, smi, aabb_seg, 2 );
    assertElementsAlmostEqual( fbs2, fbs_seg, 'absolute', 1e-5 );
    assertElementsAlmostEqual( sbs2, sbs_seg, 'absolute', 1e-5 );
    assertEqual( no_hit2, no_hit_seg );
catch ME
    if isempty(strfind(ME.message, 'AVX2 kernel requested but not available'))
        error('moxunit:exceptionNotRaised', ['Unexpected error: "', ME.message, '"']);
    end
end

end