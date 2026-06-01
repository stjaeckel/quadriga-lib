function test_triangle_mesh_segmentation

mesh = [  -1     1     1   ,    1    -1     1   ,    1     1     1;   %  1 Top NorthEast
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

mtl_ind = (1:12)';   % 1-based per-face material index (each face its own material)

[mesh_sub, mtl_ind_sub] = quadriga_lib.subdivide_triangles(mesh, 3, mtl_ind);

% Test 1: Mesh size is already below threshold, test padding
[mesh_seg, sub_mesh_index, mesh_index, mtl_ind_seg] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 1024, 8, mtl_ind_sub);

assertTrue( size(mesh_seg,1) == 112 );
assertTrue( numel(mtl_ind_seg) == 112 );
assertTrue( numel(sub_mesh_index) == 1 );
assertTrue( sub_mesh_index == uint32(1) );

assertElementsAlmostEqual( mesh_seg(1:108,:), mesh_sub, 'absolute', 1e-14 );
assertElementsAlmostEqual( mesh_seg(109:end,:), zeros(4,9), 'absolute', 1e-14 );

assertEqual( mtl_ind_seg(1:108), mtl_ind_sub );
assertEqual( mtl_ind_seg(109:end), ones(4,1, 'like', mtl_ind_seg) );   % padding -> air (index 1)

assertEqual( mesh_index(1:108), uint32(1:108)' );

% Test 2: Subdivide, no padding
[mesh_seg, sub_mesh_index, mesh_index, mtl_ind_seg] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1, mtl_ind_sub);

assertTrue( numel(sub_mesh_index) == 3 );
assertTrue( size(mesh_seg,1) == 108 );
assertTrue( numel(mtl_ind_seg) == 108 );
assertTrue( sub_mesh_index(1) == uint32(1) );

% No outputs
quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1, mtl_ind_sub);
quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1);
quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64);
quadriga_lib.triangle_mesh_segmentation(mesh_sub);
quadriga_lib.triangle_mesh_segmentation(mesh_sub, [], []);

try % 5 imputs
    quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1, mtl_ind_sub, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 0 imputs
    quadriga_lib.triangle_mesh_segmentation();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 5 outputs
    [~,~,~,~,~] = quadriga_lib.triangle_mesh_segmentation(mesh_sub);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl_ind length
    [~,~,~,~] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, [], [], mtl_ind_sub(1:5));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of faces in ''mesh'' and length of ''mtl_ind'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl prop
    [~,~,~,~] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''target_size'' cannot be 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end


try % wrong mtl prop
    [~,~,~,~] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 1, 0);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''vec_size'' cannot be 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end


end