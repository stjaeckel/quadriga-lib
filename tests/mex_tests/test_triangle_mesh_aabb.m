function test_triangle_mesh_aabb

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

mesh_sub = quadriga_lib.subdivide_triangles(mesh, 3);

% Test no output
quadriga_lib.triangle_mesh_aabb( mesh_sub );

% Test default 
aabb = quadriga_lib.triangle_mesh_aabb( mesh_sub );
assertElementsAlmostEqual( aabb, [-1 1 -1 1 -1 1], 'absolute', 1e-14 );

% Test segmentation
[mesh_seg, sub_mesh_index] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64);
assertTrue( numel(sub_mesh_index) > 1 );

aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index );
assertTrue( size(aabb,1) == numel(sub_mesh_index) );

assertElementsAlmostEqual( max(abs(aabb)), [1 1 1 1 1 1], 'absolute', 1e-14 );

aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index, 8 );
assertTrue( size(aabb,1) == 8 );

assertElementsAlmostEqual( max(abs(aabb)), [1 1 1 1 1 1], 'absolute', 1e-14 );
assertElementsAlmostEqual( min(abs(aabb)), [0 0 0 0 0 0], 'absolute', 1e-14 );

try % 2 outputs
    [~,~] = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 0 inputs
    aabb = quadriga_lib.triangle_mesh_aabb(  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 4 inputs
    aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index, 1, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong mesh
    aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg(:,1:8)  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mesh'' must have 9 columns containing x,y,z coordinates of 3 vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub mesh index
    aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index + uint32([1;0;0]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-mesh must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub mesh index
    aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index + uint32([0;0;1000]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-mesh indices cannot exceed number of faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub mesh index
    aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index + uint32([0;1000;0]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-mesh indices must be sorted in ascending order.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong vec size
    aabb = quadriga_lib.triangle_mesh_aabb( mesh_seg, sub_mesh_index, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''vec_size'' cannot be 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end








