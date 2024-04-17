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

mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 0.0],12,1); % Air
mtl_prop(:,1) = 1.1:0.1:2.2;

[mesh_sub, mtl_prop_sub] = quadriga_lib.subdivide_triangles(mesh, 3, mtl_prop);

% Test 1: Mesh size is already below threshold, test padding
[mesh_seg, sub_mesh_index, mesh_index, mtl_prop_seg] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 1024, 8, mtl_prop_sub);

assertTrue( size(mesh_seg,1) == 112 );
assertTrue( size(mtl_prop_seg,1) == 112 );
assertTrue( numel(sub_mesh_index) == 1 );
assertTrue( sub_mesh_index == uint32(0) );

assertElementsAlmostEqual( mesh_seg(1:108,:), mesh_sub, 'absolute', 1e-14 );
assertElementsAlmostEqual( mesh_seg(109:end,:), zeros(4,9), 'absolute', 1e-14 );

assertElementsAlmostEqual( mtl_prop_seg(1:108,:), mtl_prop_sub, 'absolute', 1e-14 );
assertElementsAlmostEqual( mtl_prop_seg(109:end,:), [ones(4,1), zeros(4,4)], 'absolute', 1e-14 );

assertEqual( mesh_index(1:108), uint32(1:108)' );

% Test 2: Subdivide, no padding
[mesh_seg, sub_mesh_index, mesh_index, mtl_prop_seg] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1, mtl_prop_sub);

assertTrue( numel(sub_mesh_index) == 3 );
assertTrue( size(mesh_seg,1) == 108 );
assertTrue( size(mtl_prop_seg,1) == 108 );
assertTrue( sub_mesh_index(1) == uint32(0) );

% No outputs
quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1, mtl_prop_sub);
quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1);
quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64);
quadriga_lib.triangle_mesh_segmentation(mesh_sub);
quadriga_lib.triangle_mesh_segmentation(mesh_sub, [], []);

try % 5 imputs
    quadriga_lib.triangle_mesh_segmentation(mesh_sub, 64, 1, mtl_prop_sub, 1);
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
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl prop
    [~,~,~,~] = quadriga_lib.triangle_mesh_segmentation(mesh_sub, [], [],  mtl_prop_sub(1:5,:));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''mesh'' and ''mtl_prop'' dont match.';
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