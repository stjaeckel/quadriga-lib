function test_point_cloud_aabb

% Create a 4x3 matrix initialized to zeros
points = zeros(4, 3);

% Fill the first column with values from 0.0 to 0.3 with a step of 0.1
points(:, 1) = 0.0:0.1:0.3;

% Repeat the matrix vertically twice
points = repmat(points, 2, 1);

% Add 40.0 to the first column of the second block of 4 rows
points(5:8, 1) = points(5:8, 1) + 40.0;

% Repeat the matrix vertically twice
points = repmat(points, 2, 1);

% Subtract 50.0 from the second column of the first 8 rowssub_mesh_index
points(1:8, 2) = points(1:8, 2) - 50.0;

% Add 50.0 to the second column of rows 9 to 16
points(9:16, 2) = points(9:16, 2) + 50.0;

% Add 1.0 to all elements in the third column
points(:, 3) = points(:, 3) + 1.0;

% Calculate bounding box
aabb = quadriga_lib.point_cloud_aabb(points);

T = [0.0, 40.3, -50.0, 50.0, 1.0, 1.0];
assertElementsAlmostEqual( aabb, T, 'absolute', 1e-14 );

sub_cloud_index =  uint32([0 8])';

% Calculate bounding box
aabb = quadriga_lib.point_cloud_aabb(points, sub_cloud_index);

T = [0.0, 40.3, -50.0, -50.0, 1.0, 1.0 ; 0.0, 40.3, 50.0, 50.0, 1.0, 1.0  ];
assertElementsAlmostEqual( aabb, T, 'absolute', 1e-14 );

try % 2 outputs
    [~,~] = quadriga_lib.point_cloud_aabb( points, sub_cloud_index );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 0 inputs
    aabb = quadriga_lib.point_cloud_aabb(  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 4 inputs
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index, 1, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong foramt
    aabb = quadriga_lib.point_cloud_aabb( points(:,1:2)  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''points'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub index
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index + uint32([1;0]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-cloud must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub index
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index + uint32([0;1000]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-cloud indices cannot exceed number of points.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong vec size
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_index, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''vec_size'' cannot be 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end








