% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_point_cloud_aabb
% MEX interface test for quadriga_lib.point_cloud_aabb.

%% Setup: 16 points in four blocks (x = 0..0.3 or 40..40.3, y = -50 or +50, z = 1)
points = zeros(4, 3);

points(:, 1) = 0.0:0.1:0.3;

points = repmat(points, 2, 1);

points(5:8, 1) = points(5:8, 1) + 40.0;

points = repmat(points, 2, 1);

points(1:8, 2) = points(1:8, 2) - 50.0;

points(9:16, 2) = points(9:16, 2) + 50.0;

points(:, 3) = points(:, 3) + 1.0;

aabb = quadriga_lib.point_cloud_aabb(points);

T = [0.0, 40.3, -50.0, 50.0, 1.0, 1.0];
assertElementsAlmostEqual( aabb, T, 'absolute', 1e-14 );

sub_cloud_ind =  uint32([1 9])';

% One box per sub-cloud
aabb = quadriga_lib.point_cloud_aabb(points, sub_cloud_ind);

T = [0.0, 40.3, -50.0, -50.0, 1.0, 1.0 ; 0.0, 40.3, 50.0, 50.0, 1.0, 1.0  ];
assertElementsAlmostEqual( aabb, T, 'absolute', 1e-14 );

%% Errors
try % 2 outputs
    [~,~] = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
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
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind, 1, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong format
    aabb = quadriga_lib.point_cloud_aabb( points(:,1:2)  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''points'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub index
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind + uint32([1;0]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''sub_cloud_ind'' must start at the first point.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong sub index
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind + uint32([0;1000]));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Entries of ''sub_cloud_ind'' cannot point beyond the last point.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Wrong vec size
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''vec_size'' cannot be 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % Vector for the scalar vec_size
    aabb = quadriga_lib.point_cloud_aabb( points, sub_cloud_ind, [8, 1] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''vec_size'' must be a scalar.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
