function test_ray_point_intersect

points = zeros(4, 3);
points(:, 1) = -0.1:0.1:0.2;
points = repmat(points, 2, 1);
points(5:8, 1) = points(5:8, 1) + 40.0;
points = repmat(points, 2, 1);
points(1:8, 2) = points(1:8, 2) - 5.0;
points(9:16, 2) = points(9:16, 2) + 5.0;
points(:, 3) = points(:, 3) + 1.0;

[ orig, ~, trivec, tridir ] = quadriga_lib.icosphere( 20, 1, 1 );
orig = orig - [10,20,30];

hit_count = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points );
assertTrue( all( hit_count == uint32(1) ) );

% Test second output: ray_ind shape, type, 1-based content
[hit_count, ray_ind] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points );
assertTrue( isa(ray_ind, 'uint32') );
assertTrue( isa(hit_count, 'uint32') );

assertEqual( size(ray_ind, 2), size(points, 1) );          % one column per point
assertEqual( uint32(sum(ray_ind ~= 0, 1)'), hit_count );           % nnz per col matches hit_count
nz = ray_ind(ray_ind ~= 0);
assertTrue( all( nz >= 1 & nz <= uint32(size(orig,1)) ) ); % indices are 1-based

% Empty sub_cloud_index behaves like omitting it
hit_count_e = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, [] );
assertEqual( hit_count, hit_count_e );

% Valid manual sub_cloud_index (split into two halves of 8 points each)
sub_idx = uint32([1, 9]);
[hit_count_s, ray_ind_s] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, sub_idx );
assertEqual( hit_count, hit_count_s );
assertEqual( ray_ind, ray_ind_s );

% Integration with point_cloud_segmentation: results must match after un-permuting
[pointsR, sub_cloud_index, ~, reverse_index] = ...
    quadriga_lib.point_cloud_segmentation( points, 4, 8 );
hit_count_R = quadriga_lib.ray_point_intersect( orig, trivec, tridir, pointsR, sub_cloud_index );
assertEqual( hit_count, hit_count_R(reverse_index) );

% Explicit GENERIC kernel must produce the same result as auto
hit_count_g = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, [], 1 );
assertEqual( hit_count, hit_count_g );

try % 3 imputs
    quadriga_lib.ray_point_intersect( orig, trivec, tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 8 imputs
    quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, 0, 0, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 3 outputs
    [~,~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % orig error
    [~,~] = quadriga_lib.ray_point_intersect( orig(1:2,:), trivec, tridir, points);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''trivec'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % orig error
    [~,~] = quadriga_lib.ray_point_intersect( orig(:,1:2), trivec, tridir, points);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orig'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % trivec error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec(:,1:2), tridir, points);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''trivec'' must have 9 columns containing x,y,z coordinates of ray tube vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % tridir error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir(1:2,:), points);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''tridir'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % tridir error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir(:,1:2), points);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tridir'' must have 9 columns containing ray directions in Cartesian format.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % points error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points(:));
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''points'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % sub-cloud error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, [2,8]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-cloud must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % sub-cloud error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, [1,33]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-cloud indices cannot exceed number of points.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
