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

[  ~, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 0 );
assertTrue( isempty( ray_ind ) );

[ ~, ray_ind_reference ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1 );

[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, [], 5 );
assertTrue( all( hit_count == uint32(1) ) );
assertTrue( all( ray_ind == ray_ind_reference ) );

[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 2, uint32(0), 5 );
assertTrue( all( hit_count == uint32(1) ) );
assertTrue( all( ray_ind(:,1) == ray_ind_reference ) );
assertTrue( all( ray_ind(:,2) == zeros(size(points,1),1,'uint32') ) );

quadriga_lib.ray_point_intersect( orig, trivec, tridir, points );

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
    quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, uint32(0), 5, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 3 outputs
    [~,~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % orig error
    [~,~] = quadriga_lib.ray_point_intersect( orig(1:2,:), trivec, tridir, points, 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''trivec'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % orig error
    [~,~] = quadriga_lib.ray_point_intersect( orig(:,1:2), trivec, tridir, points, 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orig'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % trivec error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec(:,1:2), tridir, points, 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''trivec'' must have 9 columns containing x,y,z coordinates of ray tube vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % tridir error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir(1:2,:), points, 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''tridir'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % tridir error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir(:,1:2), points, 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tridir'' must have 9 columns containing ray directions in Cartesian format.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % points error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points(:), 1, uint32(0), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''points'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % sub-cloud error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, uint32(1), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-cloud must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % sub-cloud error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, uint32([0,7]), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-clouds must be aligned with the SIMD vector size (8 for AVX2, 32 for CUDA).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % sub-cloud error
    [~,~] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, points, 1, uint32([0,32]), 5);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-cloud indices cannot exceed number of points.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end
