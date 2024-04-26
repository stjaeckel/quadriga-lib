function test_point_cloud_segmentation

points = zeros(4, 3);
points(:, 1) = 0.0:0.1:0.3;
points = repmat(points, 2, 1);
points(5:8, 1) = points(5:8, 1) + 40.0;
points = repmat(points, 2, 1);
points(1:8, 2) = points(1:8, 2) - 50.0;
points(9:16, 2) = points(9:16, 2) + 50.0;
points(:, 3) = points(:, 3) + 1.0;

% Test 1: Mesh size is already below threshold, test padding
[points_out, sub_cloud_index, forward_index, reverse_index] = quadriga_lib.point_cloud_segmentation(points, 1024, 10);

assertTrue( size(points_out,1) == 20 );
assertTrue( size(reverse_index,1) == 16 );
assertTrue( numel(sub_cloud_index) == 1 );
assertTrue( sub_cloud_index == uint32(0) );
assertTrue( forward_index(1) == uint32(1) );
assertTrue( reverse_index(1) == uint32(1) );

assertElementsAlmostEqual( points_out(1:16,:), points, 'absolute', 1e-14 );
assertElementsAlmostEqual( points_out(17:end,:), repmat([20.15,0,1],4,1), 'absolute', 1e-14 );

assertEqual( forward_index(1:16), uint32(1:16)' );
assertEqual( reverse_index, uint32(1:16)' );
assertEqual( forward_index(17:20), uint32([0,0,0,0])' );

% Test 2: Subdivide, no padding
points(:,2) = points(:,2) * 0.1;
[points_out, sub_cloud_index, forward_index, reverse_index] = quadriga_lib.point_cloud_segmentation(points, 4, 5);

assertTrue( numel(sub_cloud_index) == 4 );
assertTrue( size(points_out,1) == 20 );
assertTrue( size(reverse_index,1) == 16 );

assertTrue( all(sub_cloud_index == uint32([0,5,10,15])') );
assertTrue( all(forward_index == uint32([1,2,3,4,0,9,10,11,12,0,5,6,7,8,0,13,14,15,16,0])') );
assertTrue( all(reverse_index == uint32([1,2,3,4,11,12,13,14,6,7,8,9,16,17,18,19])') );

% No outputs
quadriga_lib.point_cloud_segmentation(points, 64, 1);
quadriga_lib.point_cloud_segmentation(points, 64, 1);
quadriga_lib.point_cloud_segmentation(points, 64);
quadriga_lib.point_cloud_segmentation(points);
quadriga_lib.point_cloud_segmentation(points, [], []);

try % 4 imputs
    quadriga_lib.point_cloud_segmentation(points, 64, 1, 1);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 0 imputs
    quadriga_lib.point_cloud_segmentation();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 5 outputs
    [~,~,~,~,~] = quadriga_lib.point_cloud_segmentation(points);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end