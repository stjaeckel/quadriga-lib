function test_cube_plane_winding

% Cube: a negative product of the scales swaps the winding, so the normals stay outward
scales = { -1, [1,1,-2], [-1,-1,1], [-1,-2,-0.5] };
for k = 1:numel(scales)
    loc = [3, -1, 2];
    m = quadriga_lib.cube( scales{k}, [], loc );
    n = cross( m(:,4:6) - m(:,1:3), m(:,7:9) - m(:,1:3), 2 );
    fc = ( m(:,1:3) + m(:,4:6) + m(:,7:9) ) / 3 - loc;
    assertTrue( all( sum( n .* fc, 2 ) > 0 ) );
end

% Plane: a negative product of the x and y scales swaps the winding, the normals stay +z
scales = { [-1,1], [2,-0.5], -1, [-1,-3], [-1,1,5] };
for k = 1:numel(scales)
    m = quadriga_lib.plane( scales{k} );
    n = cross( m(:,4:6) - m(:,1:3), m(:,7:9) - m(:,1:3), 2 );
    assertTrue( all( n(:,3) > 0 ) );
    assertElementsAlmostEqual( n(:,1:2), zeros(size(n,1),2), 'absolute', 1e-12 );
end

end
