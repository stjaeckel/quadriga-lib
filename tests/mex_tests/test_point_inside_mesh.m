function test_point_inside_mesh
    % Define the mesh as a 12×9 matrix (each row: [x1 y1 z1  x2 y2 z2  x3 y3 z3]):
    mesh = [ ...
        -1.0,  1.0,  1.0,   1.0, -1.0,  1.0,   1.0,  1.0,  1.0;   %  0 Top NorthEast
         1.0, -1.0,  1.0,  -1.0, -1.0, -1.0,   1.0, -1.0, -1.0;   %  1 South Lower
        -1.0, -1.0,  1.0,  -1.0,  1.0, -1.0,  -1.0, -1.0, -1.0;   %  2 West Lower
         1.0,  1.0, -1.0,  -1.0, -1.0, -1.0,  -1.0,  1.0, -1.0;   %  3 Bottom NorthWest
         1.0,  1.0,  1.0,   1.0, -1.0, -1.0,   1.0,  1.0, -1.0;   %  4 East Lower
        -1.0,  1.0,  1.0,   1.0,  1.0, -1.0,  -1.0,  1.0, -1.0;   %  5 North Lower
        -1.0,  1.0,  1.0,  -1.0, -1.0,  1.0,   1.0, -1.0,  1.0;   %  6 Top SouthWest
         1.0, -1.0,  1.0,  -1.0, -1.0,  1.0,  -1.0, -1.0, -1.0;   %  7 South Upper
        -1.0, -1.0,  1.0,  -1.0,  1.0,  1.0,  -1.0,  1.0, -1.0;   %  8 West Upper
         1.0,  1.0, -1.0,   1.0, -1.0, -1.0,  -1.0, -1.0, -1.0;   %  9 Bottom SouthEast
         1.0,  1.0,  1.0,   1.0, -1.0,  1.0,   1.0, -1.0, -1.0;   % 10 East Upper
        -1.0,  1.0,  1.0,   1.0,  1.0,  1.0,   1.0,  1.0, -1.0    % 11 North Upper
    ];

    % Define the two query points as a 2×3 array:
    points = [ ...
         0.0,  0.0,  0.5;   % inside
        -1.1,  0.0,  0.0    % outside
    ];

    % Define obj_ind = [2;2;…;2]
    obj_ind = ones(12,1) * 2;

    % ───────────────────────────────────────────────────────────────
    % Case 1: Provide (points, mesh, obj_ind) without distance
    % Expected (from C++/Python reference):
    %   res should be a 2×1 uint32 vector,
    %   res(1) == 2, res(2) == 0
    % ───────────────────────────────────────────────────────────────
    res = quadriga_lib.point_inside_mesh(points, mesh, obj_ind);
    assertEqual(2, numel(res), 'Number of results must be 2');
    assertEqual(uint32(2), res(1), 'First point should map to object index 2');
    assertEqual(uint32(0), res(2), 'Second point is outside → 0');

    % ───────────────────────────────────────────────────────────────
    % Case 2: Provide only (points, mesh), no obj_ind, no distance
    % Expected:
    %   res(1) == 1  (face index + 1 of the containing triangle)
    %   res(2) == 0
    % ───────────────────────────────────────────────────────────────
    res = quadriga_lib.point_inside_mesh(points, mesh);
    assertEqual(2, numel(res), 'Number of results must be 2');
    assertEqual(uint32(1), res(1), 'Without obj_ind, returns face index+1 = 1');
    assertEqual(uint32(0), res(2), 'Outside → 0');

    % ───────────────────────────────────────────────────────────────
    % Case 3: Provide (points, mesh, [], distance = 0.12)
    % Passing empty obj_ind ([]) and specifying distance
    % Expected:
    %   res(1) == 1  (within tolerance of face 0)
    %   res(2) == 1  (second point is within 0.12 of some face)
    % ───────────────────────────────────────────────────────────────
    res = quadriga_lib.point_inside_mesh(points, mesh, [], 0.12);
    assertEqual(2, numel(res), 'Number of results must be 2');
    assertEqual(uint32(1), res(1), 'First point within 0.12 → face index+1 = 1');
    assertEqual(uint32(1), res(2), 'Second point within 0.12 of face 0 → 1');

    % ───────────────────────────────────────────────────────────────
    % Case 4: Provide (points, mesh, obj_ind, distance = 0.09)
    % Expected (tolerance too small to catch either point near face):
    %   res(1) == 2
    %   res(2) == 0
    % ───────────────────────────────────────────────────────────────
    res = quadriga_lib.point_inside_mesh(points, mesh, obj_ind, 0.09);
    assertEqual(2, numel(res), 'Number of results must be 2');
    assertEqual(uint32(2), res(1), 'First point exactly inside → obj_ind=2');
    assertEqual(uint32(0), res(2), 'Second still outside → 0');

    % ───────────────────────────────────────────────────────────────
    % Case 5: Provide (points, mesh, obj_ind, distance = 2.0)
    % With large tolerance, both points are “close enough” to some face.
    % Expected:
    %   res(1) == 2
    %   res(2) == 2
    % ───────────────────────────────────────────────────────────────
    res = quadriga_lib.point_inside_mesh(points, mesh, obj_ind, 2.0);
    assertEqual(2, numel(res), 'Number of results must be 2');
    assertEqual(uint32(2), res(1), 'First point → obj_ind = 2');
    assertEqual(uint32(2), res(2), 'Second point within 2.0 → obj_ind = 2');
end
