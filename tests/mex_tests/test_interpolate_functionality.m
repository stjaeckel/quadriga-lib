function test_interpolate_functionality

% Simple interpolation in az-direction
[vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( [-2,2], [-1,1], [3,1], [6,2], [0,pi], 0, [0,1,2,3]*pi/4, [-0.5,0,0,0.5] );
assertElementsAlmostEqual( vr, [-2, -1, 0, 1], 1e-14);
assertElementsAlmostEqual( vi, [-1, -0.5, 0, 0.5], 1e-14);
assertElementsAlmostEqual( hr, [3, 2.5, 2, 1.5], 1e-14);
assertElementsAlmostEqual( hi, [6, 5, 4, 3], 1e-14);
assertElementsAlmostEqual( ds, [0, 0, 0, 0], 1e-14);
assertElementsAlmostEqual( az, [0,1,2,3]*pi/4, 1e-14);
assertElementsAlmostEqual( el, [-0.5, 0, 0, 0.5], 1e-14);

% Simple interpolation in el-direction
[vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( [-2;2], [-1;1], [3;1], [6;2], 0, [0, pi/2], [-0.5,0,0,0.5], [0,1,2,3]*pi/8 );
assertElementsAlmostEqual( vr, [-2, -1, 0, 1], 1e-14);
assertElementsAlmostEqual( vi, [-1, -0.5, 0, 0.5], 1e-14);
assertElementsAlmostEqual( hr, [3, 2.5, 2, 1.5], 1e-14);
assertElementsAlmostEqual( hi, [6, 5, 4, 3], 1e-14);
assertElementsAlmostEqual( ds, [0, 0, 0, 0], 1e-14);
assertElementsAlmostEqual( az, [-0.5, 0, 0, 0.5], 1e-14);
assertElementsAlmostEqual( el, [0,1,2,3]*pi/8, 1e-14);

% Spheric interpolation in az-direction
[vr,vi,hr,hi] = quadriga_lib.arrayant_interpolate( [1,0], [0,1], [-2,0], [0,-1], [0,pi], 0, [0,1,2,3]*pi/4, [0,0,0,0] );
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);

% Spheric interpolation in el-direction
[vr,vi,hr,hi] = quadriga_lib.arrayant_interpolate( [1;0], [0;1], [-2;0], [0;-1], 0, [0,pi/2], [0,0,0,0], [0,1,2,3]*pi/8 );
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);

% Spheric interpolation in az-direction with z-rotation
[vr,vi,hr,hi,~,az] = quadriga_lib.arrayant_interpolate( [1,0], [0,1], [-2,0], [0,-1], [0,pi], 0, [0,1,2,3]*pi/4, [0,0,0,0], 1, [0;0;-pi/8] );
assertElementsAlmostEqual( az, [0,1,2,3]*pi/4+pi/8, 1e-14);
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);

% Spheric interpolation in el-direction with y-rotation
[vr,vi,hr,hi,~,az,el] = quadriga_lib.arrayant_interpolate( [1;0], [0;1], [-2;0], [0;-1], 0, [0,pi/2], [0,0,0,0], [0,1,2,3]*pi/8, 1, [0;-pi/16;0] );
assertElementsAlmostEqual( az, [0,0,0,0], 1e-14);
assertElementsAlmostEqual( el, [0,1,2,3]*pi/8+pi/16, 1e-14);
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);

% Polarization rotation using x-rotation
[vr,vi,hr,hi] = quadriga_lib.arrayant_interpolate( [1,0], [1,0], [0,0], [0,0], [0,pi], 0, 0, 0, [1,1], [pi/4 -pi/4;0 0; 0 0] );
assertElementsAlmostEqual( vr, [1;1]/sqrt(2), 1e-14);
assertElementsAlmostEqual( vi, [1;1]/sqrt(2), 1e-14);
assertElementsAlmostEqual( hr, [1;-1]/sqrt(2), 1e-14);
assertElementsAlmostEqual( hi, [1;-1]/sqrt(2), 1e-14);

% Test projected distance
[vr,vi,hr,hi,ds] = quadriga_lib.arrayant_interpolate( 1, 0, 0, 0, 0, 0, 0, 0, [1,1,1], [], eye(3));
assertElementsAlmostEqual( vr, [1;1;1], 1e-14);
assertElementsAlmostEqual( vi, [0;0;0], 1e-14);
assertElementsAlmostEqual( hr, [0;0;0], 1e-14);
assertElementsAlmostEqual( hi, [0;0;0], 1e-14);
assertElementsAlmostEqual( ds, [-1;0;0], 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( 1, 0, 0, 0, 0, 0, 0, 0, [1,1,1], [], eye(3));
assertElementsAlmostEqual( ds, [-1;0;0], 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( 1, 0, 0, 0, 0, 0, 3*pi/4, 0, [1,1,1], [], eye(3));
assertElementsAlmostEqual( ds, [1;-1;0]/sqrt(2), 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( 1, 0, 0, 0, 0, 0, 0, -pi/4, [1,1,1], [], eye(3));
assertElementsAlmostEqual( ds, [-1;0;1]/sqrt(2), 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( 1, 0, 0, 0, 0, 0, [-pi,-pi/2,0], [0,0,-pi/2], [1,1,1], [], -eye(3));
assertElementsAlmostEqual( ds, -eye(3), 1e-14);
