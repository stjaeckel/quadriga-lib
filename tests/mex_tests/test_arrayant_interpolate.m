function test_arrayant_interpolate

% Simple interpolation in az-direction
ant = struct( 'e_theta_re', [-2,2], 'e_theta_im', [-1,1], ...
    'e_phi_re', [3,1], 'e_phi_im', [6,2], ...
    'azimuth_grid', [0,pi], 'elevation_grid', 0 );

[vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( ant, [0,1,2,3]*pi/4, [-0.5,0,0,0.5] );
assertElementsAlmostEqual( vr, [-2, -1, 0, 1], 1e-14);
assertElementsAlmostEqual( vi, [-1, -0.5, 0, 0.5], 1e-14);
assertElementsAlmostEqual( hr, [3, 2.5, 2, 1.5], 1e-14);
assertElementsAlmostEqual( hi, [6, 5, 4, 3], 1e-14);
assertElementsAlmostEqual( ds, [0, 0, 0, 0], 1e-14);
assertElementsAlmostEqual( az, [0,1,2,3]*pi/4, 1e-14);
assertElementsAlmostEqual( el, [-0.5, 0, 0, 0.5], 1e-14);

[vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( [], [0,1,2,3]*pi/4, [-0.5,0,0,0.5], [], [], [], [], [-2,2], [-1,1], [3,1], [6,2], [0,pi], 0 );
assertElementsAlmostEqual( vr, [-2, -1, 0, 1], 1e-14);
assertElementsAlmostEqual( vi, [-1, -0.5, 0, 0.5], 1e-14);
assertElementsAlmostEqual( hr, [3, 2.5, 2, 1.5], 1e-14);
assertElementsAlmostEqual( hi, [6, 5, 4, 3], 1e-14);
assertElementsAlmostEqual( ds, [0, 0, 0, 0], 1e-14);
assertElementsAlmostEqual( az, [0,1,2,3]*pi/4, 1e-14);
assertElementsAlmostEqual( el, [-0.5, 0, 0, 0.5], 1e-14);

% Simple interpolation in el-direction
ant = struct( 'e_theta_re', [-2; 2], 'e_theta_im', [-1; 1], ...
    'e_phi_re', [3; 1], 'e_phi_im', [6; 2], ...
    'azimuth_grid', 0, 'elevation_grid', [0, pi/2] );

[vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( ant, [-0.5,0,0,0.5], [0,1,2,3]*pi/8 );
assertElementsAlmostEqual( vr, [-2, -1, 0, 1], 1e-14);
assertElementsAlmostEqual( vi, [-1, -0.5, 0, 0.5], 1e-14);
assertElementsAlmostEqual( hr, [3, 2.5, 2, 1.5], 1e-14);
assertElementsAlmostEqual( hi, [6, 5, 4, 3], 1e-14);
assertElementsAlmostEqual( ds, [0, 0, 0, 0], 1e-14);
assertElementsAlmostEqual( az, [-0.5, 0, 0, 0.5], 1e-14);
assertElementsAlmostEqual( el, [0,1,2,3]*pi/8, 1e-14);

% Spheric interpolation in az-direction
ant = struct( 'e_theta_re', [1,0], 'e_theta_im', [0,1], ...
    'e_phi_re', [-2,0], 'e_phi_im', [0,-1], ...
    'azimuth_grid', [0,pi], 'elevation_grid', 0 );

[vr,vi,hr,hi] = quadriga_lib.arrayant_interpolate( ant, [0,1,2,3]*pi/4, [0,0,0,0] );
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);

% Spheric interpolation in el-direction
ant = struct( 'e_theta_re', [1;0], 'e_theta_im', [0;1], ...
    'e_phi_re', [-2;0], 'e_phi_im', [0;-1], ...
    'azimuth_grid', 0, 'elevation_grid', [0,pi/2] );

[vr,vi,hr,hi] = quadriga_lib.arrayant_interpolate( ant, [0,0,0,0], [0,1,2,3]*pi/8 );
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8).*[2,1.75,1.5,1.25], 1e-14);

% Spheric interpolation in az-direction with z-rotation
ant = struct( 'e_theta_re', [1,0], 'e_theta_im', [0,1], ...
    'e_phi_re', [-2,0], 'e_phi_im', [0,-1], ...
    'azimuth_grid', [0,pi], 'elevation_grid', 0 );

[vr,vi,hr,hi,~,az] = quadriga_lib.arrayant_interpolate( ant, [0,1,2,3]*pi/4, [0,0,0,0], 1, [0;0;-pi/8] );
assertElementsAlmostEqual( az, [0,1,2,3]*pi/4+pi/8, 1e-14);
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);

% Spheric interpolation in el-direction with y-rotation
ant = struct( 'e_theta_re', [1;0], 'e_theta_im', [0;1], ...
    'e_phi_re', [-2;0], 'e_phi_im', [0;-1], ...
    'azimuth_grid', 0, 'elevation_grid', [0,pi/2] );

[vr,vi,hr,hi,~,az,el] = quadriga_lib.arrayant_interpolate( ant, [0,0,0,0], [0,1,2,3]*pi/8, 1, [0;-pi/16;0] );
assertElementsAlmostEqual( az, [0,0,0,0], 1e-14);
assertElementsAlmostEqual( el, [0,1,2,3]*pi/8+pi/16, 1e-14);
assertElementsAlmostEqual( vr, cos([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( vi, sin([0,1,2,3]*pi/8+pi/16), 1e-14);
assertElementsAlmostEqual( hr, -cos([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);
assertElementsAlmostEqual( hi, -sin([0,1,2,3]*pi/8+pi/16).*[1.875,1.625, 1.375, 1.125], 1e-14);

% Polarization rotation using x-rotation
ant = struct( 'e_theta_re', [1,0], 'e_theta_im', [1,0], ...
    'e_phi_re', [0,0], 'e_phi_im', [0,0], ...
    'azimuth_grid', [0,pi], 'elevation_grid', 0 );

[vr,vi,hr,hi] = quadriga_lib.arrayant_interpolate( ant, 0, 0, [1,1], [pi/4 -pi/4;0 0; 0 0] );
assertElementsAlmostEqual( vr, [1;1]/sqrt(2), 1e-14);
assertElementsAlmostEqual( vi, [1;1]/sqrt(2), 1e-14);
assertElementsAlmostEqual( hr, [1;-1]/sqrt(2), 1e-14);
assertElementsAlmostEqual( hi, [1;-1]/sqrt(2), 1e-14);

% Test projected distance
ant = struct( 'e_theta_re', 1, 'e_theta_im', 0, ...
    'e_phi_re', 0, 'e_phi_im', 0, ...
    'azimuth_grid', 0, 'elevation_grid', 0 );

[vr,vi,hr,hi,ds] = quadriga_lib.arrayant_interpolate( ant, 0, 0, [1,1,1], [], eye(3));
assertElementsAlmostEqual( vr, [1;1;1], 1e-14);
assertElementsAlmostEqual( vi, [0;0;0], 1e-14);
assertElementsAlmostEqual( hr, [0;0;0], 1e-14);
assertElementsAlmostEqual( hi, [0;0;0], 1e-14);
assertElementsAlmostEqual( ds, [-1;0;0], 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( ant, 0, 0, [1,1,1], [], eye(3));
assertElementsAlmostEqual( ds, [-1;0;0], 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( ant, 3*pi/4, 0, [1,1,1], [], eye(3));
assertElementsAlmostEqual( ds, [1;-1;0]/sqrt(2), 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( ant, 0, -pi/4, [1,1,1], [], eye(3));
assertElementsAlmostEqual( ds, [-1;0;1]/sqrt(2), 1e-14);

[~,~,~,~,ds] = quadriga_lib.arrayant_interpolate( ant, [-pi,-pi/2,0], [0,0,-pi/2], [1,1,1], [], -eye(3));
assertElementsAlmostEqual( ds, -eye(3), 1e-14);

% ===================== Multi-frequency mode =====================

% Two isotropic single-element antennas at different center frequencies
ant1 = struct( 'e_theta_re', 1, 'e_theta_im', 0, ...
    'e_phi_re', 2, 'e_phi_im', 0, ...
    'azimuth_grid', 0, 'elevation_grid', 0, 'center_freq', 1e9 );
ant2 = struct( 'e_theta_re', 3, 'e_theta_im', 0, ...
    'e_phi_re', 4, 'e_phi_im', 0, ...
    'azimuth_grid', 0, 'elevation_grid', 0, 'center_freq', 2e9 );
ant_multi = [ant1, ant2];

% Request at the exact center frequencies -> stored values, cube [1,1,2]
[vr, vi, hr, hi] = quadriga_lib.arrayant_interpolate( ant_multi, 0, 0, [], [], [], [1e9, 2e9] );
assertEqual( size(vr), [1, 1, 2] );
assertEqual( size(hi), [1, 1, 2] );
assertElementsAlmostEqual( vr(:,:,1), 1, 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), 3, 1e-14 );
assertElementsAlmostEqual( hr(:,:,1), 2, 1e-14 );
assertElementsAlmostEqual( hr(:,:,2), 4, 1e-14 );
assertElementsAlmostEqual( vi, zeros(1,1,2), 1e-14 );
assertElementsAlmostEqual( hi, zeros(1,1,2), 1e-14 );

% Out-of-range frequencies clamp to nearest center_freq (no extrapolation)
[vr, ~, hr, ~] = quadriga_lib.arrayant_interpolate( ant_multi, 0, 0, [], [], [], [0.5e9, 3e9] );
assertElementsAlmostEqual( vr(:,:,1), 1, 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), 3, 1e-14 );
assertElementsAlmostEqual( hr(:,:,1), 2, 1e-14 );
assertElementsAlmostEqual( hr(:,:,2), 4, 1e-14 );

% Struct array with freq omitted -> multi-freq auto-mode using center_freq values
[vr, ~, hr, ~] = quadriga_lib.arrayant_interpolate( ant_multi, 0, 0 );
assertEqual( size(vr), [1, 1, 2] );
assertElementsAlmostEqual( vr(:,:,1), 1, 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), 3, 1e-14 );

% Multiple angles + multiple frequencies -> cube [n_out, n_ang, n_freq]
az_q = [0, 0, 0, 0];
el_q = [0, 0, 0, 0];
[vr, ~, ~, ~] = quadriga_lib.arrayant_interpolate( ant_multi, az_q, el_q, [], [], [], [1e9, 2e9] );
assertEqual( size(vr), [1, 4, 2] );
assertElementsAlmostEqual( vr(:,:,1), [1 1 1 1], 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), [3 3 3 3], 1e-14 );

% Multi-freq with element selection and duplication
ant_me_a = struct( 'e_theta_re', cat(3, 1, 10), 'e_theta_im', cat(3, 0, 0), ...
    'e_phi_re', cat(3, 0, 0), 'e_phi_im', cat(3, 0, 0), ...
    'azimuth_grid', 0, 'elevation_grid', 0, 'center_freq', 1e9 );
ant_me_b = struct( 'e_theta_re', cat(3, 3, 30), 'e_theta_im', cat(3, 0, 0), ...
    'e_phi_re', cat(3, 0, 0), 'e_phi_im', cat(3, 0, 0), ...
    'azimuth_grid', 0, 'elevation_grid', 0, 'center_freq', 2e9 );
ant_me = [ant_me_a, ant_me_b];

% Select only element 2
[vr, ~, ~, ~] = quadriga_lib.arrayant_interpolate( ant_me, 0, 0, 2, [], [], [1e9, 2e9] );
assertEqual( size(vr), [1, 1, 2] );
assertElementsAlmostEqual( vr(:,:,1), 10, 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), 30, 1e-14 );

% Duplicate elements: [1, 2, 1] -> n_out = 3
[vr, ~, ~, ~] = quadriga_lib.arrayant_interpolate( ant_me, 0, 0, [1, 2, 1], [], [], [1e9, 2e9] );
assertEqual( size(vr), [3, 1, 2] );
assertElementsAlmostEqual( vr(:,:,1), [1;10;1], 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), [3;30;3], 1e-14 );

% Multi-freq with element_pos override
[vr, ~, ~, ~] = quadriga_lib.arrayant_interpolate( ant_me, 0, 0, [1, 2], [], [0 0; 0 0; 0 1], [1e9, 2e9] );
assertEqual( size(vr), [2, 1, 2] );
assertElementsAlmostEqual( vr(:,:,1), [1;10], 1e-14 );
assertElementsAlmostEqual( vr(:,:,2), [3;30], 1e-14 );

% Exceptins

e = rand( 5,10,3 );
az_grid = linspace(-pi,pi,10);
el_grid = linspace(-pi/2, pi/2, 5);
az = 2*pi*(rand(1,6)-0.5);
el = pi*(rand(1,6)-0.5);

ant = struct( 'e_theta_re', e, 'e_theta_im', e, 'e_phi_re', e, 'e_phi_im', e, 'azimuth_grid', az_grid, 'elevation_grid', el_grid );

try
    vr = quadriga_lib.arrayant_interpolate(  );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( [], [0,1,2,3]*pi/4, [-0.5,0,0,0.5], [], [], [], [], [-2,2], [-1,1], [3,1], [6,2], [0,pi], 0, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [vr,vi,hr,hi,ds,az,el] = quadriga_lib.arrayant_interpolate( [], [0,1,2,3]*pi/4, [-0.5,0,0,0.5], [], [], [], [-2,2]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    vr = quadriga_lib.arrayant_interpolate( ant );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try
    [~,~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, az, el );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant.e_theta_re = [];
try
    [~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, az, el );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Missing data for any of: e_theta_re, e_theta_im, e_phi_re, e_phi_im, azimuth_grid, elevation_grid';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant.e_theta_re = 1;
try
    [~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, az, el );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sizes of ''e_theta_re'' and ''e_theta_im'' do not match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant.e_theta_re = single(e);
[~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, single(az), single(el) );
[~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, single(az), single(el),[] );

ant.azimuth_grid = 2*az_grid;
try
    [~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, az, el );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Values of ''azimuth_grid'' must be between -pi and pi (equivalent to -180 to 180 degree).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant.azimuth_grid = az_grid;
ant.elevation_grid = 2*el_grid;
try
    [~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, az, el );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Values of ''elevation_grid'' must be between -pi/2 and pi/2 (equivalent to -90 to 90 degree).';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

ant.elevation_grid = el_grid;
try
    [~,~,~,~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant, az, el, [], 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orientation'' must have 3 elements on the first dimension.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% >4 outputs in multi-frequency mode
try
    [~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant_multi, 0, 0, [], [], [], [1e9, 2e9] );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Multi-frequency mode supports at most 4 outputs';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Struct array without freq -> requesting dist (5th output) must fail: multi-freq auto-selected
try
    [~,~,~,~,~] = quadriga_lib.arrayant_interpolate( ant_multi, 0, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Multi-frequency mode supports at most 4 outputs';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end