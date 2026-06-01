function test_ray_mesh_interact

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

mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],12,1); % Air, fRef = 1 GHz
[mtl_ind, mtl_st] = m2p(mtl_prop);

% Calc FBS and SBS
orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];    % FBS West Upper (9), SBS East Upper (11)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];    % FBS East Lower Top (5), SBS West Lower (3)
[ fbs, sbs, ~, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( orig, dest, mesh );

% Call without outputs - should be fine
quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );

% Reflection
origN = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( origN, [-1.001, 0, 0.5; 1.001, 0, -0.5], 'absolute', 1e-6 );

[~, destN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( destN, [-12, 0, 0.5; 12, 0, -0.5], 'absolute', 1e-6 );

[~, ~, gainN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( gainN, [0;0], 'absolute', 1e-6 );

[~, ~, ~, xprmatN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( xprmatN, zeros(2,8), 'absolute', 1e-6 );

[~, ~, ~, ~, trivecN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertTrue( isempty(trivecN) );

[~, ~, ~, ~, ~, tridirN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertTrue( isempty(tridirN) );

[~, ~, ~, ~, ~, ~, orig_lengthN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( orig_lengthN, [9.001;9.001], 'absolute', 1e-6 );

[~, ~, ~, ~, ~, ~, ~, fbs_angleN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( fbs_angleN, [pi/2;pi/2], 'absolute', 1e-6 );

[~, ~, ~, ~, ~, ~, ~, ~, thicknessN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( thicknessN, [2;2], 'absolute', 1e-6 );

[~, ~, ~, ~, ~, ~, ~, ~, ~, edge_lengthN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( edge_lengthN, [0;0], 'absolute', 1e-6 );

[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, normal_vecN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( normal_vecN, [-1,0,0,1,0,0 ; 1,0,0,-1,0,0], 'absolute', 1e-6 );

try % 13 outputs
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Transmission, use custom "orig_length"
[ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, normal_vecN]  = ...
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], [], [5;3] );
assertElementsAlmostEqual( origN, [-0.999, 0, 0.5; 0.999, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( destN, dest, 'absolute', 1e-6 );
assertElementsAlmostEqual( gainN, [1;1], 'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN, [1,0,0,0,0,0,1,0 ; 1,0,0,0,0,0,1,0], 'absolute', 1e-6 );
assertTrue( isempty(trivecN) );
assertTrue( isempty(tridirN) );
assertElementsAlmostEqual( orig_lengthN, [14.001;12.001], 'absolute', 1e-6 );
assertElementsAlmostEqual( fbs_angleN, [pi/2;pi/2], 'absolute', 1e-6 );
assertElementsAlmostEqual( thicknessN, [2;2], 'absolute', 1e-6 );
assertElementsAlmostEqual( edge_lengthN, [0;0], 'absolute', 1e-6 );
assertElementsAlmostEqual( normal_vecN, [-1,0,0,1,0,0 ; 1,0,0,-1,0,0], 'absolute', 1e-6 );

% Using trivec and tridir, reflection
trivec = repmat([0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0],2,1);
tridir = zeros(2,6);
tridir(2,:) = [pi,0,pi,0,pi,1*pi/180];

[ ~, ~, ~, ~, trivecN, tridirN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec, tridir );
a = tan(1.0 * pi/180) * 9.0;
assertElementsAlmostEqual( trivecN, [0.001, -0.1, 0.2, 0.001, -0.1, -0.2, 0.001, 0.2, 0 ; -0.001, -0.1, 0.2, -0.001, -0.1, -0.2, -0.001, 0.2, a], 'absolute', 1e-6 );
assertElementsAlmostEqual( tridirN, [pi,0,pi,0,pi,0 ; 0,0,0,0,0,1*pi/180], 'absolute', 1e-6 );

% out_typeN: int32 column vector, single-hit outside→inside on each ray
[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, out_typeN] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertEqual( class(out_typeN), 'int32' );
assertEqual( size(out_typeN), [2 1] );
assertElementsAlmostEqual( double(out_typeN), [1;1], 'absolute', 0 );  % code 1 = outside→inside

% Scalar reflection (type 3): air-to-air → zero gain
[origN, ~, gainN, xprmatN] = ...
    quadriga_lib.ray_mesh_interact( 3, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( origN,    [-1.001, 0, 0.5; 1.001, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( gainN,    [0;0],        'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN,  zeros(2,8),   'absolute', 1e-6 );  % [Re Im 0 0 0 0 0 0]

% Scalar transmission (type 4): air-to-air → unity gain, [Re Im 0 0 0 0 0 0]
[~, destN, gainN, xprmatN] = ...
    quadriga_lib.ray_mesh_interact( 4, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( destN,   dest,         'absolute', 1e-6 );
assertElementsAlmostEqual( gainN,   [1;1],        'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN, [1,0,0,0,0,0,0,0 ; 1,0,0,0,0,0,0,0], 'absolute', 1e-6 );

% Penetration-loss frequency scaling (9-col mtl_prop, exp B = 1)
%   Att = 6 dB @ 2 GHz, exp = 1   →  at 10 GHz: Att = 6·(10/2) = 30 dB  →  gain = 1e-3
mtl_att = repmat([1.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 2.0], 12, 1);
[mtl_att_ind, mtl_att_st] = m2p(mtl_att);
orig_p = [-2.0, 0.0, 0.5];
dest_p = [ 2.0, 0.0, 0.5];
[fbs_p, sbs_p, ~, fbs_ind_p, sbs_ind_p] = quadriga_lib.ray_triangle_intersect( orig_p, dest_p, mesh );
trivec_p = [0.0, -0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.0];
tridir_p = zeros(1,6);
[~, ~, gainN] = quadriga_lib.ray_mesh_interact( 1, 10e9, orig_p, dest_p, fbs_p, sbs_p, ...
    mesh, mtl_att_ind, mtl_att_st, fbs_ind_p, sbs_ind_p, trivec_p, tridir_p );
assertElementsAlmostEqual( gainN, 1e-3, 'absolute', 1e-9 );

% fRef parameterization equivalence: same physical material at fRef=1 GHz vs fRef=2 GHz
%   (linear scaling exponents = 1 throughout)
mtl_A = repmat([2.0, 1.0, 0.01, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0], 12, 1);  % at 1 GHz
mtl_B = repmat([4.0, 1.0, 0.02, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0], 12, 1);  % at 2 GHz
[mtl_A_ind, mtl_A_st] = m2p(mtl_A);
[mtl_B_ind, mtl_B_st] = m2p(mtl_B);
orig_e = [-1.5, 0.0, 0.0];
dest_e = [ 0.0, 0.0, 1.5];
[fbs_e, sbs_e, ~, fbs_ind_e, sbs_ind_e] = quadriga_lib.ray_triangle_intersect( orig_e, dest_e, mesh );
trivec_e = [0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0];
tridir_e = (pi/180) * [0.0, 45.0, 0.0, 45.0, 0.0, 45.0];
[oNa, dNa, gNa, xNa] = quadriga_lib.ray_mesh_interact( 2, 10e9, orig_e, dest_e, fbs_e, sbs_e, ...
    mesh, mtl_A_ind, mtl_A_st, fbs_ind_e, sbs_ind_e, trivec_e, tridir_e );
[oNb, dNb, gNb, xNb] = quadriga_lib.ray_mesh_interact( 2, 10e9, orig_e, dest_e, fbs_e, sbs_e, ...
    mesh, mtl_B_ind, mtl_B_st, fbs_ind_e, sbs_ind_e, trivec_e, tridir_e );
assertElementsAlmostEqual( gNa, gNb, 'absolute', 1e-12 );
assertElementsAlmostEqual( xNa, xNb, 'absolute', 1e-12 );
assertElementsAlmostEqual( oNa, oNb, 'absolute', 1e-12 );
assertElementsAlmostEqual( dNa, dNb, 'absolute', 1e-12 );

% α in-medium absorption (col 7,8 of mtl_prop):
%   ε_r=1.5, σ=0, α = 2 dB/m @ 5 GHz, exp = 1   →   at 10 GHz: α = 4 dB/m
%   ray starts INSIDE cube, hits east wall at 45°, thickness traversed in-medium = √(0.5²+0.5²)+0.001
mtl_alpha = repmat([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 5.0], 12, 1);
[mtl_alpha_ind, mtl_alpha_st] = m2p(mtl_alpha);
orig_a = [0.5, 0.1, 0.0];
dest_a = [2.0, 1.6, 0.0];
[fbs_a, sbs_a, ~, fbs_ind_a, sbs_ind_a] = quadriga_lib.ray_triangle_intersect( orig_a, dest_a, mesh );
trivec_a = [0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0];
tridir_a = (pi/180) * [45.0, 0.0, 45.0, 0.0, 45.0, 0.0];
[~, ~, gainN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig_a, dest_a, fbs_a, sbs_a, ...
    mesh, mtl_alpha_ind, mtl_alpha_st, fbs_ind_a, sbs_ind_a, trivec_a, tridir_a );
% Lossless reflection coefficient at 45°, ε_r=1.5 → ε_r=1: from ITU-R P.2040, R_TE/R_TM averaged
% Easier: just check gain is in expected α-loss bracket (4 dB/m × 0.708 m ≈ 2.83 dB)
assertTrue( gainN > 0 && gainN < 1 );

% Expected gain: lossless 45° reflection coefficient × α-loss over the in-medium path
cos_th  = cos(45*pi/180);
sin2_th = sin(45*pi/180)^2;
eta1 = 1.5;  eta2 = 1.0;
cos_th2 = sqrt(1 - (eta1/eta2) * sin2_th);
n1 = sqrt(eta1);  n2 = sqrt(eta2);
R_te = (n1*cos_th - n2*cos_th2) / (n1*cos_th + n2*cos_th2);
R_tm = (n2*cos_th - n1*cos_th2) / (n2*cos_th + n1*cos_th2);
refl_gain = 0.5 * (abs(R_te)^2 + abs(R_tm)^2);          % ≈ 0.03847

thickness  = sqrt(0.5^2 + 0.5^2) + 0.001;               % orig → fbs + ray offset
alpha_10   = 2.0 * (10/5)^1;                            % 4 dB/m at 10 GHz
alpha_loss = 10^(-0.1 * thickness * alpha_10);

assertElementsAlmostEqual( gainN, refl_gain * alpha_loss, 'relative', 1e-9 );

try % 9 imputs
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 14 imputs
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], [], [5;3], 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong type
    quadriga_lib.ray_mesh_interact( 5, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Interaction type must be';   % use shortest stable substring
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong freq
    quadriga_lib.ray_mesh_interact( 2, -1, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Center frequency must be provided in Hertz and have values > 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig(1,:), dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig(:,1), dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orig'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest(1,:), fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest(:,1), fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''dest'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs(1,:), sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''fbs'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs(:,1), sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''fbs'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs(1,:), mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''sbs'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs(:,1), mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''sbs'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh(1,:), mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh(:,1), mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mesh'' must have 9 columns containing x,y,z coordinates of 3 vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind(1), mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl_prop
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind(1), mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs_ind
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind(1), sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''fbs_ind'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs_ind
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind(1) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''sbs_ind'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % missing tridir
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'In order to use ray tubes, both ''trivec'' and ''tridir'' must be given.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % missing trivec
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'In order to use ray tubes, both ''trivec'' and ''tridir'' must be given.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong trivec
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec(1,:), tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''trivec'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong trivec
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec(:,1), tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''trivec'' must have 9 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong tridir
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec, tridir(1,:) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''tridir'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong tridir
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec, tridir(:,1) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tridir'' must have 6 or 9 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig_length
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], [], 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''orig_length'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

% Convert a per-face [n_face, 9] material matrix with columns
% {a,b,c,d,att,attB,alpha,alphaB,fRef} into the (mtl_ind, struct) pair the new
% API expects. Identical rows are deduplicated; mtl_ind is 1-based.
function [mtl_ind, st] = m2p(M)
names = {'a','b','c','d','att','attB','alpha','alphaB','fRef'};
n = size(M,1);
uniq = zeros(0, size(M,2));
mtl_ind = zeros(n,1,'uint64');
for f = 1:n
    hit = 0;
    for m = 1:size(uniq,1)
        if all( abs(M(f,:) - uniq(m,:)) == 0 )
            hit = m; break;
        end
    end
    if hit == 0
        uniq(end+1,:) = M(f,:); %#ok<AGROW>
        hit = size(uniq,1);
    end
    mtl_ind(f) = hit;   % 1-based
end
st = struct();
for c = 1:9
    st.(names{c}) = uniq(:,c);
end
end