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

mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 0.0],12,1); % Air

% Calc FBS and SBS
orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];    % FBS West Upper (9), SBS East Upper (11)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];    % FBS East Lower Top (5), SBS West Lower (3)
[ fbs, sbs, ~, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( orig, dest, mesh );

% Call without outputs - should be fine
quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );

% Reflection
origN = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( origN, [-1.001, 0, 0.5; 1.001, 0, -0.5], 'absolute', 1e-14 );

[~, destN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( destN, [-12, 0, 0.5; 12, 0, -0.5], 'absolute', 1e-14 );

[~, ~, gainN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( gainN, [0;0], 'absolute', 1e-14 );

[~, ~, ~, xprmatN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( xprmatN, zeros(2,8), 'absolute', 1e-14 );

[~, ~, ~, ~, trivecN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertTrue( isempty(trivecN) );

[~, ~, ~, ~, ~, tridirN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertTrue( isempty(tridirN) );

[~, ~, ~, ~, ~, ~, orig_lengthN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( orig_lengthN, [9.001;9.001], 'absolute', 1e-14 );

[~, ~, ~, ~, ~, ~, ~, fbs_angleN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( fbs_angleN, [pi/2;pi/2], 'absolute', 1e-14 );

[~, ~, ~, ~, ~, ~, ~, ~, thicknessN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( thicknessN, [2;2], 'absolute', 1e-14 );

[~, ~, ~, ~, ~, ~, ~, ~, ~, edge_lengthN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( edge_lengthN, [0;0], 'absolute', 1e-14 );

[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, normal_vecN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
assertElementsAlmostEqual( normal_vecN, [-1,0,0,1,0,0 ; 1,0,0,-1,0,0], 'absolute', 1e-14 );

try % 12 outputs
    [~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Transmission, use custom "orig_length"
[ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, normal_vecN]  = ...
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, [], [], [5;3] );
assertElementsAlmostEqual( origN, [-0.999, 0, 0.5; 0.999, 0, -0.5], 'absolute', 1e-14 );
assertElementsAlmostEqual( destN, dest, 'absolute', 1e-14 );
assertElementsAlmostEqual( gainN, [1;1], 'absolute', 1e-14 );
assertElementsAlmostEqual( xprmatN, [1,0,0,0,0,0,1,0 ; 1,0,0,0,0,0,1,0], 'absolute', 1e-14 );
assertTrue( isempty(trivecN) );
assertTrue( isempty(tridirN) );
assertElementsAlmostEqual( orig_lengthN, [14.001;12.001], 'absolute', 1e-14 );
assertElementsAlmostEqual( fbs_angleN, [pi/2;pi/2], 'absolute', 1e-14 );
assertElementsAlmostEqual( thicknessN, [2;2], 'absolute', 1e-14 );
assertElementsAlmostEqual( edge_lengthN, [0;0], 'absolute', 1e-14 );
assertElementsAlmostEqual( normal_vecN, [-1,0,0,1,0,0 ; 1,0,0,-1,0,0], 'absolute', 1e-14 );

% Using trivec and tridir, reflection
trivec = repmat([0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0],2,1);
tridir = zeros(2,6);
tridir(2,:) = [pi,0,pi,0,pi,1*pi/180];

[ ~, ~, ~, ~, trivecN, tridirN] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec, tridir );
a = tan(1.0 * pi/180) * 9.0;
assertElementsAlmostEqual( trivecN, [0.001, -0.1, 0.2, 0.001, -0.1, -0.2, 0.001, 0.2, 0 ; -0.001, -0.1, 0.2, -0.001, -0.1, -0.2, -0.001, 0.2, a], 'absolute', 1e-14 );
assertElementsAlmostEqual( tridirN, [pi,0,pi,0,pi,0 ; 0,0,0,0,0,1*pi/180], 'absolute', 1e-14 );

try % 9 imputs
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Need at least 10 input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 14 imputs
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, [], [], [5;3], 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Can have at most 13 input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong type
    quadriga_lib.ray_mesh_interact( 3, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Interaction type must be either (0) Reflection, (1) Transmission or (2) Refraction.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong freq
    quadriga_lib.ray_mesh_interact( 2, -1, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Center frequency must be provided in Hertz and have values > 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig(1,:), dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig(:,1), dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orig'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest(1,:), fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest(:,1), fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''dest'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs(1,:), sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''fbs'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs(:,1), sbs, mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''fbs'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs(1,:), mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''sbs'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs(:,1), mesh, mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''sbs'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh(1,:), mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''mesh'' and ''mtl_prop'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh(:,1), mtl_prop, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mesh'' must have 9 columns containing x,y,z coordinates of 3 vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop(1,:), fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''mesh'' and ''mtl_prop'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop(:,1), fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mtl_prop'' must have 5 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs_ind
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind(1), sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''fbs_ind'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs_ind
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind(1) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''sbs_ind'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % missing tridir
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'In order to use ray tubes, both ''trivec'' and ''tridir'' must be given.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % missing trivec
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, [], tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'In order to use ray tubes, both ''trivec'' and ''tridir'' must be given.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong trivec
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec(1,:), tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''trivec'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong trivec
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec(:,1), tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''trivec'' must have 9 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong tridir
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec, tridir(1,:) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''tridir'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong tridir
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, trivec, tridir(:,1) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tridir'' must have 6 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig_length
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, mesh, mtl_prop, fbs_ind, sbs_ind, [], [], 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''orig_length'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end