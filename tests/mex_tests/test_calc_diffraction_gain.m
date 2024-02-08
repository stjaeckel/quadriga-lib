function test_calc_diffraction_gain

cube = [  -1     1     1   ,    1    -1     1   ,    1     1     1;   %  1 Top NorthEast
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

mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 3.0],12,1);

orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];    % FBS West Upper (9), SBS East Upper (11)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];    % FBS East Lower Top (5), SBS West Lower (3)

gain = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0 );
assertElementsAlmostEqual( gain, [10^(-0.3);10^(-0.3)], 'absolute', 1e-14 );

% 0 outputs should be fine
quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9 );

% 2 outputs 
[gain, coord] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 5 );
assertElementsAlmostEqual( gain, [10^(-0.3);10^(-0.3)], 'absolute', 1e-14 );
assertElementsAlmostEqual( coord, permute([0,0 ; 0,0 ; 0.5,-0.5],[1,3,2]), 'absolute', 1e-14 );

try % 3 outputs
    [~, ~, ~,] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 5 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 8 input arguments
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop, 1e9, 0, 0, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest(1,:), cube, mtl_prop, 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl_prop
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop(:,1), 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mtl_prop'' must have 5 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mtl_prop
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_prop(1,:), 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''mesh'' and ''mtl_prop'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

end

