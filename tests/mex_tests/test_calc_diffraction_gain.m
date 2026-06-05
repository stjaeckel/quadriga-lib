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

% New 9-column mtl_prop: [eps_r, eps_rB, sigma, sigmaB, att, attB, alpha, alphaB, fRef_GHz]
% Trivial Fresnel/conductive params; att = 3 dB at fRef = 1 GHz, no freq scaling.
mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[mtl_ind, mtl_st] = m2p(mtl_prop);

orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];    % FBS West Upper (9), SBS East Upper (11)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];    % FBS East Lower Top (5), SBS West Lower (3)

% Basic diffraction gain, lod = 0
gain = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0 );
assertElementsAlmostEqual( gain, [10^(-0.3);10^(-0.3)], 'absolute', 1e-10 );

% 0 outputs should be fine
quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9 );

% 1 output only
gain_only = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2 );
assertTrue( numel(gain_only) == 2 );

% 2 outputs, lod = 5
[gain5, coord5] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 5 );
assertElementsAlmostEqual( gain5, [10^(-0.3);10^(-0.3)], 'absolute', 1e-6 );
assertElementsAlmostEqual( coord5, permute([0,0 ; 0,0 ; 0.5,-0.5],[1,3,2]), 'absolute', 1e-10 );

% LOS (unobstructed) path: TX and RX above the cube, gain should be ~1.0
orig_los = [ 0, 0, 5 ];
dest_los = [ 0, 0, 10 ];
gain_los = quadriga_lib.calc_diffraction_gain( orig_los, dest_los, cube, mtl_ind, mtl_st, 1e9, 2 );
assertElementsAlmostEqual( gain_los, 1.0, 'absolute', 1e-6 );

% Single-precision inputs should work (cast to double internally)
gain_single = quadriga_lib.calc_diffraction_gain( single(orig), single(dest), ...
    single(cube), mtl_ind, mtl_st, 1e9, 2 );
assertElementsAlmostEqual( gain_only, gain_single, 'absolute', 1e-5 );

% Empty sub-mesh index
[~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, [] );
[~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 1 );

% sub_mesh_index with non-uint32 numeric (typecast path in new wrapper)
gain_smi = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 1 );
assertElementsAlmostEqual( gain, gain_smi, 'absolute', 1e-10 );

% use_kernel = 1 (GENERIC), should match default
gain_generic = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2, 0, [], 1 );
assertElementsAlmostEqual( gain_only, gain_generic, 'absolute', 1e-10 );

% use_kernel = 1 with gpu_id = 0 (all 10 args)
gain_full = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2, 0, [], 1, 0 );
assertElementsAlmostEqual( gain_only, gain_full, 'absolute', 1e-10 );

% Verify coord dimensions for each lod value
[~, c1] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 1 );
assertTrue( isequal(size(c1), [3, 2, 2]) );   % lod 1 -> n_seg=2

[~, c2] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2 );
assertTrue( isequal(size(c2), [3, 2, 2]) );   % lod 2 -> n_seg=2

[~, c3] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 3 );
assertTrue( isequal(size(c3), [3, 3, 2]) );   % lod 3 -> n_seg=3

[~, c4] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 4 );
assertTrue( isequal(size(c4), [3, 4, 2]) );   % lod 4 -> n_seg=4

[~, c6] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 6 );
assertTrue( isequal(size(c6), [3, 1, 2]) );   % lod 6 -> n_seg=1

% 5 args (explicit center_freq)
gain5a = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 2e9 );

% Alpha (in-medium distance absorption):
%   eps_r = 1 (no Fresnel), sigma = 0, att = 0, alpha = 4 dB/m, all exponents 0, fRef = 1.
%   Path from (-10,0,0.5) to (0.5,0,0.5) enters cube at x=-1 and ends at x=0.5
%   -> 1.5 m inside the medium -> 4 dB/m * 1.5 m = 6 dB -> gain = 10^(-0.6).
mtl_alpha = repmat([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], 12, 1);
[mtl_alpha_ind, mtl_alpha_st] = m2p(mtl_alpha);
orig_in   = [ -10.0, 0.0, 0.5 ];
dest_in   = [   0.5, 0.0, 0.5 ];
gain_alpha = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, mtl_alpha_ind, mtl_alpha_st, 10e9, 0 );
assertElementsAlmostEqual( gain_alpha, 10^(-0.6), 'absolute', 1e-7 );

% Penetration loss frequency scaling (attB):
%   att = 3 dB at fRef = 2 GHz, attB = 1, no Fresnel/conductive/alpha losses.
%   At 10 GHz: Att = 3 * (10/2)^1 = 15 dB -> gain = 10^(-1.5).
mtl_attB = repmat([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0], 12, 1);
[mtl_attB_ind, mtl_attB_st] = m2p(mtl_attB);
gain_attB = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, mtl_attB_ind, mtl_attB_st, 10e9, 0 );
assertElementsAlmostEqual( gain_attB, 10^(-1.5), 'absolute', 1e-10 );

% fRef parameterization equivalence:
%   At every frequency f the two materials yield identical effective values:
%     eps_r = 1.5*f,  sigma = 0.001*f,  Att = 2*f dB,  alpha = 0.5*f dB/m.
%   mat_A states them at fRef = 1 GHz, mat_B at fRef = 2 GHz with doubled
%   reference values; gains must agree at any evaluation frequency.
mat_A = repmat([1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0], 12, 1);   % fRef = 1 GHz
mat_B = repmat([3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0], 12, 1);   % fRef = 2 GHz
[mat_A_ind, mat_A_st] = m2p(mat_A);
[mat_B_ind, mat_B_st] = m2p(mat_B);
% Use lod = 3 to exercise the full multi-path / multi-hit ray-state machine.
gain_A = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, mat_A_ind, mat_A_st, 10e9, 3 );
gain_B = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, mat_B_ind, mat_B_st, 10e9, 3 );
assertElementsAlmostEqual( gain_A, gain_B, 'absolute', 1e-12 );

% Error: too few inputs (5 args, missing center_freq)
try
    [~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: too many inputs (11 args)
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, [], 0, 0, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: 3 outputs
try
    [~, ~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 5 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Too many output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: wrong dest size
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest(1,:), cube, mtl_ind, mtl_st, 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: wrong mtl_prop columns (now expects 9)
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind(1), mtl_st, 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: wrong mtl_prop rows
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind(1), mtl_st, 1e9, 0 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: sub_mesh_index first element not 0
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 2);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'First sub-mesh must start at index 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Error: sub_mesh_index exceeds mesh count
try
    [~, ~] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, [1,33]);
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Sub-mesh indices cannot exceed number of faces.';
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
    mtl_ind(f) = hit;
end
st = struct();
for c = 1:9
    st.(names{c}) = uniq(:,c);
end
end