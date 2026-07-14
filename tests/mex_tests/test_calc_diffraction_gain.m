function test_calc_diffraction_gain
% MEX integration tests for quadriga_lib.calc_diffraction_gain.
% Scope: MEX interface plumbing, argument/output handling, and error paths,
% plus light physics sanity. Full physics validation lives in the Catch2 suite
% (test_calc_diffraction_gain.cpp) and is intentionally not duplicated here.

cube = quadriga_lib.cube;   % 2x2x2 box at origin (+/-1), 12 triangles

% Base material via m2p (9-col struct): att = 3 dB at fRef = 1 GHz, eps_r = 1
% (no Fresnel), no other loss and no frequency scaling.
mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[mtl_ind, mtl_st] = m2p(mtl_prop);

% Two straight-through paths (enter west wall, exit east wall). att is charged
% once per body on the entry face, so each path sees a single 3 dB loss.
orig(1,:) = [ -10.0,  0.0,  0.5 ]; dest(1,:) = [  10.0,  0.0,  0.5 ];
orig(2,:) = [  10.0,  0.0, -0.5 ]; dest(2,:) = [ -10.0,  0.0, -0.5 ];

% Reused single-path geometries.
orig_in    = [ -10.0, 0.0, 0.5 ];   % enters west wall, dest inside -> one entry face
dest_in    = [   0.5, 0.0, 0.5 ];
orig_clear = [ -10.0, 0.0, 0.5 ];   % entirely left of the cube -> no interaction
dest_clear = [  -5.0, 0.0, 0.5 ];

%% Output plumbing

% 0 outputs is valid (6 args).
quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9 );

% 1 output, lod = 2.
gain_only = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2 );
assertTrue( numel(gain_only) == 2 );

% Basic gain, lod = 0, explicit verbose. Per-body att -> 3 dB per path.
gain = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0 );
assertElementsAlmostEqual( gain, [10^(-0.3); 10^(-0.3)], 'absolute', 1e-10 );

% 2 outputs -> gain + xprmat (EM: [8, n_pos]).
[gain2, xpr2] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2 );
assertElementsAlmostEqual( gain2, gain_only, 'absolute', 1e-12 );
assertTrue( isequal( size(xpr2), [8, 2] ) );

% 3 outputs -> gain + xprmat + coord. lod = 5 -> n_seg = 1 (path midpoints).
[gain5, xpr5, coord5] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 5 );
assertElementsAlmostEqual( gain5, [10^(-0.3); 10^(-0.3)], 'absolute', 1e-6 );
assertTrue( isequal( size(xpr5), [8, 2] ) );
assertElementsAlmostEqual( coord5, permute([0,0 ; 0,0 ; 0.5,-0.5],[1,3,2]), 'absolute', 1e-10 );

%% Input casting & acceleration passthrough

% Single-precision inputs are cast to double internally.
gain_single = quadriga_lib.calc_diffraction_gain( single(orig), single(dest), ...
    single(cube), mtl_ind, mtl_st, 1e9, 2 );
assertElementsAlmostEqual( gain_only, gain_single, 'absolute', 1e-5 );

% Empty and single-entry sub-mesh index run without error.
quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, [] );
quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 1 );

% Non-uint32 numeric sub_mesh_index (typecast path) gives identical result.
gain_smi = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 1 );
assertElementsAlmostEqual( gain, gain_smi, 'absolute', 1e-10 );

% use_kernel = 1 (GENERIC) matches the default (auto) kernel.
gain_generic = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2, 0, [], 1 );
assertElementsAlmostEqual( gain_only, gain_generic, 'absolute', 1e-10 );

% Full acceleration args (use_kernel + gpu_id) match the default.
gain_full = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 2, 0, [], 1, 0 );
assertElementsAlmostEqual( gain_only, gain_full, 'absolute', 1e-10 );

%% coord dimensions per lod

% coord is [3, n_seg, n_pos]; xprmat (checked above) is [8, n_pos].
% n_seg = 2 (lod 1,2), 3 (lod 3), 4 (lod 4), 1 (lod 5,6).
lods   = [1 2 3 4 5 6];
n_segs = [2 2 3 4 1 1];
for k = 1:numel(lods)
    [~, ~, ck] = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, lods(k) );
    assertTrue( isequal( size(ck), [3, n_segs(k), 2] ), ...
        sprintf('coord dimension mismatch at lod %d', lods(k)) );
end

%% xprmat sanity

% Clear path (entirely outside) -> gain 1 and identity Jones matrix. Row layout
% is col-major 2x2: [ReVV ImVV ReHV ImHV ReVH ImVH ReHH ImHH].
[g_clr, xpr_clr] = quadriga_lib.calc_diffraction_gain( orig_clear, dest_clear, cube, mtl_ind, mtl_st, 10e9, 0 );
assertElementsAlmostEqual( g_clr, 1.0, 'absolute', 1e-12 );
assertTrue( isequal( size(xpr_clr), [8, 1] ) );
assertElementsAlmostEqual( xpr_clr, [1;0; 0;0; 0;0; 1;0], 'absolute', 1e-12 );

% Normal-incidence transmission into a lossless dielectric (eps_r = 2): real
% Fresnel loss (gain < 1) but TE == TM, so the normalized Jones matrix is still
% identity and the power normalization 0.5*sum|xpr|^2 == 1 holds.
mtl_eps2 = repmat([2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[eps2_ind, eps2_st] = m2p(mtl_eps2);
[g_e2, xpr_e2] = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, eps2_ind, eps2_st, 10e9, 0 );
R0 = (1 - sqrt(2)) / (1 + sqrt(2));
assertElementsAlmostEqual( g_e2, 1 - R0^2, 'absolute', 1e-9 );
assertElementsAlmostEqual( abs(xpr_e2(1) + 1i*xpr_e2(2)), 1.0, 'absolute', 1e-9 );  % |VV|
assertElementsAlmostEqual( abs(xpr_e2(7) + 1i*xpr_e2(8)), 1.0, 'absolute', 1e-9 );  % |HH|
assertElementsAlmostEqual( xpr_e2(3:6), zeros(4,1), 'absolute', 1e-9 );             % off-diagonals
assertElementsAlmostEqual( 0.5*sum(xpr_e2.^2), 1.0, 'absolute', 1e-9 );             % normalization

%% scalar_mode

% Scalar transmission (arg 12 = true). At normal incidence TE == TM, so scalar
% equals EM. xprmat collapses to [2, n_pos].
[g_sc, xpr_sc] = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, eps2_ind, eps2_st, 10e9, 0, 0, [], 0, 0, true );
assertElementsAlmostEqual( g_sc, g_e2, 'absolute', 1e-9 );   % scalar == EM at normal incidence
assertTrue( isequal( size(xpr_sc), [2, 1] ) );

% Scalar clear path -> gain 1, coefficient (1,0).
[g_sc_c, xpr_sc_c] = quadriga_lib.calc_diffraction_gain( orig_clear, dest_clear, cube, eps2_ind, eps2_st, 10e9, 0, 0, [], 0, 0, true );
assertElementsAlmostEqual( g_sc_c, 1.0, 'absolute', 1e-12 );
assertTrue( isequal( size(xpr_sc_c), [2, 1] ) );
assertElementsAlmostEqual( xpr_sc_c, [1;0], 'absolute', 1e-12 );

%% Material index 0 (no material)

% Index 0 means "no material": the face is intersected but applies no transition.
% All-zero indices -> fully transparent (gain 1). This exercises the wrapper
% passing mtl_ind through 1-based with no decrement.
mtl_ind_zero = zeros(12, 1, 'uint64');
gain_zero = quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind_zero, mtl_st, 10e9, 0 );
assertElementsAlmostEqual( gain_zero, [1.0; 1.0], 'absolute', 1e-12 );

% Same geometry, real lossy material (att = 6 dB) clearly attenuates.
mtl_lossy = repmat([1.0, 0.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[lossy_ind, lossy_st] = m2p(mtl_lossy);
gain_lossy = quadriga_lib.calc_diffraction_gain( orig, dest, cube, lossy_ind, lossy_st, 10e9, 0 );
assertTrue( all( gain_lossy < 0.5 ) );

%% Physics sanity

% LOS (unobstructed) path above the cube -> gain ~ 1.
gain_los = quadriga_lib.calc_diffraction_gain( [0,0,5], [0,0,10], cube, mtl_ind, mtl_st, 1e9, 2 );
assertElementsAlmostEqual( gain_los, 1.0, 'absolute', 1e-6 );

% Alpha (in-medium distance absorption): eps_r = 1, alpha = 4 dB/m. Path enters
% at x = -1 and ends at x = 0.5 -> 1.5 m in medium -> 6 dB -> 10^(-0.6).
mtl_alpha = repmat([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0, 1.0], 12, 1);
[alpha_ind, alpha_st] = m2p(mtl_alpha);
gain_alpha = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, alpha_ind, alpha_st, 10e9, 0 );
assertElementsAlmostEqual( gain_alpha, 10^(-0.6), 'absolute', 1e-7 );

% Penetration loss frequency scaling: att = 3 dB at fRef = 2 GHz, attB = 1.
% At 10 GHz -> 3*(10/2)^1 = 15 dB -> 10^(-1.5).
mtl_attB = repmat([1.0, 0.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0], 12, 1);
[attB_ind, attB_st] = m2p(mtl_attB);
gain_attB = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, attB_ind, attB_st, 10e9, 0 );
assertElementsAlmostEqual( gain_attB, 10^(-1.5), 'absolute', 1e-10 );

% fRef parameterization equivalence: two materials specified at different
% reference frequencies but numerically identical at every frequency must give
% identical gain. lod = 3 exercises the multi-arc / multi-hit ray-state machine.
mat_A = repmat([1.5, 1.0, 0.001, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0], 12, 1);   % fRef = 1 GHz
mat_B = repmat([3.0, 1.0, 0.002, 1.0, 4.0, 1.0, 1.0, 1.0, 2.0], 12, 1);   % fRef = 2 GHz
[matA_ind, matA_st] = m2p(mat_A);
[matB_ind, matB_st] = m2p(mat_B);
gain_A = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, matA_ind, matA_st, 10e9, 3 );
gain_B = quadriga_lib.calc_diffraction_gain( orig_in, dest_in, cube, matB_ind, matB_st, 10e9, 3 );
assertElementsAlmostEqual( gain_A, gain_B, 'absolute', 1e-12 );

%% thin_slab_threshold (Fabry-Perot plumbing)

% Thin lossless dielectric slab tuned to half-wave optical thickness at 10 GHz.
% With resolution on (threshold = 0) the internal interference is kept; with
% resolution off (threshold = 1) it is discarded. The two must differ, and the
% half-wave resonance transmits more than the incoherent result. Exact Airy
% transmittance values are covered in the Catch2 suite.
f = 10e9;
n = 1.5;                           % eps_r = 2.25
t_half = 299792458 / f / (2 * n);  % half-wave thickness (~1 cm)
slab = quadriga_lib.cube( [t_half/2, 5, 5], [], [t_half/2, 0, 2.001] );  % x in [0, t_half]
mtl_slab = repmat([n^2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[slab_ind, slab_st] = m2p(mtl_slab);
orig_fp = [ -10.0, 0.0, 0.0 ];
dest_fp = [  10.0, 0.0, 0.0 ];     % normal incidence through both faces
g_on  = quadriga_lib.calc_diffraction_gain( orig_fp, dest_fp, slab, slab_ind, slab_st, f, 0, 0, [], 0, 0, false, 0.0 );
g_off = quadriga_lib.calc_diffraction_gain( orig_fp, dest_fp, slab, slab_ind, slab_st, f, 0, 0, [], 0, 0, false, 1.0 );
assertTrue( g_on  > 0 && g_on  <= 1 + 1e-9 );
assertTrue( g_off > 0 && g_off <= 1 + 1e-9 );
assertTrue( g_on > g_off + 1e-3 );   % half-wave resonance enhances transmission

%% Error handling

% MEX-layer guards (exact strings from the wrapper).
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st ), ...
    'Wrong number of input arguments.' );                                          % too few (5 args)
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, [], 0, 0, false, 0, 0 ), ...
    'Wrong number of input arguments.' );                                          % too many (14 args)
assert_throws( @() out4( orig, dest, cube, mtl_ind, mtl_st, 1e9, 5 ), ...
    'Too many output arguments.' );                                                % 4 outputs
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 0 ), ...
    'Entries in ''sub_mesh_index'' cannot be 0 (1-based index).' );                % sub_mesh entry 0

% Library-layer guards. Strings marked (guess) are best-effort and may need
% updating to match the refactored library messages.
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest(1,:), cube, mtl_ind, mtl_st, 1e9, 0 ), ...
    'Number of rows in ''orig'' and ''dest'' dont match.' );
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind(1), mtl_st, 1e9, 0 ), ...
    'Length of ''mtl_ind'' must match the number of mesh faces.' );
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, 2 ), ...
    'First sub-mesh must start at index 0.' );
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 1e9, 0, 0, [1,33] ), ...
    'Sub-mesh indices cannot exceed number of faces.' );
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig(:,1:2), dest, cube, mtl_ind, mtl_st, 1e9, 0 ), ...
    'orig' );                                                                      % (guess) orig not 3 columns
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest(:,1:2), cube, mtl_ind, mtl_st, 1e9, 0 ), ...
    'dest' );                                                                      % (guess) dest not 3 columns
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube(:,1:8), mtl_ind, mtl_st, 1e9, 0 ), ...
    'mesh' );                                                                      % (guess) mesh not 9 columns
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind, mtl_st, 0, 0 ), ...
    'frequency' );                                                                 % (guess) center_freq <= 0
mtl_ind_big = mtl_ind; mtl_ind_big(1) = 40000;
assert_throws( @() quadriga_lib.calc_diffraction_gain( orig, dest, cube, mtl_ind_big, mtl_st, 1e9, 0 ), ...
    '32767' );                                                                     % (guess) material index > 32767

end

% Request 4 outputs so the nlhs > 3 guard fires (cannot be done from an
% anonymous function, which requests at most one output).
function out4(varargin)
[~, ~, ~, ~] = quadriga_lib.calc_diffraction_gain( varargin{:} ); %#ok<ASGLU>
end

% Assert that fn() raises an error whose message contains expected (substring).
function assert_throws(fn, expected)
try
    fn();
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expected)) %#ok<STREMP>
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expected, '", GOT: "', ME.message, '"']);
    end
end
end

% Convert a per-face [n_face, 9] material matrix with columns
% {a,b,c,d,att,attB,alpha,alphaB,fRef} into the (mtl_ind, struct) pair the API
% expects. Identical rows are deduplicated; mtl_ind is 1-based.
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