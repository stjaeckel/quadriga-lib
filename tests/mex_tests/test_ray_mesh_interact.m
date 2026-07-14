function test_ray_mesh_interact

% Shared 2-ray, air-filled cube setup (x-z plane, normal incidence on the x-walls)
mesh = quadriga_lib.cube;

mtl_prop = repmat([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);  % Air, fRef = 1 GHz
[mtl_ind, mtl_st] = m2p(mtl_prop);

orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];   % FBS West (x=-1), SBS East (x=+1)
orig(2,:) = [  10.0,  0.0,  -0.5 ]; dest(2,:) = [ -10.0,  0.0,  -0.5];   % FBS East (x=+1), SBS West (x=-1)

% fbs/sbs points are no longer inputs; we still need the face indices from ray_triangle_intersect,
% and we keep the points to cross-check the engine-computed fbsN/sbsN below.
[ fbs, sbs, ~, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( orig, dest, mesh );

% Call without outputs - should be fine
quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );

% Reflection sweep: outputs 1..13 in one call (no ray tube -> trivecN/tridirN empty)
[ origN, destN, fbsN, sbsN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, ...
    fbs_angleN, thicknessN, edge_lengthN, normal_vecN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );

% origN sits ON the face now (relaunch offset reduced to a few ULP, was 1 mm)
assertElementsAlmostEqual( origN, [-1.0, 0, 0.5; 1.0, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( destN, [-12, 0, 0.5; 12, 0, -0.5], 'absolute', 5e-6 );
% New engine-computed interaction points reproduce ray_triangle_intersect
assertElementsAlmostEqual( fbsN, fbs, 'absolute', 1e-6 );
assertElementsAlmostEqual( sbsN, sbs, 'absolute', 1e-6 );
assertElementsAlmostEqual( gainN, [0;0], 'absolute', 1e-6 );        % air reflects nothing
assertEqual( size(xprmatN), [8 2] );                               % xprmat is now [8, n_rayN]
assertElementsAlmostEqual( xprmatN, zeros(8,2), 'absolute', 1e-6 );
assertTrue( isempty(trivecN) );
assertTrue( isempty(tridirN) );
assertElementsAlmostEqual( orig_lengthN, [9;9], 'absolute', 1e-6 );  % orig->FBS, no offset (was 9.001)
assertElementsAlmostEqual( fbs_angleN, [pi/2;pi/2], 'absolute', 1e-6 );
assertElementsAlmostEqual( thicknessN, [2;2], 'absolute', 1e-6 );
assertEqual( size(edge_lengthN), [2 1] );                          % value checked in the beam test below
assertElementsAlmostEqual( normal_vecN, [-1,0,0,1,0,0 ; 1,0,0,-1,0,0], 'absolute', 1e-6 );

% Transmission sweep with custom orig_length (trivec/tridir/orig_length are args 10/11/12)
[ origN, destN, ~, ~, gainN, xprmatN, trivecN, tridirN, orig_lengthN, ...
    fbs_angleN, thicknessN, ~, normal_vecN ] = ...
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], [], [5;3] );
assertElementsAlmostEqual( origN, [-1.0, 0, 0.5; 1.0, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( destN, dest, 'absolute', 5e-6 );
assertElementsAlmostEqual( gainN, [1;1], 'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN, [1 1; 0 0; 0 0; 0 0; 0 0; 0 0; 1 1; 0 0], 'absolute', 1e-6 );
assertTrue( isempty(trivecN) );
assertTrue( isempty(tridirN) );
assertElementsAlmostEqual( orig_lengthN, [14;12], 'absolute', 1e-6 );  % 9+5, 9+3 (was 14.001/12.001)
assertElementsAlmostEqual( fbs_angleN, [pi/2;pi/2], 'absolute', 1e-6 );
assertElementsAlmostEqual( thicknessN, [2;2], 'absolute', 1e-6 );
assertElementsAlmostEqual( normal_vecN, [-1,0,0,1,0,0 ; 1,0,0,-1,0,0], 'absolute', 1e-6 );

% Ray-tube reflection (single ray, spherical tridir), geometry matched to the Catch2 oracle
orig_b = [ -10.0, 0.0, 0.5 ]; dest_b = [ 10.0, 0.0, 0.5 ];
[ ~, ~, ~, fbs_ind_b, sbs_ind_b ] = quadriga_lib.ray_triangle_intersect( orig_b, dest_b, mesh );
orig_length_b = 2.7;
trivec_b = [0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0];
tridir_sph = zeros(1,6);
tridir_sph(2) = 1*pi/180;   % v1 elevation
tridir_sph(5) = 1*pi/180;   % v3 azimuth

[ origN, destN, ~, ~, gainN, xprmatN, trivecN, tridir_sphN, orig_lengthN, ~, ~, edge_lengthN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_b, dest_b, mesh, mtl_ind, mtl_st, fbs_ind_b, sbs_ind_b, ...
    trivec_b, tridir_sph, orig_length_b );

a = tan(1*pi/180) * 9.0 + 0.2;
assertElementsAlmostEqual( origN, [-1.0, 0, 0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( destN, [-12, 0, 0.5], 'absolute', 5e-6 );
assertElementsAlmostEqual( gainN, 0, 'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN, zeros(8,1), 'absolute', 1e-6 );
assertEqual( size(tridir_sphN,2), 6 );
assertElementsAlmostEqual( trivecN, [0,-0.1,a, 0,-0.1,-0.2, 0,a,0], 'absolute', 1e-6 );
assertElementsAlmostEqual( tridir_sphN, [180,1,180,0,179,0]*pi/180, 'absolute', 1e-6 );
assertElementsAlmostEqual( orig_lengthN, 2.7 + 9.0, 'absolute', 1e-6 );
assertElementsAlmostEqual( edge_lengthN, sqrt(a*a + (a+0.1)*(a+0.1)), 'absolute', 1e-6 );

% Same ray tube with a Cartesian (9-col) tridir -> tridirN comes back in 9-col Cartesian form
sphN = [180,1,180,0,179,0]*pi/180;
tridir_crt = [ cos(tridir_sph(2))*cos(tridir_sph(1)), cos(tridir_sph(2))*sin(tridir_sph(1)), sin(tridir_sph(2)), ...
               cos(tridir_sph(4))*cos(tridir_sph(3)), cos(tridir_sph(4))*sin(tridir_sph(3)), sin(tridir_sph(4)), ...
               cos(tridir_sph(6))*cos(tridir_sph(5)), cos(tridir_sph(6))*sin(tridir_sph(5)), sin(tridir_sph(6)) ];
[ ~, ~, ~, ~, ~, ~, ~, tridir_crtN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_b, dest_b, mesh, mtl_ind, mtl_st, fbs_ind_b, sbs_ind_b, ...
    trivec_b, tridir_crt, orig_length_b );
expc = [ cos(sphN(2))*cos(sphN(1)), cos(sphN(2))*sin(sphN(1)), sin(sphN(2)), ...
         cos(sphN(4))*cos(sphN(3)), cos(sphN(4))*sin(sphN(3)), sin(sphN(4)), ...
         cos(sphN(6))*cos(sphN(5)), cos(sphN(6))*sin(sphN(5)), sin(sphN(6)) ];
assertEqual( size(tridir_crtN,2), 9 );
assertElementsAlmostEqual( tridir_crtN, expc, 'absolute', 1e-6 );

% Dielectric Fresnel sanity (45 deg into West face, eps_r = 1.5). Confirms mtl_prop is read and the
% xprmat sign/layout and gain formulas are wired correctly.
mtl_d = repmat([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[mtl_d_ind, mtl_d_st] = m2p(mtl_d);
orig_d = [ -1.5, 0.0, 0.0 ]; dest_d = [ 0.0, 0.0, 1.5 ];
[ ~, ~, ~, fbs_ind_d, sbs_ind_d ] = quadriga_lib.ray_triangle_intersect( orig_d, dest_d, mesh );

cos_th = cos(45*pi/180); sin_th = sin(45*pi/180);
eps = 1.5;
Z = sqrt(eps - cos_th*cos_th);
R_par = (eps*sin_th - Z) / (eps*sin_th + Z);   % R_tm
R_per = (sin_th - Z) / (sin_th + Z);           % R_te

[ ~, ~, ~, ~, gainN, xprmatN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_d, dest_d, mesh, mtl_d_ind, mtl_d_st, fbs_ind_d, sbs_ind_d );
assertElementsAlmostEqual( gainN, 0.5*R_par^2 + 0.5*R_per^2, 'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN, [-R_par;0;0;0;0;0;-R_per;0], 'absolute', 1e-6 );  % 180 deg reflection flip

th2  = asin(sin(45*pi/180) / sqrt(eps));
T_te = 2*cos_th / (cos_th + sqrt(eps)*cos(th2));
T_tm = 2*cos_th / (sqrt(eps)*cos_th + cos(th2));
[ ~, ~, ~, ~, gainN ] = ...
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig_d, dest_d, mesh, mtl_d_ind, mtl_d_st, fbs_ind_d, sbs_ind_d );
assertElementsAlmostEqual( gainN, 0.5*T_te^2 + 0.5*T_tm^2, 'absolute', 1e-6 );

% In-medium absorption (alpha) no longer contributes to gainN; it moved to the (non-MEX) medium_gain
% helper. With real eps and sigma = 0, gainN is the lossless Fresnel reflection only.
mtl_alpha = repmat([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 1.0, 5.0], 12, 1);  % alpha = 4 dB/m @ 10 GHz
[mtl_alpha_ind, mtl_alpha_st] = m2p(mtl_alpha);
orig_a = [0.5, 0.1, 0.0]; dest_a = [2.0, 1.6, 0.0];   % inside, 45 deg on East face (eps 1.5 -> air)
[ ~, ~, ~, fbs_ind_a, sbs_ind_a ] = quadriga_lib.ray_triangle_intersect( orig_a, dest_a, mesh );
trivec_a = [0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0];
tridir_a = (pi/180) * [45.0, 0.0, 45.0, 0.0, 45.0, 0.0];
[ ~, ~, ~, ~, gainN ] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig_a, dest_a, mesh, ...
    mtl_alpha_ind, mtl_alpha_st, fbs_ind_a, sbs_ind_a, trivec_a, tridir_a );
eta1 = 1.5; eta2 = 1.0;
cos_th2 = sqrt(1 - (eta1/eta2) * sin(45*pi/180)^2);
n1 = sqrt(eta1); n2 = sqrt(eta2);
R_te = (n1*cos_th - n2*cos_th2) / (n1*cos_th + n2*cos_th2);
R_tm = (n2*cos_th - n1*cos_th2) / (n2*cos_th + n1*cos_th2);
assertElementsAlmostEqual( gainN, 0.5*(abs(R_te)^2 + abs(R_tm)^2), 'relative', 1e-9 );

% Interface (att) penetration loss still folds into transmission-class gain
%   att = 6 dB @ 2 GHz, exp = 1  ->  30 dB @ 10 GHz  ->  gain = 1e-3
mtl_att = repmat([1.0, 0.0, 0.0, 0.0, 6.0, 1.0, 0.0, 0.0, 2.0], 12, 1);
[mtl_att_ind, mtl_att_st] = m2p(mtl_att);
orig_p = [-2.0, 0.0, 0.5]; dest_p = [ 2.0, 0.0, 0.5];
[ ~, ~, ~, fbs_ind_p, sbs_ind_p ] = quadriga_lib.ray_triangle_intersect( orig_p, dest_p, mesh );
trivec_p = [0.0, -0.01, 0.01, 0.0, -0.01, -0.01, 0.0, 0.01, 0.0];
tridir_p = zeros(1,6);
[ ~, ~, ~, ~, gainN ] = quadriga_lib.ray_mesh_interact( 1, 10e9, orig_p, dest_p, mesh, ...
    mtl_att_ind, mtl_att_st, fbs_ind_p, sbs_ind_p, trivec_p, tridir_p );
assertElementsAlmostEqual( gainN, 1e-3, 'absolute', 1e-9 );

% fRef parameterization equivalence: same physical material at fRef = 1 GHz vs 2 GHz
mtl_A = repmat([2.0, 1.0, 0.01, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0], 12, 1);  % at 1 GHz
mtl_B = repmat([4.0, 1.0, 0.02, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0], 12, 1);  % at 2 GHz
[mtl_A_ind, mtl_A_st] = m2p(mtl_A);
[mtl_B_ind, mtl_B_st] = m2p(mtl_B);
orig_e = [-1.5, 0.0, 0.0]; dest_e = [ 0.0, 0.0, 1.5];
[ ~, ~, ~, fbs_ind_e, sbs_ind_e ] = quadriga_lib.ray_triangle_intersect( orig_e, dest_e, mesh );
trivec_e = [0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0];
tridir_e = (pi/180) * [0.0, 45.0, 0.0, 45.0, 0.0, 45.0];
[ oNa, dNa, ~, ~, gNa, xNa ] = quadriga_lib.ray_mesh_interact( 2, 10e9, orig_e, dest_e, mesh, ...
    mtl_A_ind, mtl_A_st, fbs_ind_e, sbs_ind_e, trivec_e, tridir_e );
[ oNb, dNb, ~, ~, gNb, xNb ] = quadriga_lib.ray_mesh_interact( 2, 10e9, orig_e, dest_e, mesh, ...
    mtl_B_ind, mtl_B_st, fbs_ind_e, sbs_ind_e, trivec_e, tridir_e );
assertElementsAlmostEqual( gNa, gNb, 'absolute', 1e-12 );
assertElementsAlmostEqual( xNa, xNb, 'absolute', 1e-12 );
assertElementsAlmostEqual( oNa, oNb, 'absolute', 1e-12 );
assertElementsAlmostEqual( dNa, dNb, 'absolute', 1e-12 );

% Scalar interactions (types 3/4/5): scalar xprmat is now [2, n_rayN] = [Re; Im]
% Air: reflection -> zero, transmission -> unity
[ origN, ~, ~, ~, gainN, xprmatN ] = ...
    quadriga_lib.ray_mesh_interact( 3, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( origN, [-1.0, 0, 0.5; 1.0, 0, -0.5], 'absolute', 1e-6 );
assertElementsAlmostEqual( gainN, [0;0], 'absolute', 1e-6 );
assertEqual( size(xprmatN), [2 2] );
assertElementsAlmostEqual( xprmatN, zeros(2,2), 'absolute', 1e-6 );

[ ~, destN, ~, ~, gainN, xprmatN ] = ...
    quadriga_lib.ray_mesh_interact( 4, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
assertElementsAlmostEqual( destN, dest, 'absolute', 5e-6 );
assertElementsAlmostEqual( gainN, [1;1], 'absolute', 1e-6 );
assertEqual( size(xprmatN), [2 2] );
assertElementsAlmostEqual( xprmatN, [1 1; 0 0], 'absolute', 1e-6 );

% Dielectric (inside -> air, 45 deg, eps 1.5): scalar R equals the EM TE coefficient
mtl_s = repmat([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[mtl_s_ind, mtl_s_st] = m2p(mtl_s);
orig_s = [0.5, 0.1, 0.0]; dest_s = [2.0, 1.6, 0.0];
[ ~, ~, ~, fbs_ind_s, sbs_ind_s ] = quadriga_lib.ray_triangle_intersect( orig_s, dest_s, mesh );
cos_th2s = sqrt(1 - 1.5*cos_th*cos_th);   % 0.5
R = (sqrt(1.5)*cos_th - cos_th2s) / (sqrt(1.5)*cos_th + cos_th2s);

[ ~, ~, ~, ~, gainN, xprmatN ] = ...
    quadriga_lib.ray_mesh_interact( 3, 10e9, orig_s, dest_s, mesh, mtl_s_ind, mtl_s_st, fbs_ind_s, sbs_ind_s );
assertElementsAlmostEqual( gainN, R^2, 'absolute', 1e-6 );
assertEqual( size(xprmatN), [2 1] );

[ ~, ~, ~, ~, gainN ] = ...
    quadriga_lib.ray_mesh_interact( 4, 10e9, orig_s, dest_s, mesh, mtl_s_ind, mtl_s_st, fbs_ind_s, sbs_ind_s );
assertElementsAlmostEqual( gainN, 1 - R^2, 'absolute', 1e-6 );

% Type 5 (scalar refraction): Snell-bent path, gain = |1+R|^2 (pressure-coefficient power).
% For a dense->rare interface this exceeds 1, mirroring the EM refraction coefficient.
[ ~, ~, ~, ~, gainN, xprmatN ] = ...
    quadriga_lib.ray_mesh_interact( 5, 10e9, orig_s, dest_s, mesh, mtl_s_ind, mtl_s_st, fbs_ind_s, sbs_ind_s );
assertEqual( size(xprmatN), [2 1] );
assertElementsAlmostEqual( gainN, (1+R)^2, 'absolute', 1e-6 );
assertElementsAlmostEqual( xprmatN(1)^2 + xprmatN(2)^2, gainN, 'absolute', 1e-6 );  % coeff energy == gain

% Transmission factor (tf/tfB optional columns) splits reflection/transmission energy
[mtl_tf_ind, mtl_tf_st] = m2p(mtl_s);
mtl_tf_st.tf  = 0.5 * ones(numel(mtl_tf_st.a), 1);
mtl_tf_st.tfB = zeros(numel(mtl_tf_st.a), 1);
R_te = (sqrt(1.5)*cos_th - cos_th2s) / (sqrt(1.5)*cos_th + cos_th2s);
R_tm = (cos_th - sqrt(1.5)*cos_th2s) / (cos_th + sqrt(1.5)*cos_th2s);
reflectance = 0.5*(R_te^2 + R_tm^2);
[ ~, ~, ~, ~, rP ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_s, dest_s, mesh, mtl_tf_ind, mtl_tf_st, fbs_ind_s, sbs_ind_s );
[ ~, ~, ~, ~, tP ] = ...
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig_s, dest_s, mesh, mtl_tf_ind, mtl_tf_st, fbs_ind_s, sbs_ind_s );
assertElementsAlmostEqual( rP, reflectance*0.5,     'absolute', 1e-6 );
assertElementsAlmostEqual( tP, 1 - reflectance*0.5, 'absolute', 1e-6 );
assertElementsAlmostEqual( rP + tP, 1,              'absolute', 1e-6 );

% compact flag + ray_indN: 3 rays, ray 2 (offset in y) misses the cube
mtl_c = repmat([1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);
[mtl_c_ind, mtl_c_st] = m2p(mtl_c);
orig_c = [-1.5, 0.0, 0.0; -1.5, 3.0, 0.0; -1.5, 0.5, 0.5];
dest_c = [ 0.0, 0.0, 0.0;  0.0, 3.0, 0.0;  0.0, 0.5, 0.5];
[ ~, ~, ~, fbs_ind_c, sbs_ind_c ] = quadriga_lib.ray_triangle_intersect( orig_c, dest_c, mesh );
assertTrue( fbs_ind_c(2) == 0 );   % ray 2 really misses

% compact = true (default): missed ray dropped, ray_indN maps survivors to 1-based input indices
[ origN, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, out_typeN, ~, ray_indN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_c, dest_c, mesh, mtl_c_ind, mtl_c_st, fbs_ind_c, sbs_ind_c );
assertEqual( size(origN,1), 2 );
assertEqual( double(ray_indN(:)), [1;3] );

% compact = false: all rays retained in input order; the missed ray passes through untouched
[ origN, ~, ~, ~, gainN, xprmatN, ~, ~, ~, ~, ~, ~, ~, out_typeN, ~, ray_indN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_c, dest_c, mesh, mtl_c_ind, mtl_c_st, ...
    fbs_ind_c, sbs_ind_c, [], [], [], false );
assertEqual( size(origN,1), 3 );
assertEqual( numel(gainN), 3 );
assertEqual( size(xprmatN), [8 3] );
assertEqual( double(ray_indN(:)), [1;2;3] );
assertElementsAlmostEqual( gainN(2), 1, 'absolute', 1e-9 );              % miss -> transparent
assertEqual( double(out_typeN(2)), 0 );                                 % no valid interaction
assertElementsAlmostEqual( xprmatN(:,2), [1;0;0;0;0;0;1;0], 'absolute', 1e-9 );  % identity column
assertTrue( gainN(1) < 1 && gainN(3) < 1 );                             % survivors actually interacted
assertTrue( bitand(double(out_typeN(1)),1) ~= 0 && bitand(double(out_typeN(3)),1) ~= 0 );

% path_dirN direction contract (45 deg into West face, eps 1.5)
orig_pd = [-1.5, 0.0, 0.0]; dest_pd = [0.0, 0.0, 1.5];
[ ~, ~, ~, fbs_ind_pd, sbs_ind_pd ] = quadriga_lib.ray_triangle_intersect( orig_pd, dest_pd, mesh );
incoming = (dest_pd - orig_pd) ./ norm(dest_pd - orig_pd);

[ o0, d0, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, pd0 ] = ...   % reflection -> mirror
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_pd, dest_pd, mesh, mtl_d_ind, mtl_d_st, fbs_ind_pd, sbs_ind_pd );
assertElementsAlmostEqual( norm(pd0), 1, 'absolute', 1e-6 );
assertElementsAlmostEqual( pd0, (d0-o0)./norm(d0-o0), 'absolute', 1e-6 );

[ o2, d2, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, pd2 ] = ...   % refraction -> Snell
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig_pd, dest_pd, mesh, mtl_d_ind, mtl_d_st, fbs_ind_pd, sbs_ind_pd );
assertElementsAlmostEqual( norm(pd2), 1, 'absolute', 1e-6 );
assertElementsAlmostEqual( pd2, (d2-o2)./norm(d2-o2), 'absolute', 1e-6 );

[ o1, d1, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, pd1 ] = ...   % undeviated transmission
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig_pd, dest_pd, mesh, mtl_d_ind, mtl_d_st, fbs_ind_pd, sbs_ind_pd );
assertElementsAlmostEqual( norm(pd1), 1, 'absolute', 1e-6 );
assertElementsAlmostEqual( pd1, pd2, 'absolute', 1e-6 );                       % path_dir is the Snell direction
assertElementsAlmostEqual( (d1-o1)./norm(d1-o1), incoming, 'absolute', 1e-6 ); % geometry stays undeviated
assertTrue( norm(pd1 - incoming) > 1e-2 );                                     % and differs from incoming

% out_typeN bit encoding (uint32): entry = 3, exit = 1, exit|TIR = 33
[ ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, out_typeN ] = ...   % air -> dielectric, front face -> entry
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig_d, dest_d, mesh, mtl_d_ind, mtl_d_st, fbs_ind_d, sbs_ind_d );
assertEqual( class(out_typeN), 'uint32' );
assertEqual( double(out_typeN(1)), 3 );

[ ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, out_typeN ] = ...   % inside dielectric -> air, no TIR -> exit
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig_s, dest_s, mesh, mtl_s_ind, mtl_s_st, fbs_ind_s, sbs_ind_s );
assertEqual( double(out_typeN(1)), 1 );
assertEqual( bitand(double(out_typeN(1)),32), 0 );

mtl_tir = repmat([2.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], 12, 1);   % eps 2.5, 45 deg -> TIR
[mtl_tir_ind, mtl_tir_st] = m2p(mtl_tir);
[ ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, out_typeN ] = ...
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig_s, dest_s, mesh, mtl_tir_ind, mtl_tir_st, fbs_ind_s, sbs_ind_s );
assertEqual( double(out_typeN(1)), 33 );
assertTrue( bitand(double(out_typeN(1)),32) ~= 0 );

% Build a valid ray tube for the ray-tube error checks below
trivec = repmat([0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0], 2, 1);
tridir = zeros(2,6);

% ---- Error handling ----

try % 8 inputs (too few, minimum is 9)
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 14 inputs (too many, maximum is 13)
    quadriga_lib.ray_mesh_interact( 1, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], [], [5;3], true, 99 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % 17 outputs (too many, maximum is 16)
    [~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~,~] = quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong type (valid range is now 0..5)
    quadriga_lib.ray_mesh_interact( 6, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Interaction type must be';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong freq
    quadriga_lib.ray_mesh_interact( 2, -1, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Center frequency must be provided in Hertz and have values > 0.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig (row mismatch)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig(1,:), dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig (columns)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig(:,1), dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''orig'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest (row mismatch)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest(1,:), mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''dest'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong dest (columns)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest(:,1), mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''dest'' must have 3 columns containing x,y,z coordinates.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong mesh (columns)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh(:,1), mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''mesh'' must have 9 columns containing x,y,z coordinates of 3 vertices.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % mesh/mtl_ind length mismatch (via short mesh)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh(1,:), mtl_ind, mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % mtl_ind length mismatch (via short mtl_ind)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind(1), mtl_st, fbs_ind, sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Length of ''mtl_ind'' must match the number of mesh faces.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong fbs_ind (length)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind(1), sbs_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''fbs_ind'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong sbs_ind (length)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind(1) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of elements in ''sbs_ind'' does not match number of rows in ''orig''.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % missing tridir
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'In order to use ray tubes, both ''trivec'' and ''tridir'' must be given.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % missing trivec
    quadriga_lib.ray_mesh_interact( 0, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'In order to use ray tubes, both ''trivec'' and ''tridir'' must be given.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong trivec (row mismatch)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec(1,:), tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''trivec'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong trivec (columns)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec(:,1), tridir );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''trivec'' must have 9 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong tridir (row mismatch)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec, tridir(1,:) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Number of rows in ''orig'' and ''tridir'' dont match.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong tridir (columns)
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, trivec, tridir(:,1) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Input ''tridir'' must have 6 or 9 columns.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

try % wrong orig_length
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, mesh, mtl_ind, mtl_st, fbs_ind, sbs_ind, [], [], 1 );
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