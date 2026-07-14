clear all
close all

no_pos  = 100e6;
cube = quadriga_lib.cube;

mtl_ind  = ones(12,1);       % all faces -> material 1 (1-based; 0 would mean "no material")
mtl_prop = struct('a', 1.5); % one material; each field is a length-n_mtl column (here n_mtl = 1)

orig(1,:) = [ -10.0,  0.0,   0.5 ]; dest(1,:) = [  10.0,  0.0,   0.5];
[ fbs, sbs, ~, fbs_ind, sbs_ind ] = quadriga_lib.ray_triangle_intersect( orig, dest, cube );

orig = repmat( orig, no_pos, 1 );
dest = repmat( dest, no_pos, 1 );
fbs = repmat( fbs, no_pos, 1 );
sbs = repmat( sbs, no_pos, 1 );
fbs_ind = repmat( fbs_ind, no_pos, 1 );
sbs_ind = repmat( sbs_ind, no_pos, 1 );

% Using trivec and tridir, reflection
trivec = repmat([0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.0, 0.2, 0.0],no_pos,1);
tridir = zeros(no_pos,6);
orig_length = ones(no_pos,1);

tic
[ origN, destN, gainN, xprmatN, trivecN, tridirN, orig_lengthN, fbs_angleN, thicknessN, edge_lengthN, normal_vecN]  = ...
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, cube, mtl_ind, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, orig_length );
toc
% 4.52 seconds, 16 cores (32 threads) @ 100 million rays







