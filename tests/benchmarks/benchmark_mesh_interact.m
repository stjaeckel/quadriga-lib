clear all
close all

no_pos  = 100e6;

% Cube
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

mtl_prop = repmat([1.5, 0.0, 0.0, 0.0, 0.0],12,1); % Air

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
    quadriga_lib.ray_mesh_interact( 2, 10e9, orig, dest, fbs, sbs, cube, mtl_prop, fbs_ind, sbs_ind, trivec, tridir, orig_length );
toc
% 35 seconds, single core @ 100 million rays

clear all




