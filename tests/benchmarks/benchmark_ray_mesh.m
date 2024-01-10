clear all
close all

no_mesh = 1e5;
no_pos  = 1e5;

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

% Plane
cube(:,[1,2,4,5,7,8]) = cube(:,[1,2,4,5,7,8]) * 100;

% Subdivide
mesh = quadriga_lib.subdivide_triangles(cube, round(sqrt(no_mesh/12)));

x_min = -99; x_max = 99; y_min = -99; y_max = 99;   % Map edges (set by model size)
pixel_size = 198 / round(sqrt(no_pos));                 % in [m]

% Generate a grid of receiver positions
x = x_min : pixel_size : x_max;
y = y_min : pixel_size : y_max;
[X,Y] = meshgrid(x,y);
orig = [X(:), Y(:), ones(numel(X),1)*20];
dest = [X(:), Y(:), -ones(numel(X),1)*20];

%orig(:,[1,2]) = 0;

origS = single(orig);
destS = single(dest);
meshS = single(mesh);

tic
[ fbs, sbs, no_hit, ifbs, isbs ] = quadriga_lib.ray_triangle_intersect( origS, destS, meshS );
toc

