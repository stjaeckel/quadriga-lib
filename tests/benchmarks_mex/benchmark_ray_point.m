clear all

no_ray  = 1e6;
no_pos  = 1e7;

% Generatea rays
no_div = round(sqrt(no_ray / 20));
[ orig, ~, trivec, tridir ] = quadriga_lib.icosphere( no_div, 1, 1 );
orig = orig + [0,0,2];

% Generate points
res    = 20/round(sqrt(no_pos));
rx_pos = [ -10, -10, 0.1 ]';  % Lower left point
rx_xy  = [ 20, 20 ]';                % x-y scale

x = rx_pos(1) + ( 0 : res : rx_xy(1) );
y = rx_pos(2) + ( 0 : res : rx_xy(2) );
[X,Y] = meshgrid(x,y);
points = [ X(:), Y(:),  ones(numel(X),1)*rx_pos(3) ];


target_size = 1024; 

tic
[pointsR, sub_cloud_index, forward_index, reverse_index ] = quadriga_lib.point_cloud_segmentation( points, target_size, 8 );
toc

tic
[ hit_count, ray_ind ] = quadriga_lib.ray_point_intersect( orig, trivec, tridir, pointsR, sub_cloud_index );
toc

h = reshape( ray_ind(1,reverse_index), numel(y), numel(x) );
imagesc(y,x,h);

