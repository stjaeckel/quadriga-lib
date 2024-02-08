clear all
close all
clc

res = 0.02;
no_pos  = 1e5;
scl = 10;

lod = 5;

s1  = 1;
xo1 = 0  ;
yo1 = 4;

s2  = 2.5;
xo2 = -1;
yo2 = 2;

[cube, ~, vert, face] = quadriga_lib.obj_file_read('cube.obj');
cube2 = s1*cube;
cube2(:,[1,4,7]) = cube2(:,[1,4,7]) + xo1;
cube2(:,[2,5,8]) = cube2(:,[2,5,8]) + yo1;
cube3 = s2*cube;
cube3(:,[1,4,7]) = cube3(:,[1,4,7]) + xo2;
cube3(:,[2,5,8]) = cube3(:,[2,5,8]) + yo2;
msh = [ cube; cube2; cube3 ];

mtl_prop = repmat([1.0, 0.0, 0.0003, 0.0, 1.0],12,1);
mtl_prop = [ mtl_prop ; repmat([1.0, 0.0, 0.0001, 0.0, 0.0],12,1) ];
mtl_prop = [ mtl_prop ; repmat([1.0, 0.0, 0.0001, 0.0, 1.0],12,1) ];

x = -scl : res : scl;
y = -scl : res : scl;
[X,Y] = meshgrid(x,y);
dest = [ X(:), Y(:),  ones(numel(X),1)*0 ];

orig = [ -2, 5, 0.1 ];
orig = repmat( orig, size(dest,1), 1 );

tic
gain = quadriga_lib.calc_diffraction_gain( orig, dest, msh, mtl_prop, 1e9, lod, 1 );
toc

gainX = quadriga_lib.calc_diffraction_gain( orig(1,:), [4.3, -3.4, 0], msh, mtl_prop, 1e9, lod,2 );



gain = reshape(gain,numel(y),[]);

han = figure('Position',[ 100 , 100 , 1000 , 700]);
title('Gain')
patch ("Faces", face, "Vertices", vert, 'FaceColor', [0.7;0.7;0.7], 'EdgeColor', [0.3;0.3;0.3]/2  );
hold on
plot3(orig(1,1),orig(1,2),orig(1,3),'xr','Markersize',12)
patch ("Faces", face, "Vertices", s1*vert + [xo1,yo1,0], 'FaceColor', [0.7;0.7;0.7], 'EdgeColor', [0.3;0.3;0.3]/2  );
patch ("Faces", face, "Vertices", s2*vert + [xo2,yo2,0], 'FaceColor', [0.7;0.7;0.7], 'EdgeColor', [0.3;0.3;0.3]/2  );
alpha(0.0)
axis([-1 1 -1 1 -1 1]*scl)
grid on
imagesc(x,y,gain);
hold off
view([0,90])
clim([0,1])
axis equal
colorbar
%view(-10,50)

%print(['work_in_progress.png'],'-dpng');

