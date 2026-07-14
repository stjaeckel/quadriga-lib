clear all
close all
clc

probe = [-5.9, -4, 0];

slab_threshold = 0.0;
lod = 0;

% BUild scene

mtl_prop = struct();
mtl_prop.a   = [ 6;    3.1;    1.0    ];
mtl_prop.c   = [ 0.000; 0.000; 0.0001 ];
mtl_prop.att = [ 0.0;    0.0;    1.0    ];

switch 6
    case 1 % Single dielectric slab
        msh = quadriga_lib.cube( [5.0001, 1, 5], [], [0,5,0] );
        mtl_ind = [ 1*ones(12,1) ];

    case 2 % Two slabs
        msh = [ quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,5.5,0] );
                quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,4.5,0] ) ];
        mtl_ind = [ 1*ones(12,1), 2*ones(12,1) ];

    case 3 % Overlapping slabs
        msh = [ quadriga_lib.cube( [5.0001, 0.6, 5], [], [0,5.4,0] );
                quadriga_lib.cube( [5.0001, 0.6, 5], [], [0,4.6,0] ) ];
        mtl_ind = [ 1*ones(12,1), 2*ones(12,1) ];

    case 4 % Duplicate mesh
        msh = [ quadriga_lib.cube( [5.0001, 1, 5], [], [0,5,0] );
                quadriga_lib.cube( [5.0001, 1, 5], [], [0,5,0] ) ];
        mtl_ind = [ 1*ones(12,1), 2*ones(12,1) ];

    case 5 % Overlapping entry
        msh = [ quadriga_lib.cube( [5.0001, 1, 5], [], [0,5,0] );
                quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,5.5,0] ) ];
        mtl_ind = [ 1*ones(12,1), 2*ones(12,1) ];

    case 6 % Overlapping exit
        msh = [ quadriga_lib.cube( [5.0001, 1, 5], [], [0,5,0] );
                quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,4.5,0] ) ];
        mtl_ind = [ 1*ones(12,1), 2*ones(12,1) ]; 

    case 7 % 3 slabs
             msh = [ quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,5.5,0] );
                quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,4.5,0] );
                quadriga_lib.cube( [5.0001, 0.5, 5], [], [0,5.0,0] ) ];
        mtl_ind = [ 1*ones(12,1), 2*ones(12,1) 2*ones(12,1) ];

    case 8
          msh = quadriga_lib.cube( [3, 3, 5], [0,0,0*pi/180], [0,0,0] );
        mtl_ind = [ 1*ones(12,1) ];

end

[ vert, face ] = quadriga_lib.obj_file_write( '', msh );

% Receive positions
scl = 10;
res = 0.05;
x = -scl : res : scl;
y = -scl : res : scl;
[X,Y] = meshgrid(x,y);
dest = [ X(:), Y(:),  ones(numel(X),1)*0 ];

% Origin
orig = [ -2, 8, 0.1 ];
orig = repmat( orig, size(dest,1), 1 );

tic
[ gain, xprmat ]= quadriga_lib.calc_diffraction_gain( orig, dest, msh, mtl_ind, mtl_prop, 1e9, lod, 1, [],0,0,0,slab_threshold );
toc
[ gainX, xprmatX ] = quadriga_lib.calc_diffraction_gain( orig(1,:), probe, msh, mtl_ind, mtl_prop, 1e9, lod,2, [],0,0,0,slab_threshold );

ang = angle(xprmat(1,:) + 1j* xprmat(2,:))*180/pi;
ang = reshape(ang,numel(y),[]);

gain = reshape(gain,numel(y),[]);

han = figure('Position',[ 100 , 2000 , 1000 , 700]);
title('Gain')
patch ("Faces", face, "Vertices", vert, 'FaceColor', [0.7;0.7;0.7], 'EdgeColor', [0.3;0.3;0.3]/2  );
hold on
plot3(orig(1,1),orig(1,2),orig(1,3),'+r','Markersize',12)
plot3(probe(1),probe(2),probe(3),'xr','Markersize',12)
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

han = figure('Position',[ 1100 , 2000 , 1000 , 700]);
title('Gain')
patch ("Faces", face, "Vertices", vert, 'FaceColor', [0.7;0.7;0.7], 'EdgeColor', [0.3;0.3;0.3]/2  );
hold on
plot3(orig(1,1),orig(1,2),orig(1,3),'+r','Markersize',12)
plot3(probe(1),probe(2),probe(3),'xr','Markersize',12)
alpha(0.0)
axis([-1 1 -1 1 -1 1]*scl)
grid on
imagesc(x,y,ang);
hold off
view([0,90])
clim([-180,180])
axis equal
colorbar

%print(['work_in_progress.png'],'-dpng');

ii = find(abs(y-probe(2))<0.001,1);
han = figure('Position',[ 100 ,600 , 1000 , 700]);
plot(x,gain(ii,:))

han = figure('Position',[ 1100 ,600 , 1000 , 700]);
plot(x,ang(ii,:))


[x(1:10:end);gain(ii,1:10:end)]