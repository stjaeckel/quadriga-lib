function test_obj_file_write

fn = 'cube.obj';
mtl_fn = 'cube.mtl';
if exist(fn,'file');     delete(fn);     end
if exist(mtl_fn,'file'); delete(mtl_fn); end

% Reconstruct a [n_face, 9] mesh from a vertex list and 1-based face indices
mk = @(V,F) [ V(F(:,1),:), V(F(:,2),:), V(F(:,3),:) ];

% A unit cube: 8 vertices, 12 triangular faces (1-based, same as the reader test)
vert_list = [ 1 1 1; 1 1 -1; 1 -1 1; 1 -1 -1; -1 1 1; -1 1 -1; -1 -1 1; -1 -1 -1 ];
face_ind  = [ 5 3 1; 3 8 4; 7 6 8; 2 8 6; 1 4 2; 5 2 6; 5 7 3; 3 7 8; 7 5 6; 2 4 8; 1 3 4; 5 1 2 ];
mesh = mk( vert_list, face_ind );

%% Mesh round-trip (geometry only)
[ vlo, fio ] = quadriga_lib.obj_file_write( fn, mesh );

assertEqual( size(vlo), [8,3] );
assertEqual( size(fio), [12,3] );
assertTrue( isa(vlo, 'double') );
assertTrue( isa(fio, 'uint64') );
assertElementsAlmostEqual( mk(vlo, fio), mesh, 'absolute', 1e-12 );
assertTrue( exist(mtl_fn,'file') ~= 2 );   % no materials -> no .mtl

[ mesh_rd, ~, vert_list_rd, face_ind_rd, obj_ind_rd, mtl_ind_rd, obj_names_rd, mtl_names_rd ] = ...
    quadriga_lib.obj_file_read( fn );

assertEqual( size(vert_list_rd), [8,3] );
assertElementsAlmostEqual( mesh_rd, mesh, 'absolute', 1e-12 );
assertElementsAlmostEqual( mk(vert_list_rd, face_ind_rd), mesh, 'absolute', 1e-12 );
assertEqual( size(obj_names_rd), [1,1] );
assertEqual( obj_names_rd{1,1}, 'object' );
assertTrue( isempty(mtl_names_rd) );
assertTrue( all( obj_ind_rd == 1 ) );
assertTrue( all( mtl_ind_rd == 0 ) );
delete(fn);

%% vert_list / face_ind round-trip
[ vlo, fio ] = quadriga_lib.obj_file_write( fn, [], [], [], [], [], vert_list, face_ind );

% In this mode the outputs are exact copies of the inputs
assertElementsAlmostEqual( vlo, vert_list, 'absolute', 1e-14 );
assertEqual( fio, uint64(face_ind) );

[ ~, ~, vert_list_rd, face_ind_rd ] = quadriga_lib.obj_file_read( fn );
assertElementsAlmostEqual( mk(vert_list_rd, face_ind_rd), mesh, 'absolute', 1e-12 );
delete(fn);

%% Materials round-trip (named ITU materials)
obj_ind = ones(12,1);
mtl_ind = ones(12,1); mtl_ind(5:12) = 2;   % faces 1-4 concrete, 5-12 wood
obj_names = { 'Cube' };
mtl_names = { 'itu_concrete', 'itu_wood' };

quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names );
assertTrue( exist(mtl_fn,'file') == 2 );

[ ~, mtl_prop, ~, ~, ~, mtl_ind_rd, ~, mtl_names_rd ] = quadriga_lib.obj_file_read( fn );

assertEqual( size(mtl_names_rd), [2,1] );
assertEqual( mtl_names_rd{1,1}, 'itu_concrete' );
assertEqual( mtl_names_rd{2,1}, 'itu_wood' );
assertElementsAlmostEqual( mtl_prop(1,:), [5.24, 0, 0.0462, 0.7822], 'absolute', 1e-12 );
assertElementsAlmostEqual( mtl_prop(5,:), [1.99, 0, 0.0047, 1.0718], 'absolute', 1e-12 );
assertEqual( mtl_ind_rd, uint64([1;1;1;1;2;2;2;2;2;2;2;2]) );
delete(fn); delete(mtl_fn);

%% Materials round-trip (custom inline :: syntax)
obj_ind = ones(12,1);
mtl_ind = ones(12,1);
obj_names = { 'Cube' };
mtl_names = { 'glass::6.0:0:0.1:1.2' };

quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names );

[ ~, mtl_prop, ~, ~, ~, mtl_ind_rd, ~, mtl_names_rd ] = quadriga_lib.obj_file_read( fn );

assertEqual( mtl_names_rd{1,1}, 'glass::6.0:0:0.1:1.2' );
assertElementsAlmostEqual( mtl_prop(1,:), [6.0, 0, 0.1, 1.2], 'absolute', 1e-12 );
assertTrue( all( mtl_ind_rd == 1 ) );
delete(fn); delete(mtl_fn);

%% BSDF round-trip
obj_ind = ones(12,1);
mtl_ind = ones(12,1);
obj_names = { 'Cube' };
mtl_names = { 'painted' };

% Distinct non-default values; clamped fields inside [0, 1], ior in a sane range
bsdf = [ 0.1 0.2 0.3, 0.7, 0.4, 0.6, 1.7, 0.8, 0.05 0.15 0.25, 0.3, 0.35, 0.45, 0.55, 0.65, 0.9 ];

quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, [], [], bsdf );

[ ~, ~, ~, ~, ~, ~, ~, mtl_names_rd, bsdf_rd ] = quadriga_lib.obj_file_read( fn );

assertEqual( mtl_names_rd{1,1}, 'painted' );
assertEqual( size(bsdf_rd), [1,17] );
assertElementsAlmostEqual( bsdf_rd, bsdf, 'absolute', 1e-9 );
delete(fn); delete(mtl_fn);

%% Multiple objects
meshA = mesh;
meshB = mesh;
meshB(:,[1 4 7]) = meshB(:,[1 4 7]) + 10;   % shift x of all three corners -> disjoint cube
mesh2 = [ meshA; meshB ];                    % [24, 9]
obj_ind = [ ones(12,1); 2*ones(12,1) ];
obj_names = { 'CubeA', 'CubeB' };

[ vlo, ~ ] = quadriga_lib.obj_file_write( fn, mesh2, obj_ind, [], obj_names );
assertEqual( size(vlo), [16,3] );   % no cross-object merging -> 8 + 8

[ ~, ~, vert_list_rd, face_ind_rd, obj_ind_rd, ~, obj_names_rd ] = quadriga_lib.obj_file_read( fn );

assertEqual( size(vert_list_rd), [16,3] );
assertEqual( size(obj_names_rd), [2,1] );
assertEqual( obj_names_rd{1,1}, 'CubeA' );
assertEqual( obj_names_rd{2,1}, 'CubeB' );
assertEqual( obj_ind_rd, uint64([ ones(12,1); 2*ones(12,1) ]) );
assertElementsAlmostEqual( mk(vert_list_rd, face_ind_rd), mesh2, 'absolute', 1e-12 );
delete(fn);

%% Outputs only (empty filename)
[ vlo, fio ] = quadriga_lib.obj_file_write( '', mesh );
assertEqual( size(vlo), [8,3] );
assertEqual( size(fio), [12,3] );
assertElementsAlmostEqual( mk(vlo, fio), mesh, 'absolute', 1e-12 );

%% Error handling - library validation (generic: any non-moxunit error is accepted)

% Both mesh and vert_list / face_ind given
try
    quadriga_lib.obj_file_write( fn, mesh, [], [], [], [], vert_list, face_ind );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: both mesh and vert_list/face_ind given.');
    end
end

% Neither geometry source given
try
    quadriga_lib.obj_file_write( fn );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: no geometry given.');
    end
end

% File name does not end in .obj
try
    quadriga_lib.obj_file_write( 'cube.txt', mesh );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: file name must end with .obj.');
    end
end

% Non-contiguous obj_ind: {1,1,2,2,1,...}
try
    obj_bad = ones(12,1); obj_bad(3) = 2; obj_bad(4) = 2;
    quadriga_lib.obj_file_write( fn, mesh, obj_bad, [], {'A','B'} );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: non-contiguous obj_ind.');
    end
end

% obj_names too short for obj_ind
try
    obj_ind = [ ones(6,1); 2*ones(6,1) ];
    quadriga_lib.obj_file_write( fn, mesh, obj_ind, [], {'OnlyOne'} );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: obj_names too short.');
    end
end

% mtl_names too short for mtl_ind
try
    mtl_ind = [ ones(6,1); 2*ones(6,1) ];
    quadriga_lib.obj_file_write( fn, mesh, [], mtl_ind, [], {'OnlyOne'} );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: mtl_names too short.');
    end
end

% bsdf given without mtl_ind / mtl_names
try
    quadriga_lib.obj_file_write( fn, mesh, [], [], [], [], [], [], zeros(1,17) );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: bsdf without materials.');
    end
end

%% Error handling - wrapper argument counts (specific messages)

% Too many inputs (> 10)
try
    quadriga_lib.obj_file_write( fn, mesh, [], [], [], [], [], [], [], 0.001, 1 );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of input arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% Too many outputs (> 2)
try
    [~,~,~] = quadriga_lib.obj_file_write( fn, mesh );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    expectedErrorMessage = 'Wrong number of output arguments.';
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised') || isempty(strfind(ME.message, expectedErrorMessage))
        error('moxunit:exceptionNotRaised', ['EXPECTED: "', expectedErrorMessage, '", GOT: "',ME.message,'"']);
    end
end

% None of the error cases should have produced a file
assertTrue( exist(fn,'file') ~= 2 );
assertTrue( exist('cube.txt','file') ~= 2 );

if exist(fn,'file');     delete(fn);     end
if exist(mtl_fn,'file'); delete(mtl_fn); end

end
