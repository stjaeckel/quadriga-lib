function test_obj_file_write

fn = 'cube.obj';
mtl_fn = 'cube.mtl';
csv_fn = 'custom_materials.csv';
csv_obj_fn = 'cube.csv';   % companion CSV written next to the .obj
if exist(fn,'file');         delete(fn);         end
if exist(mtl_fn,'file');     delete(mtl_fn);     end
if exist(csv_fn,'file');     delete(csv_fn);     end
if exist(csv_obj_fn,'file'); delete(csv_obj_fn); end

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

[ mesh_rd, vert_list_rd, face_ind_rd, obj_ind_rd, obj_names_rd, mtl_ind_rd, mtl_names_rd, ~, ...
    csv_ind_rd ] = quadriga_lib.obj_file_read( fn );

assertEqual( size(vert_list_rd), [8,3] );
assertElementsAlmostEqual( mesh_rd, mesh, 'absolute', 1e-12 );
assertElementsAlmostEqual( mk(vert_list_rd, face_ind_rd), mesh, 'absolute', 1e-12 );
assertEqual( size(obj_names_rd), [1,1] );
assertEqual( obj_names_rd{1,1}, 'object' );
% No usemtl written -> no materials on read-back (no synthetic "default")
assertTrue( isempty(mtl_names_rd) );
assertTrue( all( obj_ind_rd == 1 ) );
assertTrue( all( mtl_ind_rd == 0 ) );   % no material
assertTrue( all( csv_ind_rd == 0 ) );   % no material
delete(fn);

%% vert_list / face_ind round-trip
[ vlo, fio ] = quadriga_lib.obj_file_write( fn, [], [], [], [], [], vert_list, face_ind );

% In this mode the outputs are exact copies of the inputs
assertElementsAlmostEqual( vlo, vert_list, 'absolute', 1e-14 );
assertEqual( fio, uint64(face_ind) );

[ ~, vert_list_rd, face_ind_rd ] = quadriga_lib.obj_file_read( fn );
assertElementsAlmostEqual( mk(vert_list_rd, face_ind_rd), mesh, 'absolute', 1e-12 );
delete(fn);

%% Materials round-trip (named ITU materials)
obj_ind = ones(12,1);
mtl_ind = ones(12,1); mtl_ind(5:12) = 2;   % faces 1-4 concrete, 5-12 wood
obj_names = { 'Cube' };
mtl_names = { 'itu_concrete', 'itu_wood' };

quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names );
assertTrue( exist(mtl_fn,'file') == 2 );

% Resolve EM properties from the built-in default table (names are ITU materials)
[ ~, ~, ~, ~, ~, mtl_ind_rd, mtl_names_rd, ~, csv_ind_rd, ~, csv_prop_rd ] = ...
    quadriga_lib.obj_file_read( fn );

assertEqual( size(mtl_names_rd), [2,1] );
assertEqual( mtl_names_rd{1,1}, 'itu_concrete' );
assertEqual( mtl_names_rd{2,1}, 'itu_wood' );

assertElementsAlmostEqual( prop_at(csv_prop_rd, 'a', csv_ind_rd(1)), 5.24, 'absolute', 1e-2 );
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'a', csv_ind_rd(5)), 1.99, 'absolute', 1e-2 );
assertEqual( mtl_ind_rd, uint64([1;1;1;1;2;2;2;2;2;2;2;2]) );
delete(fn); delete(mtl_fn);

%% Materials round-trip (custom material via CSV)
obj_ind = ones(12,1);
mtl_ind = ones(12,1);
obj_names = { 'Cube' };
mtl_names = { 'glass' };

quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names );

% EM properties come from a CSV, not from the OBJ/MTL
f = fopen(csv_fn,'w');
fprintf(f,'%s\n','name,a,b,c,d,att');
fprintf(f,'%s\n','air,1.0,0.0,0.0,0.0,0.0');
fprintf(f,'%s\n','glass,6.0,0.0,0.1,1.2,0.0');
fclose(f);

[ ~, ~, ~, ~, ~, mtl_ind_rd, mtl_names_rd, ~, csv_ind_rd, ~, csv_prop_rd ] = ...
    quadriga_lib.obj_file_read( fn, csv_fn );

assertEqual( mtl_names_rd{1,1}, 'glass' );
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'a', csv_ind_rd(1)), 6.0, 'absolute', 1e-12 );
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'c', csv_ind_rd(1)), 0.1, 'absolute', 1e-12 );
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'd', csv_ind_rd(1)), 1.2, 'absolute', 1e-12 );
assertTrue( all( mtl_ind_rd == 1 ) );
delete(fn); delete(mtl_fn); delete(csv_fn);

%% BSDF round-trip
obj_ind = ones(12,1);
mtl_ind = ones(12,1);
obj_names = { 'Cube' };
mtl_names = { 'painted' };

% Distinct non-default values; clamped fields inside [0, 1], ior in a sane range
bsdf = [ 0.1 0.2 0.3, 0.7, 0.4, 0.6, 1.7, 0.8, 0.05 0.15 0.25, 0.3, 0.35, 0.45, 0.55, 0.65, 0.9 ];

quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, [], [], bsdf );

[ ~, ~, ~, ~, ~, ~, mtl_names_rd, bsdf_rd ] = quadriga_lib.obj_file_read( fn );

assertEqual( mtl_names_rd{1,1}, 'painted' );
assertEqual( size(bsdf_rd), [1,17] );
assertElementsAlmostEqual( bsdf_rd, bsdf, 'absolute', 1e-9 );
delete(fn); delete(mtl_fn);

%% CSV material table round-trip
obj_ind = ones(12,1);
mtl_ind = ones(12,1); mtl_ind(5:12) = 2;   % faces 1-4 concrete, 5-12 wood
obj_names = { 'Cube' };
mtl_names = { 'concrete', 'wood' };          % usemtl names must match csv_names to resolve on read-back

csv_ind = ones(12,1); csv_ind(5:12) = 2;     % 1-based, same split as mtl_ind
csv_names = { 'concrete', 'wood' };
csv_prop = struct();
csv_prop.a    = [ 5.24; 1.99 ];
csv_prop.c    = [ 0.0462; 0.0047 ];
csv_prop.d    = [ 0.7822; 1.0718 ];
csv_prop.fRef = [ 1.0; 1.0 ];

% threshold default (0.001), csv_write_defaults = false
quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, ...
    [], [], [], 0.001, csv_ind, csv_names, csv_prop, false );

assertTrue( exist(fn,'file') == 2 );
assertTrue( exist(mtl_fn,'file') == 2 );
assertTrue( exist(csv_obj_fn,'file') == 2 );   % companion .csv named after the .obj

[ ~, ~, ~, ~, ~, mtl_ind_rd, ~, ~, csv_ind_rd, csv_names_rd, csv_prop_rd ] = ...
    quadriga_lib.obj_file_read( fn, csv_obj_fn );

assertEqual( numel(csv_names_rd), 2 );
assertEqual( csv_names_rd{1,1}, 'concrete' );
assertEqual( csv_names_rd{2,1}, 'wood' );

% csv_ind is 1-based and indexes csv_prop directly (no -1)
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'a', csv_ind_rd(1)), 5.24,   'absolute', 1e-2 );
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'a', csv_ind_rd(5)), 1.99,   'absolute', 1e-2 );
assertElementsAlmostEqual( prop_at(csv_prop_rd, 'd', csv_ind_rd(1)), 0.7822, 'absolute', 1e-3 );

% Visual side round-trips 1-based
assertEqual( mtl_ind_rd, uint64([1;1;1;1;2;2;2;2;2;2;2;2]) );
delete(fn); delete(mtl_fn); delete(csv_obj_fn);

%% CSV columns, defaults and validation
obj_ind = ones(12,1);
mtl_ind = ones(12,1);
obj_names = { 'Cube' };
mtl_names = { 'slab' };

csv_names = { 'slab' };
csv_prop = struct();
csv_prop.c   = 0.05;   % canonical, present
csv_prop.tf  = 2.0;    % canonical, present
csv_prop.zzz = 7.0;    % extra (non-canonical)

% csv_write_defaults = false -> only present columns (canonical order, then extras)
quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, ...
    [], [], [], 0.001, [], csv_names, csv_prop, false );
assertTrue( exist(csv_obj_fn,'file') == 2 );

lines = read_lines(csv_obj_fn);
assertEqual( lines{1}, 'name,c,tf,zzz' );
assertEqual( lines{2}, 'slab,0.05,2,7' );
delete(fn); delete(mtl_fn); delete(csv_obj_fn);

% csv_write_defaults = true -> full canonical set with defaults (a, e, fRef = 1, else 0)
quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, ...
    [], [], [], 0.001, [], csv_names, csv_prop, true );
lines = read_lines(csv_obj_fn);
assertEqual( lines{1}, 'name,a,b,c,d,e,f,g,h,att,attB,alpha,alphaB,fRef,m,resF,resQ,resS,coiF,coiQ,coiA,tf,tfB,zzz' );
assertEqual( lines{2}, 'slab,1,0,0.05,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,2,0,7' );
delete(fn); delete(mtl_fn); delete(csv_obj_fn);

% Validation: csv_prop column length must match numel(csv_names)
try
    bad_prop = struct('a', [1.0; 2.0]);   % 2 values, 1 material
    quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, ...
        [], [], [], 0.001, [], csv_names, bad_prop, false );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: csv_prop column length mismatch.');
    end
end

% Validation: csv_ind out of range (only 1 material in csv_names)
try
    csv_ind_bad = ones(12,1); csv_ind_bad(1) = 5;
    quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, ...
        [], [], [], 0.001, csv_ind_bad, csv_names, csv_prop, false );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: csv_ind out of range.');
    end
end

% Validation: csv inputs without csv_names
try
    quadriga_lib.obj_file_write( fn, mesh, obj_ind, mtl_ind, obj_names, mtl_names, ...
        [], [], [], 0.001, [], [], csv_prop, false );
    error('moxunit:exceptionNotRaised', 'Expected an error!');
catch ME
    if strcmp(ME.identifier, 'moxunit:exceptionNotRaised')
        error('moxunit:exceptionNotRaised', 'Expected error: csv inputs without csv_names.');
    end
end
if exist(fn,'file');         delete(fn);         end
if exist(mtl_fn,'file');     delete(mtl_fn);     end
if exist(csv_obj_fn,'file'); delete(csv_obj_fn); end

%% Multiple objects
meshA = mesh;
meshB = mesh;
meshB(:,[1 4 7]) = meshB(:,[1 4 7]) + 10;   % shift x of all three corners -> disjoint cube
mesh2 = [ meshA; meshB ];                    % [24, 9]
obj_ind = [ ones(12,1); 2*ones(12,1) ];
obj_names = { 'CubeA', 'CubeB' };

[ vlo, ~ ] = quadriga_lib.obj_file_write( fn, mesh2, obj_ind, [], obj_names );
assertEqual( size(vlo), [16,3] );   % no cross-object merging -> 8 + 8

[ ~, vert_list_rd, face_ind_rd, obj_ind_rd, obj_names_rd ] = quadriga_lib.obj_file_read( fn );

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
    quadriga_lib.obj_file_write( fn, mesh, [], [], [], [], [], [], [], 0.001, [], [], [], false, 1 );
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

if exist(fn,'file');         delete(fn);         end
if exist(mtl_fn,'file');     delete(mtl_fn);     end
if exist(csv_fn,'file');     delete(csv_fn);     end
if exist(csv_obj_fn,'file'); delete(csv_obj_fn); end

end

% Read a property value for 1-based material index iM from struct field 'key',
% applying the documented per-column default when the field is absent.
function v = prop_at(s, key, iM)
defaults = struct('a',1.0,'b',0.0,'c',0.0,'d',0.0,'att',0.0, ...
                  'attB',0.0,'alpha',0.0,'alphaB',0.0,'fRef',1.0);
if isfield(s, key)
    col = s.(key);
    v = col(iM);
elseif isfield(defaults, key)
    v = defaults.(key);
else
    v = 0.0;
end
end

% Read all lines of a text file into a cell array (newlines stripped)
function lines = read_lines(fn)
lines = {};
f = fopen(fn,'r');
while true
    l = fgetl(f);
    if ~ischar(l); break; end
    lines{end+1,1} = l; %#ok<AGROW>
end
fclose(f);
end