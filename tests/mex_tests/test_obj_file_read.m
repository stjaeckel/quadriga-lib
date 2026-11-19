% SPDX-License-Identifier: Apache-2.0
% Copyright (C) 2022-2026 Stephan Jaeckel (http://quadriga-lib.org)
% Part of quadriga-lib — see LICENSE for terms.

function test_obj_file_read
% MEX interface test for quadriga_lib.obj_file_read.
% Physics validation lives in the Catch2 suite (test_obj_file_read.cpp).
fn = 'cube.obj';
if exist( fn,'file' )
    delete(fn);
end

f = fopen( fn,'w' );
fprintf(f,'%s\n','# A very nice, but useless comment ;-)');
fprintf(f,'%s\n','o Cube');
fprintf(f,'%s\n','v 1.0 1.0 1.0');
fprintf(f,'%s\n','v 1.0 1.0 -1.0');
fprintf(f,'%s\n','v 1.0 -1.0 1.0');
fprintf(f,'%s\n','v 1.0 -1.0 -1.0');
fprintf(f,'%s\n','v -1.0 1.0 1.0');
fprintf(f,'%s\n','v -1.0 1.0 -1.0');
fprintf(f,'%s\n','v -1.0 -1.0 1.0');
fprintf(f,'%s\n','v -1.0 -1.0 -1.0');
fprintf(f,'%s\n','s 0');
fprintf(f,'%s\n','f 5 3 1');
fprintf(f,'%s\n','f 3 8 4');
fprintf(f,'%s\n','f 7 6 8');
fprintf(f,'%s\n','f 2 8 6');
fprintf(f,'%s\n','f 1 4 2');
fprintf(f,'%s\n','f 5 2 6');
fprintf(f,'%s\n','f 5 7 3');
fprintf(f,'%s\n','f 3 7 8');
fprintf(f,'%s\n','f 7 5 6');
fprintf(f,'%s\n','f 2 4 8');
fprintf(f,'%s\n','f 1 3 4');
fprintf(f,'%s\n','f 5 1 2');
fclose(f);

vert_list_correct = [1.0 1.0 1.0; 1.0 1.0 -1.0; 1.0 -1.0 1.0; 1.0 -1.0 -1.0; -1.0 1.0 1.0; -1.0 1.0 -1.0'; -1.0 -1.0 1.0; -1.0 -1.0 -1.0 ];
face_ind_correct  = [ 5 3 1;  3 8 4; 7 6 8;  2 8 6;    1 4 2;  5 2 6;  5 7 3;  3 7 8;    7 5 6;  2 4 8;  1 3 4;  5 1 2 ];
mesh_correct = [ vert_list_correct( face_ind_correct(:,1),: ), vert_list_correct( face_ind_correct(:,2),: ), vert_list_correct( face_ind_correct(:,3),: ) ];

% No output should be fine
quadriga_lib.obj_file_read(fn);

% Read all 11 outputs
[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read(fn);

assertEqual( size(mesh), [12,9] );
assertEqual( size(vert_list), [8,3] );
assertEqual( size(face_ind), [12,3] );
assertEqual( size(obj_ind), [12,1] );
assertEqual( size(mtl_ind), [12,1] );
assertEqual( size(csv_ind), [12,1] );
assertEqual( size(obj_names), [1,1] );

assertTrue( isa(mesh, "double") );
assertTrue( isa(vert_list, "double") );
assertTrue( isa(face_ind, "uint64") );
assertTrue( isa(obj_ind, "uint64") );
assertTrue( isa(mtl_ind, "uint64") );
assertTrue( isa(csv_ind, "uint64") );
assertTrue( isstruct(csv_prop) );

assertElementsAlmostEqual( vert_list, vert_list_correct, 'absolute', 1e-14 );
assertEqual( face_ind, uint64(face_ind_correct) );
assertElementsAlmostEqual( mesh, mesh_correct, 'absolute', 1e-14 );

% Single object, 1-based
assertTrue( all( obj_ind == 1 ) );
assertEqual( obj_names{1,1}, 'Cube' );

% No usemtl in the file -> no materials assigned (no synthetic "default");
% csv side is still the full default table with air at row 1
assertTrue( isempty(mtl_names) );
assertTrue( numel(csv_names) > 1 );
assertEqual( csv_names{1,1}, 'air' );
assertTrue( all( mtl_ind == 0 ) );   % no material
assertTrue( all( csv_ind == 0 ) );   % no material
% Air at csv row 1 is transparent (a = 1)
assertElementsAlmostEqual( prop_at(csv_prop, 'a', 1), 1.0, 'absolute', 1e-14 );

% Two planes, second uses a named ITU material
delete(fn);

f = fopen( fn,'w' );
fprintf(f,'%s\n','o Plane');
fprintf(f,'%s\n','v -1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v -1.000000 1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 1.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 0.000000');
fprintf(f,'%s\n','vt 0.000000 1.000000');
fprintf(f,'%s\n','vt 0.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 1.000000');
fprintf(f,'%s\n','f 2/1 3/2 1/3');
fprintf(f,'%s\n','f 2/1 4/4 3/2');
fprintf(f,'%s\n','o Plane.001');
fprintf(f,'%s\n','v -1.000000 -1.000000 1.26');
fprintf(f,'%s\n','v 1.000000 -1.000000 1.26');
fprintf(f,'%s\n','v -1.000000 1.000000 1.26');
fprintf(f,'%s\n','v 1.000000 1.000000 1.26');
fprintf(f,'%s\n','vt 1.000000 0.000000');
fprintf(f,'%s\n','vt 0.000000 1.000000');
fprintf(f,'%s\n','vt 0.000000 0.000000');
fprintf(f,'%s\n','vt 1.000000 1.000000');
fprintf(f,'%s\n','usemtl itu_wood');
fprintf(f,'%s\n','f 6/5 7/6 5/7');
fprintf(f,'%s\n','f 6/5 8/8 7/6');
fclose(f);

% Read all
[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read(fn);

assertEqual( size(obj_names), [2,1] );

assertTrue( isa(mesh, "double") );
assertTrue( isa(vert_list, "double") );

% Faces 1-2 have no usemtl -> no material (csv_ind = 0); faces 3-4 are itu_wood (a = 1.99)
assertTrue( all( csv_ind(1:2) == 0 ) );
assertElementsAlmostEqual( prop_at(csv_prop, 'a', csv_ind(3)), 1.99, 'absolute', 1e-3 );
assertEqual( face_ind, uint64([2,3,1;2,4,3;6,7,5;6,8,7]) );
assertEqual( obj_ind, uint64([1;1;2;2]) );

% mtl_names: only itu_wood (faces 1-2 unassigned -> mtl_ind 0, faces 3-4 -> itu_wood)
assertEqual( mtl_names{1,1}, 'itu_wood' );
assertEqual( mtl_ind, uint64([0;0;1;1]) );

% Custom material via CSV
delete(fn);
csv_fn = 'custom_materials.csv';

f = fopen(csv_fn,'w');
fprintf(f,'%s\n','name,a,b,c,d,att');
fprintf(f,'%s\n','air,1.0,0.0,0.0,0.0,0.0');
fprintf(f,'%s\n','custom_material_1,2.5,0.0,0.001,0.5,5.0');
fprintf(f,'%s\n','custom_material_2,4.0,-0.1,0.05,1.2,10.0');
fclose(f);

f = fopen( fn,'w' );
fprintf(f,'%s\n','o Plane');
fprintf(f,'%s\n','v -1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 -1.000000 0.000000');
fprintf(f,'%s\n','v -1.000000 1.000000 0.000000');
fprintf(f,'%s\n','v 1.000000 1.000000 0.000000');
fprintf(f,'%s\n','usemtl custom_material_1');
fprintf(f,'%s\n','f 2 3 1');
fprintf(f,'%s\n','usemtl custom_material_2');
fprintf(f,'%s\n','f 2 4 3');
fclose(f);

[ mesh, vert_list, face_ind, obj_ind, obj_names, mtl_ind, mtl_names, bsdf, ...
    csv_ind, csv_names, csv_prop ] = quadriga_lib.obj_file_read(fn, csv_fn);

% Face 1 -> custom_material_1, face 2 -> custom_material_2
assertElementsAlmostEqual( prop_at(csv_prop, 'a',   csv_ind(1)), 2.5,   'absolute', 1e-14 );
assertElementsAlmostEqual( prop_at(csv_prop, 'c',   csv_ind(1)), 0.001, 'absolute', 1e-14 );
assertElementsAlmostEqual( prop_at(csv_prop, 'att', csv_ind(1)), 5.0,   'absolute', 1e-14 );

assertElementsAlmostEqual( prop_at(csv_prop, 'a',   csv_ind(2)), 4.0,  'absolute', 1e-14 );
assertElementsAlmostEqual( prop_at(csv_prop, 'b',   csv_ind(2)), -0.1, 'absolute', 1e-14 );
assertElementsAlmostEqual( prop_at(csv_prop, 'att', csv_ind(2)), 10.0, 'absolute', 1e-14 );

assertTrue( csv_ind(1) ~= csv_ind(2) );

assertEqual( mtl_names{1,1}, 'custom_material_1' );
assertEqual( mtl_names{2,1}, 'custom_material_2' );
assertEqual( obj_ind, uint64([1;1]) );
assertEqual( mtl_ind, uint64([1;2]) );
assertEqual( size(bsdf), [0,0] );

delete(csv_fn);

% CSV with a subset of columns; unspecified ones take per-column defaults
csv_fn = 'custom_materials.csv';
f = fopen(csv_fn,'w');
fprintf(f,'%s\n','name,a,c,fRef');
fprintf(f,'%s\n','air,1.0,0.0,1.0');
fprintf(f,'%s\n','wifi_wall,5.0,0.02,2.4');
fclose(f);

f = fopen(fn,'w');
fprintf(f,'%s\n','o Obj');
fprintf(f,'%s\n','v -1 -1 0');
fprintf(f,'%s\n','v  1 -1 0');
fprintf(f,'%s\n','v -1  1 0');
fprintf(f,'%s\n','usemtl wifi_wall');
fprintf(f,'%s\n','f 1 2 3');
fclose(f);

[ ~, ~, ~, ~, ~, ~, ~, ~, csv_ind, ~, csv_prop ] = quadriga_lib.obj_file_read(fn, csv_fn);

assertElementsAlmostEqual( prop_at(csv_prop, 'a',    csv_ind(1)), 5.0,  'absolute', 1e-14 );
assertElementsAlmostEqual( prop_at(csv_prop, 'c',    csv_ind(1)), 0.02, 'absolute', 1e-14 );
assertElementsAlmostEqual( prop_at(csv_prop, 'fRef', csv_ind(1)), 2.4,  'absolute', 1e-14 );
% Columns absent from the CSV are not fields of the struct
assertFalse( isfield(csv_prop, 'b') );
delete(fn);
delete(csv_fn);

% Unknown material, non-strict -> no material (csv_ind 0)
f = fopen(fn,'w');
fprintf(f,'%s\n','o Obj');
fprintf(f,'%s\n','v -1 -1 0');
fprintf(f,'%s\n','v  1 -1 0');
fprintf(f,'%s\n','v -1  1 0');
fprintf(f,'%s\n','usemtl not_a_real_material');
fprintf(f,'%s\n','f 1 2 3');
fclose(f);

[ ~, ~, ~, ~, ~, ~, mtl_names, ~, csv_ind ] = quadriga_lib.obj_file_read(fn, '', false);
assertEqual( mtl_names{1,1}, 'not_a_real_material' );  % raw name kept on .mtl side
assertTrue( all( csv_ind == 0 ) );                     % unmatched, non-strict -> no material

% Same scene, strict -> error (the table is consulted only when csv_ind is requested)
assert_throws( @() out9(fn, '', true), 'Material ''not_a_real_material'' referenced in' );
delete(fn);

% Recreate the simple cube for the argument-count error cases
f = fopen( fn,'w' );
fprintf(f,'%s\n','o Cube');
fprintf(f,'%s\n','v -1 -1 0');
fprintf(f,'%s\n','v  1 -1 0');
fprintf(f,'%s\n','v -1  1 0');
fprintf(f,'%s\n','f 1 2 3');
fclose(f);

% No file name
assert_throws( @() quadriga_lib.obj_file_read(), 'Wrong number of input arguments.' );

% An empty file name reads only the material table
[ mesh_e, ~, ~, ~, ~, ~, ~, ~, ~, csv_names_e ] = quadriga_lib.obj_file_read('');
assertTrue( isempty(mesh_e) );
assertTrue( numel(csv_names_e) > 0 );

% Too many inputs
assert_throws( @() quadriga_lib.obj_file_read(fn,'bla',true,99), 'Wrong number of input arguments.' );

% Too many outputs (12 requested, max 11)
assert_throws( @() out12(fn), 'Wrong number of output arguments.' );

% Wrong file name
assert_throws( @() quadriga_lib.obj_file_read('bla.obj'), 'Error opening file: ''bla.obj'' does not exist.' );

delete(fn);

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

% Request one output more than obj_file_read allows (11)
function out12(varargin)
[~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.obj_file_read( varargin{:} ); %#ok<ASGLU>
end

% Request the first 9 outputs (up to csv_ind)
function out9(varargin)
[~, ~, ~, ~, ~, ~, ~, ~, ~] = quadriga_lib.obj_file_read( varargin{:} ); %#ok<ASGLU>
end
